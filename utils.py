import os

import redis
import json

from datetime import datetime
from pathlib import Path

from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import  SemanticRouter
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import HFTextVectorizer

_APP_DIR = Path(__file__).resolve().parent


def resolve_env(var_name: str, default: str = "") -> str:
    """
    Read var_name from the environment. If the resulting value is itself the
    name of another environment variable that is currently set, use that
    variable's value instead of the literal value.

    This lets .env entries indirect to a real environment variable, e.g.
    REDIS_DB_PASSWORD=REDIS_PASSWORD looks up REDIS_PASSWORD first and falls
    back to the literal string "REDIS_PASSWORD" if it isn't set.
    """
    raw_value = os.getenv(var_name, default)
    if raw_value and raw_value in os.environ:
        return os.environ[raw_value]
    return raw_value


class RateLimiter:
    def __init__(self, redis_connection : redis.Redis, key_prefix="rag:limiter:", window=20):
        self.rds = redis_connection
        self.key_prefix = key_prefix
        self.window = window

    def is_allowed(self, user_key):
        allow_call = True

        key = f"{self.key_prefix}{user_key}"

        current_limit = self.rds.hincrby(key, "call_count", 1)
        self.rds.hexpire(key, self.window, "call_count", nx = True)

        user_limit = self.rds.hget(user_key, "call_limit")

        if current_limit > int(user_limit):
            allow_call = False

        return allow_call


class SemanticWrapper:
    def __init__(self, redis_connection : redis.Redis):

        #CONNECT TO REDIS DB
        if not redis_connection:
            self.rds = get_redis_connection()

        #LOAD EMBEDDING MODEL
        em = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2") 

        llm_router_file = os.getenv("LLM_ROUTER_FILE", "llm-router.json")
        llm_routes = load_dict_from_file(_APP_DIR / llm_router_file)

        #SEMANTIC ROUTER
        print("Setting Up Semantic Router...")

        # SemanticRouter's overwrite flag only recreates the index definition;
        # it does NOT drop existing route vectors (it calls index.create with
        # drop=False internally). So a route renamed/removed in
        # llm-router.json would still be indexed from a previous run and get
        # returned as a stale match, raising ValueError. Drop any existing
        # index and its data first so the router always reflects the current
        # file exactly.
        try:
            self.rds.ft("semantic:router").dropindex(delete_documents=True)
        except redis.exceptions.ResponseError:
            pass  # index did not exist yet

        self.semantic_router = SemanticRouter(
            name="semantic:router",
            vectorizer = em,
            routes = llm_routes,
            redis_client = self.rds,
            overwrite=False
        )

    

        #SEMANTIC CACHE
        print("Setting Up Semantic Cache...")

        # Entries are scoped per user and session via tag fields. An index
        # created before these fields existed can't filter on them, so drop it
        # (cache entries are short-lived anyway) and let it be recreated.
        try:
            existing_fields = [
                attr[1] for attr in self.rds.ft("semantic:cache").info()["attributes"]
            ]
            if not {"user_id", "session_id"} <= set(existing_fields):
                self.rds.ft("semantic:cache").dropindex(delete_documents=True)
        except redis.exceptions.ResponseError:
            pass  # index did not exist yet

        self.semantic_cache = SemanticCache(
            name="semantic:cache",
            redis_client=self.rds,
            ttl = 600,
            distance_threshold=0.2,
            filterable_fields=[
                {"name": "user_id", "type": "tag"},
                {"name": "session_id", "type": "tag"},
            ],
        )

        #FAQ KNOWLEDGE BASE (Redis LangCache)
        print("Loading FAQ Knowledge Base...")

        # Deferred import: lang_cache_rest imports resolve_env from this
        # module, so importing it at module scope here would create a
        # circular import.
        from lang_cache_rest import LangCacheRestClient

        self.lang_cache = LangCacheRestClient()
        knowledge_base_file = os.getenv("KNOWLEDGE_BASE_FILE", "knowledge-base.json")
        self._load_knowledge_base(_APP_DIR / knowledge_base_file)

    def _load_knowledge_base(self, file_path):
        """Load FAQ prompt/response pairs into LangCache, replacing any prior contents."""
        knowledge_base = load_dict_from_file(file_path)

        # Start from a clean slate on every run so edits to knowledge-base.json
        # (renamed/removed entries) don't leave stale entries behind.
        self.lang_cache.flush()

        for entry in knowledge_base:
            self.lang_cache.add_entry(
                prompt=entry["prompt"],
                response=entry["response"],
                attributes=entry.get("attributes"),
            )

    def get_faq_answer(self, user_query, similarity_threshold=None):
        if similarity_threshold is None:
            similarity_threshold = float(resolve_env("LANG_CACHE_SEMANTIC_THRESHOLD", "0.4"))
        matches = self.lang_cache.search(prompt=user_query, similarity_threshold=similarity_threshold)
        return matches[0]["response"] if matches else "NONE"

    def get_route(self, user_query):
        #CHECK THE ROUTE
        route_match = self.semantic_router(user_query, distance_threshold= 2.0)
        return route_match.name
    
    def get_similar(self, user_query, user_id, session_id):
        scope_filter = (Tag("user_id") == user_id) & (Tag("session_id") == session_id)
        cached_result_list = self.semantic_cache.check(
            prompt = user_query,
            filter_expression = scope_filter,
        )
        if cached_result_list:
            return cached_result_list[0]['response']

        return "NONE"

    def set_cache(self, query, response, user_id, session_id):
        # filters are part of the entry key, so the same prompt from a
        # different user/session is stored separately, not overwritten.
        self.semantic_cache.store(
                prompt= query,
                response= response,
                filters={"user_id": user_id, "session_id": session_id}
            )

    
    def clear_cache(self) :
        self.semantic_cache.clear()



def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_redis_connection():
    redis_db_host = resolve_env("REDIS_DB_HOST", "localhost")
    redis_db_port = resolve_env("REDIS_DB_PORT", "6379")
    redis_db_user = resolve_env("REDIS_DB_USER", "default")
    redis_db_password = resolve_env("REDIS_DB_PASSWORD", "")

    return redis.Redis(
        host=redis_db_host,
        port=int(redis_db_port),
        username=redis_db_user,
        password=redis_db_password,
        decode_responses=True,
    )


def time_difference(ts1, ts2):
    # Convert Unix timestamps to datetime objects
    dt1 = datetime.fromtimestamp(ts1)
    dt2 = datetime.fromtimestamp(ts2)
    delta = abs(dt2 - dt1)
    minutes, seconds = divmod(delta.total_seconds(), 60) 
    return f"{int(minutes)}:{int(seconds)}"

def main():
    redis_connection = get_redis_connection()

    rate_limiter = RateLimiter(redis_connection, "user:", 10)
    rate_limiter.is_allowed("jay")

if __name__ == "__main__":
    main()
