#!/usr/bin/env python3
"""
Thin async REST client for the Redis Agent Memory cloud service.

Replaces the self-managed `agent_memory_client` SDK with direct calls to the
Agent Memory REST API:
https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/api-reference/

Required environment variables:
- AGENT_MEMORY_SERVICE_URL: base endpoint for the Agent Memory service (must include https://)
- AGENT_MEMORY_STORE_ID: the store instance ID (shown on the service's Configuration tab)
- AGENT_MEMORY_API_KEY: the API key generated for the service
"""

import re
from datetime import UTC, datetime
from typing import Any

import httpx

from utils import resolve_env

# sessionId, actorId, ownerId, and namespace are all restricted by the API to
# this shape: 1-64 chars, letters/digits/dashes only (no underscores, colons, etc).
_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9-]")


def sanitize_id(value: str, max_length: int = 64) -> str:
    """Coerce a value into the [a-zA-Z0-9-]{1,64} shape the API requires for IDs."""
    return _INVALID_ID_CHARS.sub("-", value)[:max_length]


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _raise_for_status(resp: httpx.Response) -> None:
    """Like resp.raise_for_status(), but includes the response body in the error message."""
    if resp.is_error:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        message = (
            f"{resp.status_code} {resp.reason_phrase} for {resp.request.method} "
            f"{resp.request.url}: {detail}"
        )
        raise httpx.HTTPStatusError(message, request=resp.request, response=resp)


class AgentMemoryRestClient:
    """Minimal async wrapper around the Redis Agent Memory REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        store_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or resolve_env("AGENT_MEMORY_SERVICE_URL", "")).rstrip("/")
        self.store_id = store_id or resolve_env("AGENT_MEMORY_STORE_ID", "")
        self.api_key = api_key or resolve_env("AGENT_MEMORY_API_KEY", "")

        if not self.base_url or not self.store_id or not self.api_key:
            raise ValueError(
                "AGENT_MEMORY_SERVICE_URL, AGENT_MEMORY_STORE_ID, and AGENT_MEMORY_API_KEY "
                "must all be set to use the Redis Agent Memory cloud REST API. See .env."
            )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def _store_path(self, suffix: str) -> str:
        return f"/v1/stores/{self.store_id}{suffix}"

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        resp = await self._client.get("/health")
        _raise_for_status(resp)
        return resp.json()

    # ---- Session memory ----

    async def add_session_event(
        self,
        session_id: str,
        actor_id: str,
        role: str,
        text: str,
        namespace: str | None = None,
        created_at: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Append a single event to a session. Creates the session if needed."""
        payload: dict[str, Any] = {
            "sessionId": sanitize_id(session_id),
            "actorId": sanitize_id(actor_id),
            "role": role.upper(),
            "content": [{"text": text}],
            "createdAt": created_at or _utc_now_iso(),
        }
        if namespace:
            payload["namespace"] = sanitize_id(namespace)
        if metadata is not None:
            payload["metadata"] = metadata

        resp = await self._client.post(self._store_path("/session-memory/events"), json=payload)
        _raise_for_status(resp)
        return resp.json()

    async def get_session_memory(self, session_id: str) -> dict[str, Any]:
        """Return session events and summary. Returns an empty session if not found yet."""
        session_id = sanitize_id(session_id)
        resp = await self._client.get(self._store_path(f"/session-memory/{session_id}"))
        if resp.status_code == 404:
            return {"sessionId": session_id, "events": [], "summary": None}
        _raise_for_status(resp)
        return resp.json()

    async def delete_session_memory(self, session_id: str) -> None:
        resp = await self._client.delete(self._store_path(f"/session-memory/{sanitize_id(session_id)}"))
        if resp.status_code not in (204, 404):
            _raise_for_status(resp)

    # ---- Long-term memory ----

    async def search_long_term_memory(
        self,
        text: str,
        owner_id: str | None = None,
        namespace: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
        exclude_memory_types: tuple[str, ...] | None = ("episodic", "semantic"),
    ) -> list[dict[str, Any]]:
        filter_conditions: dict[str, Any] = {}
        if owner_id:
            filter_conditions["ownerId"] = {"eq": sanitize_id(owner_id)}
        if namespace:
            filter_conditions["namespace"] = {"eq": sanitize_id(namespace)}
        if memory_type:
            filter_conditions["memoryType"] = {"eq": memory_type}

        payload: dict[str, Any] = {"text": text, "limit": limit}
        if filter_conditions:
            payload["filter"] = filter_conditions
        if similarity_threshold is not None:
            payload["similarityThreshold"] = similarity_threshold

        resp = await self._client.post(self._store_path("/long-term-memory/search"), json=payload)
        _raise_for_status(resp)
        items = resp.json().get("items", [])

        # The API's memoryType filter only supports eq/ne/in/all - no "not in" -
        # so excluding multiple built-in types has to happen client-side. Only
        # applies when the caller didn't already ask for one specific type.
        # Note: `limit` is applied server-side before this, so fewer than
        # `limit` items may come back once exclusions are applied.
        if not memory_type and exclude_memory_types:
            items = [item for item in items if item.get("memoryType") not in exclude_memory_types]

        return items

    async def bulk_create_long_term_memories(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        for memory in memories:
            for field in ("ownerId", "namespace", "sessionId"):
                if memory.get(field):
                    memory[field] = sanitize_id(memory[field])

        resp = await self._client.post(self._store_path("/long-term-memory"), json={"memories": memories})
        _raise_for_status(resp)
        return resp.json()

    async def bulk_delete_long_term_memories(self, memory_ids: list[str]) -> dict[str, Any]:
        resp = await self._client.request(
            "DELETE",
            self._store_path("/long-term-memory"),
            json={"memoryIds": memory_ids},
        )
        _raise_for_status(resp)
        return resp.json()


async def create_agent_memory_client(**kwargs: Any) -> AgentMemoryRestClient:
    return AgentMemoryRestClient(**kwargs)
