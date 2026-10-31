#!/usr/bin/env python3
"""
Thin synchronous REST client for the Redis LangCache service.

Used to power the FAQ knowledge base: FAQ prompt/response pairs are loaded
into a LangCache cache on startup, then queried the same way as any other
semantic cache lookup.

https://redis.io/docs/latest/develop/ai/langcache/api-reference/

Required environment variables:
- LANG_CACHE_SERVICE_URL: base endpoint for the LangCache service (must include https://)
- LANG_CACHE_CACHE_ID: the cache ID (shown on the service's Configuration tab)
- LANG_CACHE_API_KEY: the API key generated for the service
"""

from typing import Any

import httpx

from utils import resolve_env


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


class LangCacheRestClient:
    """Minimal synchronous wrapper around the Redis LangCache REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        cache_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or resolve_env("LANG_CACHE_SERVICE_URL", "")).rstrip("/")
        self.cache_id = cache_id or resolve_env("LANG_CACHE_CACHE_ID", "")
        self.api_key = api_key or resolve_env("LANG_CACHE_API_KEY", "")

        if not self.base_url or not self.cache_id or not self.api_key:
            raise ValueError(
                "LANG_CACHE_SERVICE_URL, LANG_CACHE_CACHE_ID, and LANG_CACHE_API_KEY "
                "must all be set to use the Redis LangCache REST API. See .env."
            )

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def _cache_path(self, suffix: str) -> str:
        return f"/v1/caches/{self.cache_id}{suffix}"

    def close(self) -> None:
        self._client.close()

    def search(
        self,
        prompt: str,
        similarity_threshold: float | None = None,
        attributes: dict[str, Any] | None = None,
        search_strategies: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"prompt": prompt}
        if similarity_threshold is not None:
            payload["similarityThreshold"] = similarity_threshold
        if attributes:
            payload["attributes"] = attributes
        if search_strategies:
            payload["searchStrategies"] = search_strategies

        resp = self._client.post(self._cache_path("/entries/search"), json=payload)
        _raise_for_status(resp)
        return resp.json().get("data", [])

    def add_entry(
        self,
        prompt: str,
        response: str,
        attributes: dict[str, Any] | None = None,
        ttl_millis: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {"prompt": prompt, "response": response}
        if attributes:
            payload["attributes"] = attributes
        if ttl_millis is not None:
            payload["ttlMillis"] = ttl_millis

        resp = self._client.post(self._cache_path("/entries"), json=payload)
        _raise_for_status(resp)
        return resp.json().get("entryId", "")

    def delete_entries(self, attributes: dict[str, Any]) -> int:
        resp = self._client.request(
            "DELETE",
            self._cache_path("/entries"),
            json={"attributes": attributes},
        )
        _raise_for_status(resp)
        return resp.json().get("deletedEntriesCount", 0)

    def delete_entry(self, entry_id: str) -> None:
        resp = self._client.delete(self._cache_path(f"/entries/{entry_id}"))
        if resp.status_code != 404:
            _raise_for_status(resp)

    def flush(self) -> None:
        resp = self._client.post(self._cache_path("/flush"))
        if resp.status_code != 404:
            _raise_for_status(resp)
