"""Async HTTP client for the control plane's REST API."""

from __future__ import annotations

import asyncio
import os
from typing import Any, TypeVar

import httpx
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, TypeAdapter

from .constants import DEFAULT_CP_URL, ENV_API_TOKEN, ENV_CP_URL, HTTP_TIMEOUT_S

# Retry transient failures — connection errors, timeouts, 5xx — with backoff.
# The shared Fly machine occasionally errors briefly (e.g. during checkpoint
# ingest); 4xx (bad token, not found) are not retried.
_MAX_ATTEMPTS = 4
_RETRY_BACKOFF_S = 0.25

# Load .env (searched upward from cwd) for local runs; harmless if none exists
# or the vars are already set in the environment (existing vars win).
load_dotenv(find_dotenv(usecwd=True))

M = TypeVar("M", bound=BaseModel)

_BEARER_PREFIX = "Bearer "


class ControlPlaneError(Exception):
    """Raised for transport failures and non-2xx responses."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ControlPlaneClient:
    def __init__(self, base_url: str | None = None, api_token: str | None = None) -> None:
        self._base_url = (base_url or os.environ.get(ENV_CP_URL, DEFAULT_CP_URL)).rstrip("/")
        self._api_token = api_token or os.environ.get(ENV_API_TOKEN, "")
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"{_BEARER_PREFIX}{self._api_token}"},
                timeout=HTTP_TIMEOUT_S,
            )
        return self._client

    async def _request(self, method: str, path: str, *,
                       params: dict[str, Any] | None = None,
                       json: dict[str, Any] | None = None) -> Any:
        last: ControlPlaneError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._http().request(method, path, params=params, json=json)
            except httpx.HTTPError as exc:
                last = ControlPlaneError("request_failed", repr(exc))
            else:
                if response.status_code < 400:
                    return response.json()
                error = ControlPlaneError(f"http_{response.status_code}", response.text[:500])
                if response.status_code < 500:
                    raise error  # client error — retrying won't help
                last = error
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
        raise last  # exhausted retries on a transient failure

    async def get_model(self, path: str, model: type[M],
                        params: dict[str, Any] | None = None) -> M:
        return model.model_validate(await self._request("GET", path, params=params))

    async def get_list(self, path: str, model: type[M],
                       params: dict[str, Any] | None = None) -> list[M]:
        adapter: TypeAdapter[list[M]] = TypeAdapter(list[model])  # type: ignore[valid-type]
        return adapter.validate_python(await self._request("GET", path, params=params))

    async def post_model(self, path: str, body: BaseModel, model: type[M]) -> M:
        payload = await self._request("POST", path, json=body.model_dump(mode="json"))
        return model.model_validate(payload)

    async def post_raw(self, path: str, body: BaseModel) -> dict[str, Any]:
        return await self._request("POST", path, json=body.model_dump(mode="json"))
