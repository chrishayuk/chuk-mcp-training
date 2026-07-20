"""Async HTTP client for the control plane's REST API."""

from __future__ import annotations

import asyncio
import os
from typing import Any, TypeVar

import httpx
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, TypeAdapter

from . import auth
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
            # Auth is per-request (see _auth_headers), never baked into the
            # client: concurrent HTTP callers each forward their own token.
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=HTTP_TIMEOUT_S,
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        """The bearer for this call: over HTTP, the calling agent's own token
        (never substituting our credentials for an anonymous caller — a
        tokenless call must 401 at the control plane); over stdio, the
        process's configured token."""
        if auth.http_request_active():
            token = auth.bearer_from_mcp_context()
        else:
            token = self._api_token
        return {"Authorization": f"{_BEARER_PREFIX}{token}"} if token else {}

    async def _request(self, method: str, path: str, *,
                       params: dict[str, Any] | None = None,
                       json: dict[str, Any] | None = None) -> Any:
        last: ControlPlaneError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._http().request(
                    method, path, params=params, json=json, headers=self._auth_headers()
                )
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

    async def post_params(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """POST with query params and no body (e.g. an action endpoint)."""
        return await self._request("POST", path, params=params)

    async def get_raw(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET returning the raw decoded JSON (no model validation)."""
        return await self._request("GET", path, params=params)

    async def delete_params(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """DELETE with query params (e.g. removing a keyed resource)."""
        return await self._request("DELETE", path, params=params)
