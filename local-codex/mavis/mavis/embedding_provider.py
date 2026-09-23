"""Opt-in adapter for an already loaded embedding model on Mavis's oMLX server."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urlparse

from .embedding_index import EmbeddingIdentity
from .maintenance_runtime import host_admission
from .runtime import (RuntimeConfig, inventory, mavis_generation_lease,
                      owns_running_server, request_json)


INDEX_VERSION = "mavis-code-passage-8192-v1"
DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1"


@dataclass
class LocalEmbeddingProvider:
    """Use only a loaded, named Mavis embedding model; never start or load one."""

    identity: EmbeddingIdentity
    home: Path
    endpoint: str = DEFAULT_ENDPOINT
    iris_endpoint: str = "http://127.0.0.1:8000/v1"
    inventory_reader: Callable[[str], list[dict[str, Any]]] = inventory
    request: Callable[..., Any] = request_json
    ownership_check: Callable[[RuntimeConfig], bool] = owns_running_server
    compute_admission: Callable[..., tuple[bool, str]] = host_admission

    def __post_init__(self) -> None:
        self.identity.validate()
        parsed = urlparse(self.endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.port != 8001 or parsed.path != "/v1"
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("embedding endpoint must be Mavis loopback port 8001 /v1")

    def ready(self, *, lease_held: bool = False) -> None:
        config = RuntimeConfig(home=self.home, endpoint=self.endpoint)
        if not self.ownership_check(config):
            raise RuntimeError("Mavis does not own the embedding endpoint")
        allowed, reason = self.compute_admission(
            self.home, iris_endpoint=self.iris_endpoint, lease_held=lease_held)
        if not allowed:
            raise RuntimeError(f"embedding compute admission refused: {reason}")
        rows = self.inventory_reader(self.endpoint)
        if any(row.get("loaded") is True and row.get("model_type") != "embedding"
               for row in rows):
            raise RuntimeError("Mavis has a loaded generation model")
        selected = [row for row in rows if row.get("id") == self.identity.model]
        if len(selected) != 1 or selected[0].get("loaded") is not True or (
            selected[0].get("model_type") != "embedding"
            or selected[0].get("engine_type") != "embedding"
        ):
            raise RuntimeError("selected embedding model is not loaded on Mavis")
        model_path = selected[0].get("model_path")
        if not isinstance(model_path, str) or not model_path:
            raise RuntimeError("embedding inventory lacks a model path")
        path = Path(model_path)
        observed_revision = (
            path.name if path.parent.name == "snapshots"
            and re.fullmatch(r"[0-9a-f]{40}", path.name)
            else selected[0].get("revision")
        )
        if observed_revision != self.identity.revision:
            raise RuntimeError("embedding model revision is unverified or changed")

    def _embed(self, text: str) -> list[float]:
        # The selected Qwen 0.6B candidate produced non-finite output when
        # padded with another input. One input per request preserves that gate.
        with mavis_generation_lease(RuntimeConfig(home=self.home, endpoint=self.endpoint),
                                    purpose="embedding-request"):
            self.ready(lease_held=True)
            payload = self.request(
                self.endpoint, "/v1/embeddings", method="POST", timeout=120,
                headers={"X-OMLX-Require-Loaded": "true"},
                payload={"model": self.identity.model, "input": text,
                         "encoding_format": "float"},
            )
        if (not isinstance(payload, dict) or payload.get("model") != self.identity.model
                or not isinstance(payload.get("data"), list)
                or len(payload["data"]) != 1 or not isinstance(payload["data"][0], dict)
                or payload["data"][0].get("index") != 0
                or not isinstance(payload["data"][0].get("embedding"), list)):
            raise RuntimeError("Mavis returned an invalid embedding response")
        return payload["data"][0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)
