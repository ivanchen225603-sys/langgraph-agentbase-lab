import ipaddress
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentbase.http import AgentHttpLimits


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAB_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        allow_inf_nan=False,
    )
    mode: Literal["local", "openai"] = "local"
    host: str = "127.0.0.1"
    port: int = Field(8001, ge=1024, le=65535)
    data_dir: Path = Path(".runtime")
    knowledge_root: Path = Path("data/knowledge")
    auth_mode: Literal["local", "bearer"] = "local"
    bearer_principals: dict[str, str] = Field(default_factory=dict, exclude=True)
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(384, ge=64, le=3072)
    openai_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY", exclude=True)
    max_file_bytes: int = Field(262144, ge=1, le=262144)
    max_documents: int = Field(100, ge=1, le=1000)
    max_chunks: int = Field(5000, ge=1, le=20000)
    embedding_batch: int = Field(32, ge=1, le=32)
    workflows: int = Field(2, ge=1, le=8)
    queue_size: int = Field(8, ge=1, le=64)
    tools_concurrency: int = Field(4, ge=1, le=16)
    max_runs: int = Field(2000, ge=1, le=100000)
    context_chars: int = Field(3000, ge=1, le=10000)
    context_turns: int = Field(4, ge=1, le=20)
    cache_entries: int = Field(64, ge=0, le=1000)
    model_timeout: float = Field(15.0, ge=0.1, le=120)
    max_output_tokens: int = Field(512, ge=1, le=4096)
    max_model_calls: int = Field(8, ge=1, le=64)
    max_build_model_calls: int = Field(64, ge=1, le=1000)
    enable_simulation: bool = False
    request_bytes: int = Field(65536, ge=1024, le=1048576)
    rate_per_minute: int = Field(120, ge=1, le=100000)
    agent_http: AgentHttpLimits = Field(default_factory=AgentHttpLimits)
    graph_version: str = "agent-v2"
    state_version: int = 1

    @model_validator(mode="after")
    def safety(self):
        if any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", v) or v in ("public", "system", "local")
            for v in self.bearer_principals.values()
        ):
            raise ValueError("bearer owners must be valid non-reserved principal IDs")
        loopback = self.host == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            pass
        if self.auth_mode == "local" and not loopback:
            raise ValueError("non-loopback binding requires server-configured bearer identity")
        if self.auth_mode == "bearer" and (
            not self.bearer_principals or any(len(k) < 24 for k in self.bearer_principals)
        ):
            raise ValueError("bearer mode requires tokens of at least 24 characters")
        if self.mode == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI mode explicitly requires OPENAI_API_KEY")
        return self

    def public(self) -> dict:
        return {
            "mode": self.mode,
            "auth_mode": self.auth_mode,
            "host": self.host,
            "embedding": "lexical-hash-v1" if self.mode == "local" else self.embedding_model,
            "generator": "extractive-v1" if self.mode == "local" else self.chat_model,
            "semantic": self.mode != "local",
            "external_tracing": False,
            "graph_version": self.graph_version,
            "state_version": self.state_version,
        }
