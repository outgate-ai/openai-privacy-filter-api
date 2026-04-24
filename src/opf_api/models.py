"""Pydantic models matching Ollama's /api/chat request/response contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class ChatResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class TagModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "safetensors"
    family: str = "privacy-filter"
    families: list[str] = Field(default_factory=lambda: ["privacy-filter"])
    parameter_size: str = "1.5B"
    quantization_level: str = "BF16"


class TagModel(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int = 0
    digest: str = ""
    details: TagModelDetails = Field(default_factory=TagModelDetails)


class TagsResponse(BaseModel):
    models: list[TagModel]


class VersionResponse(BaseModel):
    version: str
