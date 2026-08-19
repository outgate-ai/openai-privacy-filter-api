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


class OpenAIChatMessage(BaseModel):
    role: str
    # str or a list of content parts; flattened in the handler.
    content: Any


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class OpenAIChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIChoiceMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
