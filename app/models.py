"""Pydantic schemas that match the SHL evaluator contract exactly."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
                    {"role": "assistant", "content": "Sure. What is the seniority level?"},
                    {"role": "user", "content": "Mid-level, around 4 years"},
                ]
            }
        }
    }


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
