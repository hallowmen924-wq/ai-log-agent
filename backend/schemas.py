from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NewsCollectResponse(BaseModel):
    news: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    count: int = 0
    last_new_item_time: str | None = None


class LogAnalyzeRequest(BaseModel):
    raw_logs: str | None = None
    log_dir: str = "data/logs"


class LogAnalyzeResponse(BaseModel):
    file_count: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)


class FaissBuildRequest(BaseModel):
    logs: list[dict[str, Any]] | None = None
    news: list[dict[str, Any]] | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class StrategyChatRequest(BaseModel):
    question: str


class StrategyChatResponse(BaseModel):
    answer: str
    question: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    context: dict[str, list[str]] = Field(default_factory=dict)
    prompt_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    vector_update: dict[str, int] = Field(default_factory=dict)


class WorkerConfigRequest(BaseModel):
    interval_seconds: int = 10


class FullAnalysisResponse(BaseModel):
    running: bool
    file_count: int
    vector_count: int
    total_time: float
    issues: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    news: list[dict[str, Any]] = Field(default_factory=list)
    last_news_time: str | None = None
    last_new_item_time: str | None = None
    last_run_time: str | None = None
    latest_strategy_question: str | None = None
    last_strategy_time: str | None = None
    last_log_ingest_time: str | None = None
    latest_log_briefing: str | None = None
    last_log_briefing_time: str | None = None
    latest_log_prompt_input: dict[str, Any] = Field(default_factory=dict)
    last_log_prompt_input_time: str | None = None
    latest_news_briefing: str | None = None
    last_news_briefing_time: str | None = None
    latest_news_prompt_input: dict[str, Any] = Field(default_factory=dict)
    last_news_prompt_input_time: str | None = None
    agent_statuses: dict[str, dict[str, Any]] = Field(default_factory=dict)
    agent_activity_log: list[dict[str, Any]] = Field(default_factory=list)
    vector_events: list[dict[str, Any]] = Field(default_factory=list)


class GenericMessage(BaseModel):
    status: str
    detail: str | None = None
