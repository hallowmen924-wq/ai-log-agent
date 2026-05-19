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
    store_name: str | None = None


class StrategyChatRequest(BaseModel):
    question: str
    news_prompt_template: str | None = None
    log_prompt_template: str | None = None


class CardloanDebateRequest(BaseModel):
    question: str
    reviewer_prompts: dict[str, str] = Field(default_factory=dict)
    reviewer_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)


class NewsPromptTemplateRequest(BaseModel):
    template: str | None = None


class LogPromptTemplateRequest(BaseModel):
    template: str | None = None


class AgentOllamaToggleRequest(BaseModel):
    enabled: bool = True


class OllamaRuntimeToggleRequest(BaseModel):
    enabled: bool = True


class StrategyChatResponse(BaseModel):
    answer: str
    question: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    context: dict[str, list[str]] = Field(default_factory=dict)
    prompt_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    vector_update: dict[str, int] = Field(default_factory=dict)


class CardloanDebateResponse(BaseModel):
    status: str = "completed"
    question: str | None = None
    summary: str = ""
    round_results: list[dict[str, Any]] = Field(default_factory=list)
    current_stage: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class WorkerConfigRequest(BaseModel):
    interval_seconds: int = 30


class RegulationUploadResponse(BaseModel):
    status: str = "ok"
    detail: str | None = None
    vector_count: int = 0
    added_count: int = 0
    summary: str = ""
    updated_at: str | None = None
    files: list[str] = Field(default_factory=list)
    file_stats: list[dict[str, Any]] = Field(default_factory=list)


class ProductSummaryResponse(BaseModel):
    status: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)


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
    log_prompt_template_override: str | None = None
    latest_news_briefing: str | None = None
    last_news_briefing_time: str | None = None
    latest_news_prompt_input: dict[str, Any] = Field(default_factory=dict)
    last_news_prompt_input_time: str | None = None
    news_prompt_template_override: str | None = None
    news_agent_ollama_enabled: bool = True
    log_agent_ollama_enabled: bool = True
    regulation_upload_summary_enabled: bool = False
    ollama_gpu_enabled: bool = False
    ontology_query_priority_enabled: bool = False
    latest_regulation_analysis: str | None = None
    last_regulation_analysis_time: str | None = None
    agent_statuses: dict[str, dict[str, Any]] = Field(default_factory=dict)
    agent_activity_log: list[dict[str, Any]] = Field(default_factory=list)
    vector_events: list[dict[str, Any]] = Field(default_factory=list)
    last_faiss_time: str | None = None
    ollama_runtime: dict[str, Any] = Field(default_factory=dict)
    cardloan_debate: dict[str, Any] = Field(default_factory=dict)
    backend_diagnostics: dict[str, Any] = Field(default_factory=dict)
    news_pipeline_stats: dict[str, Any] = Field(default_factory=dict)


class GenericMessage(BaseModel):
    status: str
    detail: str | None = None


class OntologySaveRequest(BaseModel):
    ontology: dict[str, Any] = Field(default_factory=dict)
    commonfeature: dict[str, Any] = Field(default_factory=dict)


class FeatureOntologyConversationTurn(BaseModel):
    question: str = ""
    answer_headline: str = ""
    answer_body: str = ""
    selected_feature_ids: list[str] = Field(default_factory=list)


class FeatureOntologyConversationFeedback(BaseModel):
    preferred_feature_ids: list[str] = Field(default_factory=list)
    avoided_feature_ids: list[str] = Field(default_factory=list)


class FeatureOntologyRuntimeRequest(BaseModel):
    product: str = ""
    query: str = ""
    feature_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    answer_mode: str = "general"
    department: str = ""
    memo_notes: str = ""
    memory_keywords: list[str] = Field(default_factory=list)
    history: list[FeatureOntologyConversationTurn] = Field(default_factory=list)
    feedback: FeatureOntologyConversationFeedback = Field(default_factory=FeatureOntologyConversationFeedback)
    allow_clarification: bool = True
    clarification_budget: int = Field(default=1, ge=0, le=3)
