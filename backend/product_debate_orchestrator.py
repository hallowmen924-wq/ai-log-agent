from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import random
import re
import time
from typing import Any, Callable


def _now_iso() -> str:
    return dt.datetime.now().isoformat()


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[0-9A-Za-z가-힣_]+", str(text or "")) if len(t) >= 2]


class JsonLongTermMemory:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
        except Exception:
            return []
        return []

    def _write(self, rows: list[dict[str, Any]]) -> None:
        try:
            self.path.write_text(
                json.dumps(rows[-200:], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def recall(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        rows = self._read()
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            text = " ".join(
                [
                    str(row.get("agenda_title") or ""),
                    str(row.get("issue_summary") or ""),
                    " ".join([str(x) for x in list(row.get("keywords") or [])]),
                ]
            )
            tokens = set(_tokenize(text))
            score = len(q_tokens & tokens)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    def append(self, entry: dict[str, Any]) -> None:
        rows = self._read()
        rows.append(entry)
        self._write(rows)


def _safe_json_obj(parse_json: Callable[[str], dict[str, object]], text: str) -> dict[str, Any]:
    try:
        payload = parse_json(text)
        if isinstance(payload, dict):
            return dict(payload)
    except Exception:
        return {}
    return {}


def _json_char_len(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return len(str(value or ""))


def _prompt_persona_turn(
    persona: dict[str, Any],
    selected_agenda: dict[str, Any],
    memory_hits: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    round_index: int,
) -> str:
    name = str(persona.get("name") or "Agent")
    role = str(persona.get("role") or "-")
    skills = ", ".join([str(s) for s in list(persona.get("skills") or [])]) or "-"
    memory = ", ".join([str(m) for m in list(persona.get("memory") or [])]) or "-"
    constraints = ", ".join([str(c) for c in list(persona.get("constraints") or [])]) or "-"
    memory_block = json.dumps(memory_hits, ensure_ascii=False, indent=2)
    recent = json.dumps(transcript[-8:], ensure_ascii=False, indent=2)
    return f"""
너는 {name}이다.

역할:
{role}

보유 skill:
{skills}

기억:
{memory}

판단 제약:
{constraints}

현재 안건:
{json.dumps(selected_agenda, ensure_ascii=False)}

장기기억 회상:
{memory_block}

현재 토론 맥락(최근 발언):
{recent}

현재 라운드: {round_index + 1}

다른 부서와 충돌 가능한 지점을 명시하고, 실행 가능한 수정안을 1개 포함해라.
반드시 JSON으로만 답해라:
{{
  "message": "한 단락 발언",
  "position": "찬성/유보/반대 중 하나",
  "risk_score": 1~5 숫자,
  "growth_score": 1~5 숫자,
  "feasibility_score": 1~5 숫자
}}
""".strip()


def _prompt_moderator_round(
    selected_agenda: dict[str, Any],
    transcript: list[dict[str, Any]],
    round_index: int,
    max_rounds: int,
) -> str:
    return f"""
너는 중재자(Moderator)다.
안건: {json.dumps(selected_agenda, ensure_ascii=False)}
라운드: {round_index + 1}/{max_rounds}
최근 토론:
{json.dumps(transcript[-12:], ensure_ascii=False, indent=2)}

합의 상태를 평가하고 계속 여부를 판단하라.
반드시 JSON으로만 답해라:
{{
  "consensus_score": 0.0~1.0,
  "stop": true/false,
  "summary": "현재 합의 요약",
  "dissent": ["주요 이견1", "주요 이견2"],
  "next_focus": ["다음 라운드 초점1", "다음 라운드 초점2"]
}}
""".strip()


def _prompt_final_synthesis(
    selected_agenda: dict[str, Any],
    transcript: list[dict[str, Any]],
    consensus_history: list[dict[str, Any]],
    fallback_final: dict[str, Any],
) -> str:
    return f"""
너는 토론 결과 정리자다.
선택 안건:
{json.dumps(selected_agenda, ensure_ascii=False)}

토론 로그:
{json.dumps(transcript, ensure_ascii=False, indent=2)}

합의 히스토리:
{json.dumps(consensus_history, ensure_ascii=False, indent=2)}

반드시 JSON으로만 답해라. 스키마:
{{
  "final": {{
    "new_product": {{
      "name": "상품명",
      "target": "대상",
      "core_logic": ["로직1", "로직2"],
      "limit_rate_policy": "한도/금리 정책",
      "risk_guardrails": ["가드레일1", "가드레일2"]
    }},
    "product_logic_improvements": [
      {{"product": "상품", "change": "변경", "expected_effect": "효과", "dev_impact": "개발영향"}}
    ],
    "implementation_plan": ["단계1", "단계2"],
    "kpis": ["KPI1", "KPI2"]
  }}
}}

아래는 fallback 참고본:
{json.dumps(fallback_final, ensure_ascii=False)}
""".strip()


def _call_with_retry(
    llm_call: Callable[[str], str],
    prompt: str,
    retries: int,
) -> str:
    last_error: Exception | None = None
    for _ in range(max(1, retries + 1)):
        try:
            return str(llm_call(prompt) or "")
        except Exception as error:  # noqa: BLE001
            last_error = error
    if last_error:
        raise last_error
    return ""


def _limit_message_text(text: str, max_chars: int = 150, max_sentences: int = 3) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    clipped = normalized[:max_chars]
    parts = [p.strip() for p in re.split(r"(?<=[.!?。])\s+", clipped) if p.strip()]
    if not parts:
        return clipped
    return " ".join(parts[:max_sentences])[:max_chars]


def _normalize_autogen_error_message(error: Exception | str) -> str:
    raw = str(error or "").strip()
    lowered = raw.lower()
    if "no module named" in lowered and "autogen_agentchat" in lowered:
        return "AutoGen 패키지가 설치되지 않아 기본 토론 엔진으로 자동 전환했습니다."
    return raw


def _run_with_autogen_framework(
    *,
    selected_agenda: dict[str, Any],
    context: dict[str, Any],
    concepts: list[dict[str, Any]],
    personas: list[dict[str, Any]],
    parse_json: Callable[[str], dict[str, object]],
    fallback_result: dict[str, Any],
    memory_hits: list[dict[str, Any]],
    max_rounds: int,
    progress_callback: Callable[[str, str], None] | None = None,
    memory_elapsed_ms: int = 0,
) -> dict[str, Any]:
    """Optional AutoGen path. Fallback to custom orchestrator on any exception."""
    import asyncio

    from autogen_agentchat.agents import AssistantAgent  # type: ignore
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore

    function_started = time.perf_counter()
    if progress_callback:
        progress_callback("autogen-init", "AutoGen 모델 클라이언트를 초기화합니다.")
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("AUTOGEN_MODEL", "mistral"),
        base_url=os.getenv("AUTOGEN_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key=os.getenv("AUTOGEN_API_KEY", "ollama"),
        temperature=float(os.getenv("PRODUCT_DEBATE_TEMPERATURE", "0.3") or 0.3),
        model_info={
            "family": "unknown",
            "vision": False,
            "json_output": False,
            "function_calling": False,
        },
    )

    agents: list[Any] = []
    system_prompt_chars = 0
    if progress_callback:
        progress_callback("autogen-personas", "부서별 에이전트 페르소나를 구성합니다.")
    for persona in personas:
        name = str(persona.get("name") or persona.get("department") or "Agent")
        role = str(persona.get("role") or "")
        skills = ", ".join([str(x) for x in list(persona.get("skills") or [])]) or "-"
        memory = ", ".join([str(x) for x in list(persona.get("memory") or [])]) or "-"
        constraints = ", ".join([str(x) for x in list(persona.get("constraints") or [])]) or "-"
        sys_msg = (
            f"너는 {name}이다.\n"
            f"역할: {role}\n"
            f"보유 skill: {skills}\n"
            f"기억: {memory}\n"
            f"판단 제약: {constraints}\n"
            "부서 관점 의견을 간결하게 제시하라."
        )
        sys_msg = f"{sys_msg}\n반드시 JSON만 출력하고, message는 최대 3문장/150자 이내로 유지하라."
        system_prompt_chars += len(sys_msg)
        agents.append(AssistantAgent(name=name, model_client=model_client, system_message=sys_msg))

    agents.append(
        AssistantAgent(
            name="Moderator",
            model_client=model_client,
            system_message=(
                "너는 토론 중재자다. 쟁점 정리, 합의 유도, 종료 판단을 담당한다. "
                "마지막 답변 끝에 FINAL_JSON: {...} 형태로 단일 JSON을 반드시 포함하라."
            ),
        )
    )

    max_turns = len(agents)

    kickoff = (
        f"[안건] {json.dumps(selected_agenda, ensure_ascii=False)}\n"
        f"[관련 컨셉] {json.dumps(concepts[:8], ensure_ascii=False)}\n"
        f"[장기기억 히트] {json.dumps(memory_hits, ensure_ascii=False)}\n"
        "각 부서는 리스크/기대효과를 제시하고, Moderator는 FINAL_JSON으로 합의안을 제시하라."
    )

    kickoff = f"{kickoff}\n모든 발언은 짧게 작성하고 JSON 블록만 유지하라."
    if not memory_hits:
        kickoff = "\n".join(line for line in kickoff.splitlines() if "[]" not in line)
    autogen_timeout_sec = float(os.getenv("PRODUCT_DEBATE_AUTOGEN_TIMEOUT_SEC", "90") or 90)
    kickoff_chars = len(kickoff)
    context_chars = _json_char_len(concepts[:8])
    agenda_chars = _json_char_len(selected_agenda)
    memory_chars = _json_char_len(memory_hits)
    setup_elapsed_ms = int((time.perf_counter() - function_started) * 1000)

    if progress_callback:
        progress_callback(
            "autogen-run",
            f"AutoGen 순차 실행 중 · agents {len(agents)} · calls {max_turns} · kickoff {kickoff_chars} chars",
        )
    autogen_started = time.perf_counter()
    agent_timeout_sec = float(os.getenv("PRODUCT_DEBATE_AGENT_TIMEOUT_SEC", "30") or 30)

    async def _run_agents_sequentially() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        previous_messages: list[dict[str, str]] = []
        for index, agent in enumerate(agents):
            agent_name = str(getattr(agent, "name", None) or f"Agent{index + 1}")
            task = "\n".join(
                [
                    kickoff,
                    "",
                    "[이전 부서 발언]",
                    json.dumps(previous_messages[-6:], ensure_ascii=False),
                    "",
                    "이번 차례의 부서 관점만 짧게 답하세요. Moderator는 마지막에 FINAL_JSON을 포함하세요.",
                ]
            )
            if progress_callback:
                progress_callback(
                    "autogen-agent-run",
                    f"{agent_name} 호출 중 · {index + 1}/{len(agents)} · timeout {agent_timeout_sec:.0f}s",
                )
            call_started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    agent.run(task=task, output_task_messages=False),
                    timeout=max(4.0, agent_timeout_sec),
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError(
                    f"AutoGen agent timeout: {agent_name} after {agent_timeout_sec:.1f}s "
                    f"(call={index + 1}/{len(agents)}, kickoff_chars={kickoff_chars}, "
                    f"system_prompt_chars={system_prompt_chars}, memory_chars={memory_chars})"
                ) from error
            elapsed_ms = int((time.perf_counter() - call_started) * 1000)
            contents: list[str] = []
            for msg in list(getattr(result, "messages", None) or []):
                text = str(getattr(msg, "content", "") or "").strip()
                if text:
                    contents.append(text)
            content = "\n".join(contents).strip()
            rows.append({"agent": agent_name, "response": content, "elapsed_ms": elapsed_ms})
            previous_messages.append({"speaker": agent_name, "message": content[:700]})
        return rows

    try:
        sequential_rows = asyncio.run(asyncio.wait_for(_run_agents_sequentially(), timeout=max(4.0, autogen_timeout_sec)))
    except asyncio.TimeoutError as error:
        raise TimeoutError(
            "AutoGen sequential timeout "
            f"after {autogen_timeout_sec:.1f}s "
            f"(agents={len(agents)}, calls={max_turns}, kickoff_chars={kickoff_chars}, "
            f"system_prompt_chars={system_prompt_chars}, memory_chars={memory_chars})"
        ) from error
    autogen_elapsed_ms = int((time.perf_counter() - autogen_started) * 1000)
    transcript: list[dict[str, Any]] = []
    final_payload: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    parse_started = time.perf_counter()
    output_chars = 0
    if progress_callback:
        progress_callback("autogen-parse", "AutoGen 발언 로그와 최종 결과를 파싱합니다.")
    agent_timings: list[dict[str, Any]] = []
    for row in sequential_rows:
        speaker = str(row.get("agent") or "Agent")
        content = str(row.get("response") or "").strip()
        if not content:
            continue
        output_chars += len(content)
        elapsed_ms = int(row.get("elapsed_ms") or 0)
        traces.append({"agent": speaker, "response": content, "elapsed_ms": elapsed_ms})
        agent_timings.append({"agent": speaker, "elapsed_ms": elapsed_ms})
        transcript.append({"speaker": speaker, "tone": "neutral", "message": content})
        if "FINAL_JSON:" in content:
            tail = content.split("FINAL_JSON:", 1)[1].strip()
            final_payload = _safe_json_obj(parse_json, tail)

    parsed_final = final_payload if isinstance(final_payload, dict) else {}
    if not parsed_final:
        parsed_final = dict(fallback_result.get("final") or {})
    parse_elapsed_ms = int((time.perf_counter() - parse_started) * 1000)
    diagnostics = {
        "path": "autogen",
        "model": os.getenv("AUTOGEN_MODEL", "mistral"),
        "base_url": os.getenv("AUTOGEN_BASE_URL", "http://127.0.0.1:11434/v1"),
        "agents": len(agents),
        "persona_agents": len(personas),
        "max_turns": max_turns,
        "max_rounds": max_rounds,
        "timeout_sec": autogen_timeout_sec,
        "setup_elapsed_ms": setup_elapsed_ms,
        "memory_elapsed_ms": memory_elapsed_ms,
        "autogen_run_elapsed_ms": autogen_elapsed_ms,
        "parse_elapsed_ms": parse_elapsed_ms,
        "kickoff_chars": kickoff_chars,
        "system_prompt_chars": system_prompt_chars,
        "agenda_chars": agenda_chars,
        "context_chars": context_chars,
        "memory_chars": memory_chars,
        "memory_enabled": bool(memory_hits),
        "output_chars": output_chars,
        "message_count": len(sequential_rows),
        "agent_timings": agent_timings,
    }

    result = {
        **fallback_result,
        "selected_agenda": selected_agenda,
        "messages": transcript or list(fallback_result.get("messages") or []),
        "final": parsed_final,
        "product_cards": list(fallback_result.get("product_cards") or []),
        "concepts": concepts,
        "orchestration": {
            "engine": "autogen",
            "max_rounds": max_rounds,
            "rounds_run": len(sequential_rows),
            "round_timings": [{"round": 1, "elapsed_ms": autogen_elapsed_ms, "calls": agent_timings}],
            "final_elapsed_ms": 0,
            "total_elapsed_ms": autogen_elapsed_ms,
            "diagnostics": diagnostics,
            "total_budget_sec": float(os.getenv("PRODUCT_DEBATE_TOTAL_BUDGET_SEC", "35") or 35),
            "budget_exceeded": False,
            "memory_hits": memory_hits,
        },
    }
    return {
        "status": "ok",
        "source": "autogen-ollama",
        "context": context,
        "result": result,
        "llm": {
            "model": os.getenv("AUTOGEN_MODEL", "mistral"),
            "mode": "autogen",
            "traces": traces[-10:],
            "diagnostics": diagnostics,
        },
    }


def run_product_debate_orchestration(
    *,
    selected_agenda: dict[str, Any],
    context: dict[str, Any],
    concepts: list[dict[str, Any]],
    personas: list[dict[str, Any]],
    llm_call: Callable[[str], str],
    parse_json: Callable[[str], dict[str, object]],
    fallback_result: dict[str, Any],
    memory_path: pathlib.Path,
    max_rounds: int = 3,
    retries: int = 1,
    consensus_threshold: float = 0.72,
    require_autogen: bool = False,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    max_agents = max(1, int(os.getenv("PRODUCT_DEBATE_MAX_AGENTS", "3") or 3))
    personas = list(personas or [])[:max_agents]
    started_at = time.perf_counter()
    total_budget_sec = float(os.getenv("PRODUCT_DEBATE_TOTAL_BUDGET_SEC", "35") or 35)

    def _budget_exceeded(reserve_sec: float = 0.0) -> bool:
        return (time.perf_counter() - started_at) >= max(5.0, total_budget_sec - reserve_sec)

    memory_enabled = str(os.getenv("PRODUCT_DEBATE_MEMORY_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    memory = JsonLongTermMemory(memory_path) if memory_enabled else None
    query = f"{selected_agenda.get('title') or ''} {selected_agenda.get('summary') or ''}".strip()
    memory_started = time.perf_counter()
    memory_hits = memory.recall(query, top_k=3) if memory_enabled and memory is not None else []
    memory_elapsed_ms = int((time.perf_counter() - memory_started) * 1000)
    if progress_callback:
        if memory_enabled:
            progress_callback("memory-recall", f"롱텀메모리 회수 {len(memory_hits)}건 · {memory_elapsed_ms}ms")
        else:
            progress_callback("memory-disabled", "롱텀메모리 조회/주입 비활성화")
    use_autogen = str(os.getenv("PRODUCT_DEBATE_USE_AUTOGEN", "1")).strip().lower() in {"1", "true", "yes", "on"}
    autogen_error = ""

    def _append_memory_from_result(result_obj: dict[str, Any], consensus_score: float = 0.0) -> None:
        if not memory_enabled or memory is None:
            return
        try:
            final_obj = dict(result_obj.get("final") or {})
            memory.append(
                {
                    "timestamp": _now_iso(),
                    "agenda_title": str(selected_agenda.get("title") or ""),
                    "issue_summary": str((final_obj.get("new_product") or {}).get("target") or ""),
                    "keywords": _tokenize(
                        " ".join(
                            [
                                str(selected_agenda.get("title") or ""),
                                str(selected_agenda.get("summary") or ""),
                                " ".join([str(x) for x in list(final_obj.get("kpis") or [])]),
                            ]
                        )
                    )[:20],
                    "consensus_score": float(consensus_score or 0.0),
                }
            )
        except Exception:
            return

    if use_autogen:
        try:
            autogen_payload = _run_with_autogen_framework(
                selected_agenda=selected_agenda,
                context=context,
                concepts=concepts,
                personas=personas,
                parse_json=parse_json,
                fallback_result=fallback_result,
                memory_hits=memory_hits,
                max_rounds=max_rounds,
                progress_callback=progress_callback,
                memory_elapsed_ms=memory_elapsed_ms,
            )
            result_obj = dict(autogen_payload.get("result") or {})
            if isinstance(result_obj.get("final"), dict) and isinstance(result_obj.get("messages"), list):
                _append_memory_from_result(result_obj, consensus_score=0.0)
                orchestration_obj = dict(result_obj.get("orchestration") or {})
                orchestration_obj["max_agents"] = max_agents
                result_obj["orchestration"] = orchestration_obj
                autogen_payload["result"] = result_obj
            if progress_callback:
                progress_callback("completed", "AutoGen 토론 완료")
            return autogen_payload
        except Exception as error:  # noqa: BLE001
            # AutoGen import/runtime 오류 시 기존 오케스트레이터로 자동 폴백
            autogen_error = _normalize_autogen_error_message(error)
            if progress_callback:
                progress_callback("autogen-error", f"AutoGen 실패: {autogen_error}")
            optional_fallback_error = (
                "자동 전환" in autogen_error
                or "설치되지 않아" in autogen_error
                or "no module named 'autogen_agentchat'" in autogen_error.lower()
            )
            if require_autogen and not optional_fallback_error:
                raise RuntimeError(f"AutoGen 경로 실패: {autogen_error}") from error

    if progress_callback:
        progress_callback("custom-fallback", "커스텀 루프로 전환합니다.")
    transcript: list[dict[str, Any]] = []
    consensus_history: list[dict[str, Any]] = []
    round_timings: list[dict[str, Any]] = []
    source = "fallback"
    llm_traces: list[dict[str, Any]] = []
    persona_retries = 0
    synthesis_retries = max(0, retries)
    early_stop_round = max(2, int(os.getenv("PRODUCT_DEBATE_EARLY_STOP_ROUND", "2") or 2))
    early_stop_threshold = float(os.getenv("PRODUCT_DEBATE_EARLY_STOP_THRESHOLD", "0.62") or 0.62)

    try:
        for round_index in range(max(1, max_rounds)):
            if _budget_exceeded(reserve_sec=3.0):
                break
            round_start = time.perf_counter()
            call_timings: list[dict[str, Any]] = []
            random.shuffle(personas)
            for persona in personas:
                if _budget_exceeded(reserve_sec=2.0):
                    break
                persona_prompt = _prompt_persona_turn(
                    persona=persona,
                    selected_agenda=selected_agenda,
                    memory_hits=memory_hits,
                    transcript=transcript,
                    round_index=round_index,
                )
                persona_prompt = f"{persona_prompt}\n추가 규칙: message는 최대 3문장/150자."
                persona_call_started = time.perf_counter()
                persona_text = _call_with_retry(llm_call, persona_prompt, retries=persona_retries)
                call_timings.append(
                    {
                        "stage": "persona",
                        "agent": str(persona.get("name") or persona.get("department") or "Agent"),
                        "elapsed_ms": int((time.perf_counter() - persona_call_started) * 1000),
                    }
                )
                llm_traces.append({"agent": persona.get("name"), "round": round_index + 1, "prompt": persona_prompt, "response": persona_text})
                obj = _safe_json_obj(parse_json, persona_text)
                message = _limit_message_text(str(obj.get("message") or "").strip(), max_chars=150, max_sentences=3) or "관점 의견을 보완 중입니다."
                transcript.append(
                    {
                        "speaker": str(persona.get("name") or persona.get("department") or "부서"),
                        "tone": str(persona.get("tone") or "neutral"),
                        "message": message,
                        "position": str(obj.get("position") or ""),
                        "scores": {
                            "risk": obj.get("risk_score"),
                            "growth": obj.get("growth_score"),
                            "feasibility": obj.get("feasibility_score"),
                        },
                    }
                )

            moderator_prompt = _prompt_moderator_round(
                selected_agenda=selected_agenda,
                transcript=transcript,
                round_index=round_index,
                max_rounds=max_rounds,
            )
            moderator_prompt = f"{moderator_prompt}\n추가 규칙: summary는 최대 2문장."
            if _budget_exceeded(reserve_sec=1.0):
                round_timings.append(
                    {
                        "round": round_index + 1,
                        "elapsed_ms": int((time.perf_counter() - round_start) * 1000),
                        "total_call_ms": sum(int(item.get("elapsed_ms") or 0) for item in call_timings),
                        "calls": call_timings,
                    }
                )
                break
            moderator_call_started = time.perf_counter()
            moderator_text = _call_with_retry(llm_call, moderator_prompt, retries=synthesis_retries)
            call_timings.append(
                {
                    "stage": "moderator",
                    "agent": "Moderator",
                    "elapsed_ms": int((time.perf_counter() - moderator_call_started) * 1000),
                }
            )
            llm_traces.append({"agent": "Moderator", "round": round_index + 1, "prompt": moderator_prompt, "response": moderator_text})
            moderator_obj = _safe_json_obj(parse_json, moderator_text)
            consensus_score = float(moderator_obj.get("consensus_score") or 0.0)
            stop = bool(moderator_obj.get("stop") or False)
            if (round_index + 1) >= early_stop_round and consensus_score >= early_stop_threshold:
                stop = True
            consensus_history.append(
                {
                    "round": round_index + 1,
                    "consensus_score": round(consensus_score, 3),
                    "summary": _limit_message_text(str(moderator_obj.get("summary") or "").strip(), max_chars=120, max_sentences=2),
                    "dissent": list(moderator_obj.get("dissent") or []),
                    "next_focus": list(moderator_obj.get("next_focus") or []),
                }
            )
            if stop or consensus_score >= consensus_threshold:
                round_timings.append(
                    {
                        "round": round_index + 1,
                        "elapsed_ms": int((time.perf_counter() - round_start) * 1000),
                        "total_call_ms": sum(int(item.get("elapsed_ms") or 0) for item in call_timings),
                        "calls": call_timings,
                    }
                )
                break
            round_timings.append(
                {
                    "round": round_index + 1,
                    "elapsed_ms": int((time.perf_counter() - round_start) * 1000),
                    "total_call_ms": sum(int(item.get("elapsed_ms") or 0) for item in call_timings),
                    "calls": call_timings,
                }
            )

        final_prompt = _prompt_final_synthesis(
            selected_agenda=selected_agenda,
            transcript=transcript,
            consensus_history=consensus_history,
            fallback_final=dict(fallback_result.get("final") or {}),
        )
        final_prompt = f"{final_prompt}\n추가 규칙: JSON 이외 설명 최소화."
        final_elapsed_ms = 0
        parsed_final = {}
        if not _budget_exceeded(reserve_sec=0.0):
            final_call_started = time.perf_counter()
            final_text = _call_with_retry(llm_call, final_prompt, retries=synthesis_retries)
            final_elapsed_ms = int((time.perf_counter() - final_call_started) * 1000)
            llm_traces.append({"agent": "FinalSynthesizer", "round": len(consensus_history), "prompt": final_prompt, "response": final_text})
            final_obj = _safe_json_obj(parse_json, final_text)
            parsed_final = final_obj.get("final") if isinstance(final_obj.get("final"), dict) else {}
        if parsed_final:
            source = "orchestrator-ollama"
        else:
            parsed_final = dict(fallback_result.get("final") or {})

        result = {
            **fallback_result,
            "selected_agenda": selected_agenda,
            "messages": transcript or list(fallback_result.get("messages") or []),
            "final": parsed_final,
            "product_cards": list(fallback_result.get("product_cards") or []),
            "concepts": concepts,
            "orchestration": {
                "max_rounds": max_rounds,
                "rounds_run": len(consensus_history),
                "retries": retries,
                "persona_retries": persona_retries,
                "synthesis_retries": synthesis_retries,
                "early_stop_round": early_stop_round,
                "early_stop_threshold": early_stop_threshold,
                "consensus_threshold": consensus_threshold,
                "consensus_history": consensus_history,
                "round_timings": round_timings,
                "final_elapsed_ms": final_elapsed_ms,
                "total_elapsed_ms": sum(int(item.get("elapsed_ms") or 0) for item in round_timings) + final_elapsed_ms,
                "total_budget_sec": total_budget_sec,
                "budget_exceeded": _budget_exceeded(reserve_sec=0.0),
                "memory_hits": memory_hits,
            },
        }

        if memory_enabled and memory is not None:
            memory.append(
                {
                    "timestamp": _now_iso(),
                    "agenda_title": str(selected_agenda.get("title") or ""),
                    "issue_summary": str((result.get("final") or {}).get("new_product", {}).get("target") or ""),
                    "keywords": _tokenize(
                        " ".join(
                            [
                                str(selected_agenda.get("title") or ""),
                                str(selected_agenda.get("summary") or ""),
                                " ".join(
                                    [
                                        str(x)
                                        for x in list(((result.get("final") or {}).get("kpis") or []))
                                    ]
                                ),
                            ]
                        )
                    )[:20],
                    "consensus_score": (consensus_history[-1].get("consensus_score") if consensus_history else 0.0),
                }
            )

        return {
            "status": "ok",
            "source": source,
            "context": context,
            "result": result,
            "llm": {
                "model": "ollama",
                "mode": "custom_orchestrator",
                "autogen_error": autogen_error,
                "traces": llm_traces[-10:],
            },
        }
    except Exception as error:  # noqa: BLE001
        return {
            "status": "ok",
            "source": "fallback",
            "context": context,
            "result": fallback_result,
            "llm": {
                "model": "ollama",
                "mode": "custom_orchestrator",
                "autogen_error": autogen_error,
                "error": str(error),
            },
        }
