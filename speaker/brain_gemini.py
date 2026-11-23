"""Gemini 두뇌: google-genai SDK 로 ClaudeBrain 과 같은 일을 한다 (도구 호출, 구글 검색, 문장 단위 스트리밍).

대화 기록 self.messages 에는 사용자 말만 {"role": "user", "content": str} 사전으로 두고,
모델의 답과 도구 결과는 types.Content 그대로 둔다. 모델의 답에는 생각의 서명(thought signature)이
붙어 오는데, 도구 호출이 이어지는 동안 이 서명을 받은 그대로 돌려보내지 않으면 400 오류가 나므로
스트리밍으로 받은 조각(part)을 서명째 모아 하나의 Content 로 남긴다.

구글 검색(서버 도구)과 우리 함수 도구를 한 요청에 함께 주려면 Gemini 3 이후 모델이어야 하고
tool_config.include_server_side_tool_invocations 를 켜야 한다 (미리보기 기능).
Gemini 2.5 이하 모델은 둘을 함께 쓸 수 없어서 검색 없이 함수 도구만 준다.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

from google import genai
from google.genai import errors, types

from .brain import MAX_TOOL_ROUNDS, BaseBrain, _Cancelled
from .config import Config
from .tools import TOOL_DEFINITIONS, ToolBox

log = logging.getLogger("speaker.brain")

# SPEAKER_EFFORT → 생각 수준. Gemini 3 이후는 thinking_level, 2.5 는 thinking_budget(토큰 수)을 쓴다.
_LEVELS = {
    "none": types.ThinkingLevel.MINIMAL,
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.XHIGH,
    "max": types.ThinkingLevel.XHIGH,
}
_BUDGETS = {"none": 0, "minimal": 0, "low": 512, "medium": 2048, "high": 8192, "xhigh": 16384, "max": 24576}

# 안전 필터 등으로 답이 막힌 경우의 종료 이유
_BLOCKED = {
    types.FinishReason.SAFETY,
    types.FinishReason.RECITATION,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.MODEL_ARMOR,
}
_REFUSAL = "죄송해요, 그 요청은 도와드리기 어려워요."

# 네트워크 오류 (SDK 는 httpx 를 쓰고, 설치돼 있으면 httpx2 를 쓰기도 한다)
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
for _module in ("httpx", "httpx2"):
    try:
        _NETWORK_ERRORS += (__import__(_module).TransportError,)
    except (ImportError, AttributeError):
        pass


def _function_declarations() -> list[types.FunctionDeclaration]:
    """Anthropic 형식 도구 정의를 Gemini 함수 선언으로 바꾼다 (eager_input_streaming 같은 Anthropic 전용 키는 버린다).

    parameters_json_schema 는 JSON Schema 를 그대로 받는다 (minimum·maximum·enum·items 등).
    실행 전에 ToolBox._validate 가 스키마를 다시 확인한다.
    """
    return [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters_json_schema=t["input_schema"],
        )
        for t in TOOL_DEFINITIONS
    ]


def _is_legacy(model: str) -> bool:
    """Gemini 2.5 이하: thinking_level 을 모르고, 검색과 함수 도구를 함께 쓸 수 없다."""
    name = model.lower().removeprefix("models/")
    return name.startswith(("gemini-1", "gemini-2"))


def _plain_text(part: types.Part) -> bool:
    """서명도 다른 내용도 없는 보통 글 조각인지 (이런 조각끼리만 이어 붙여도 안전하다)."""
    return (
        part.text is not None
        and not part.thought
        and part.thought_signature is None
        and part.model_dump(exclude_none=True).keys() <= {"text"}
    )


def _add_part(parts: list[types.Part], part: types.Part) -> None:
    """스트리밍 조각을 모은다. 서명이 붙은 조각은 그 자리에 그대로 두어야 하므로
    서명 없는 보통 글끼리만 하나로 합친다 (문서: 서명 있는 조각과 없는 조각을 합치지 말 것)."""
    if part == types.Part():
        return  # 완전히 빈 조각은 다시 보내면 오류가 날 수 있다
    if parts and _plain_text(parts[-1]) and _plain_text(part):
        parts[-1] = types.Part(text=parts[-1].text + part.text)
    else:
        parts.append(part)


class GeminiBrain(BaseBrain):
    provider = "Gemini"

    def __init__(self, cfg: Config, toolbox: ToolBox):
        if not cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY 가 없습니다. .env 파일에 넣어 주세요.")
        super().__init__(cfg, toolbox)
        self.client = genai.Client(api_key=cfg.gemini_api_key, http_options=types.HttpOptions(timeout=60_000))
        self.legacy = _is_legacy(cfg.model)
        self._minimal_ok = True  # 요약에 '최소 생각'을 쓸 수 있는지 (gemini-3.8-flash·Gemini 4 는 안 됨)
        tool = types.Tool(function_declarations=_function_declarations())
        if self.legacy:
            log.warning("%s 는 구글 검색과 함수 도구를 함께 쓸 수 없어 검색 없이 동작합니다.", cfg.model)
            self.tools = [tool]
            self.tool_config = None
        else:
            tool.google_search = types.GoogleSearch()
            self.tools = [tool]
            # 서버 도구(검색)와 함수 도구를 함께 쓰려면 이 플래그가 필요하다. 이때는 VALIDATED 모드만 된다 (AUTO 불가).
            self.tool_config = types.ToolConfig(
                include_server_side_tool_invocations=True,
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.VALIDATED),
            )

    # ---------------------------------------------------------------- 요청 옵션

    def _thinking(self, effort: str) -> types.ThinkingConfig:
        effort = (effort or "low").lower()
        if self.legacy:
            budget = _BUDGETS.get(effort, _BUDGETS["low"])
            if budget == 0 and "pro" in self.cfg.model:
                budget = 128  # 2.5 Pro 는 생각을 끌 수 없다
            return types.ThinkingConfig(thinking_budget=budget)
        if effort not in _LEVELS:
            log.warning("알 수 없는 생각 깊이 %r, low 로 합니다.", effort)
        return types.ThinkingConfig(thinking_level=_LEVELS.get(effort, types.ThinkingLevel.LOW))

    def _contents(self) -> list[types.Content]:
        """self.messages 를 요청용 Content 목록으로. 사용자 말 사전만 바꾸고 나머지는 받은 그대로 보낸다."""
        contents = []
        for m in self.messages:
            if isinstance(m, dict):
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            else:
                contents.append(m)
        return contents

    # ---------------------------------------------------------------- BaseBrain

    def _error_kind(self, exc: Exception) -> str | None:
        if isinstance(exc, errors.ClientError):
            if exc.code in (401, 403):
                return "auth"
            if exc.code == 429:
                return "rate"
            if exc.code == 400 and "API key" in str(exc.message or ""):
                return "auth"  # 잘못된 키는 400 INVALID_ARGUMENT 로 오기도 한다
            log.error("Gemini API 오류 %s %s: %s", exc.code, exc.status, exc.message)
            return "server"
        if isinstance(exc, errors.ServerError):
            log.error("Gemini API 오류 %s %s: %s", exc.code, exc.status, exc.message)
            return "server"
        if isinstance(exc, _NETWORK_ERRORS):
            return "connection"
        return None

    def _complete(self, system: str, text: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=2000,
            thinking_config=self._thinking("minimal" if self._minimal_ok else "low"),
        )
        try:
            response = self.client.models.generate_content(model=self.cfg.model, contents=text, config=config)
        except errors.ClientError as exc:
            # gemini-3.8-flash·Gemini 4 는 minimal 을 받지 않는다: 한 번 low 로 다시 하고 기억해 둔다
            if exc.code != 400 or not self._minimal_ok or self.legacy:
                raise
            self._minimal_ok = False
            config.thinking_config = self._thinking("low")
            response = self.client.models.generate_content(model=self.cfg.model, contents=text, config=config)
        if not response.candidates or not response.candidates[0].content:
            return ""
        parts = response.candidates[0].content.parts or []
        return "".join(p.text for p in parts if p.text and not p.thought)

    def _describe(self, message: Any) -> list[str]:
        if not isinstance(message, types.Content) or message.role != "model":
            return []  # 도구 결과
        lines: list[str] = []
        text = ""
        for part in message.parts or []:
            if part.text and not part.thought:
                text += part.text
                continue
            if text.strip():
                lines.append(f"스피커: {text.strip()}")
            text = ""
            if part.function_call:
                args = json.dumps(part.function_call.args or {}, ensure_ascii=False)
                lines.append(f"(도구 {part.function_call.name} {args})")
            elif part.tool_call and part.tool_call.tool_type == types.ToolType.GOOGLE_SEARCH_WEB:
                lines.append(f"(도구 web_search {json.dumps(part.tool_call.args or {}, ensure_ascii=False)})")
        if text.strip():
            lines.append(f"스피커: {text.strip()}")
        return lines

    def _run_turn(
        self,
        emit: Callable[[str], None],
        on_tool_start: Callable[[str], None] | None,
        cancel: threading.Event,
        started: float,
    ) -> None:
        splitter, said = self.timed(emit, started)
        announced_tool = False
        malformed_retries = 0
        config = types.GenerateContentConfig(
            system_instruction=self.system_text(),
            tools=self.tools,
            tool_config=self.tool_config,
            thinking_config=self._thinking(self.cfg.effort),
            max_output_tokens=16000,
            # 도구는 우리가 직접 실행한다 (SDK 의 자동 함수 호출 끔)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        rounds = 0
        while rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            parts: list[types.Part] = []
            finish = None
            usage = None
            model_version = None
            block_reason = None
            stream = self.client.models.generate_content_stream(
                model=self.cfg.model, contents=self._contents(), config=config,
            )
            try:
                for chunk in stream:
                    if cancel.is_set():
                        raise _Cancelled
                    usage = chunk.usage_metadata or usage
                    model_version = chunk.model_version or model_version
                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                        block_reason = chunk.prompt_feedback.block_reason
                    if not chunk.candidates:
                        continue
                    candidate = chunk.candidates[0]
                    finish = candidate.finish_reason or finish
                    for part in (candidate.content.parts if candidate.content else None) or []:
                        _add_part(parts, part)
                        if part.text and not part.thought:
                            splitter.feed(part.text)
                        elif (part.function_call or part.tool_call) and not announced_tool and not said():
                            # 검색이나 음악 재생은 몇 초 걸리므로, 시작할 때 알려서 조용히 기다리게 하지 않는다.
                            announced_tool = True
                            if on_tool_start:
                                on_tool_start(part.function_call.name if part.function_call else "web_search")
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()  # 끼어들어 멈춘 경우에도 연결을 닫는다
            if cancel.is_set():
                raise _Cancelled

            if usage is not None:
                self._prompt_tokens = (usage.prompt_token_count or 0) + (usage.tool_use_prompt_token_count or 0)
            log.info(
                "응답 stop=%s 모델=%s 입력=%s(캐시 %s) 출력=%s(생각 %s)",
                getattr(finish, "value", finish) or getattr(block_reason, "value", block_reason),
                model_version or self.cfg.model,
                usage.prompt_token_count if usage else None,
                usage.cached_content_token_count if usage else None,
                usage.candidates_token_count if usage else None,
                usage.thoughts_token_count if usage else None,
            )

            if block_reason is not None or finish in _BLOCKED:
                splitter.flush()
                # 막힌 답 대신 거절한 말을 남겨 두어 사용자·모델 차례가 번갈아 이어지게 한다
                self.messages.append(types.Content(role="model", parts=parts or [types.Part(text=_REFUSAL)]))
                emit(_REFUSAL)
                return

            if finish == types.FinishReason.MALFORMED_FUNCTION_CALL:
                # 함수 호출 형식이 깨진 경우: 이번 답은 버리고 같은 턴을 다시 요청한다.
                malformed_retries += 1
                log.warning("함수 호출 형식이 깨져 다시 요청합니다 (%d번째).", malformed_retries)
                if malformed_retries > 2:
                    splitter.flush()
                    emit("답을 만들다가 문제가 생겼어요. 다시 말씀해 주세요.")
                    return
                rounds -= 1
                continue
            malformed_retries = 0
            if finish == types.FinishReason.MISSING_THOUGHT_SIGNATURE:
                log.error("생각 서명이 빠졌다는 응답을 받았습니다. 대화 기록 처리 확인이 필요합니다.")

            # 서명이 붙은 조각까지 받은 그대로 기록에 남긴다 (다음 요청에 그대로 돌려보내야 한다).
            if parts:
                self.messages.append(types.Content(role="model", parts=parts))

            calls = [p.function_call for p in parts if p.function_call]
            if not calls:
                splitter.flush()
                if finish == types.FinishReason.MAX_TOKENS:
                    log.warning("답이 max_output_tokens 에서 잘렸습니다.")
                elif not parts:
                    log.warning("모델이 빈 답을 보냈습니다 (stop=%s).", finish)
                return

            splitter.flush()  # 도구를 부르기 전에 한 말은 따로 한 문장으로 끝낸다
            results = []
            for call in calls:
                if finish == types.FinishReason.MAX_TOKENS:
                    # 잘린 도구 입력은 실행하지 않는다.
                    output, error = "입력이 중간에 잘렸습니다. 다시 호출해 주세요.", True
                else:
                    output, error = self.run_tool(call.name, call.args or {})
                # 병렬 호출이면 결과도 호출 순서대로 한 Content 에 모아 보낸다 (id 로 짝을 맞춘다).
                results.append(types.Part(function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response={"error" if error else "output": output},
                )))
            self.messages.append(types.Content(role="user", parts=results))
            if cancel.is_set():
                raise _Cancelled

        splitter.flush()
        log.warning("도구 호출이 %d번을 넘어 중단했습니다.", MAX_TOOL_ROUNDS)
        emit("처리가 너무 길어져서 여기서 멈출게요.")
