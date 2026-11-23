"""OpenAI 두뇌: Responses API 로 ClaudeBrain 과 같은 일을 한다 (도구 호출, 웹 검색, 문장 단위 스트리밍)."""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable

import openai

from .brain import MAX_TOOL_ROUNDS, BaseBrain, _Cancelled
from .config import Config
from .tools import TOOL_DEFINITIONS, ToolBox

log = logging.getLogger("speaker.brain")

# reasoning 매개변수를 받는 모델: o 시리즈와 gpt-5 이후 (gpt-5-chat 같은 대화 전용 모델은 제외).
# gpt-4.1 · gpt-4o 같은 모델에 reasoning 을 보내면 400 오류가 나므로 이름으로 가린다.
_REASONING_MODEL = re.compile(r"^(o\d|gpt-([5-9]|\d{2}))")
_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class _ResponseFailed(RuntimeError):
    """서버가 응답 도중 response.failed 를 보낸 경우."""


def _function_tools() -> list[dict[str, Any]]:
    """Anthropic 형식 도구 정의를 Responses API 함수 도구로 바꾼다.

    선택 항목이 있는 스키마가 많아 strict 모드는 끈다 (strict 는 모든 항목이 required 여야 한다).
    대신 실행 전에 ToolBox._validate 가 스키마를 직접 확인한다.
    """
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
            "strict": False,
        }
        for t in TOOL_DEFINITIONS
    ]


def _web_search_tool(cfg: Config) -> dict[str, Any]:
    """OpenAI 서버에서 도는 웹 검색 도구. 사는 곳이 설정돼 있으면 그 지역 기준으로 찾는다."""
    search: dict[str, Any] = {"type": "web_search"}
    if cfg.location_city or cfg.location_country or cfg.timezone:
        loc: dict[str, Any] = {"type": "approximate"}
        for key, value in (("city", cfg.location_city), ("region", cfg.location_region),
                           ("country", cfg.location_country), ("timezone", cfg.timezone)):
            if value:
                loc[key] = value
        search["user_location"] = loc
    return search


def _item_dict(item: Any) -> dict[str, Any]:
    """응답의 출력 항목을 다음 요청에 그대로 다시 보낼 수 있는 사전으로 (생각의 암호문 등 포함)."""
    if isinstance(item, dict):
        return item
    return item.to_dict(mode="json", exclude_none=True)


class OpenAIBrain(BaseBrain):
    provider = "OpenAI"

    def __init__(self, cfg: Config, toolbox: ToolBox):
        if not cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY 가 없습니다. .env 파일에 넣어 주세요.")
        super().__init__(cfg, toolbox)
        self.client = openai.OpenAI(api_key=cfg.openai_api_key, timeout=60.0)
        self.tools = [*_function_tools(), _web_search_tool(cfg)]
        self.reasoning = bool(_REASONING_MODEL.match(cfg.model.lower())) and "chat" not in cfg.model.lower()

    # ---------------------------------------------------------------- 요청 옵션

    def _reasoning_args(self, effort: str) -> dict[str, Any]:
        """생각하는 모델에만 reasoning 을 붙인다. 대화 기록을 서버에 남기지 않으므로(store=False)
        생각 내용은 암호문(encrypted_content)으로 받아 다음 요청에 되돌려 보낸다."""
        if not self.reasoning:
            return {}
        effort = effort.lower() if effort and effort.lower() in _EFFORTS else "low"
        return {"reasoning": {"effort": effort}, "include": ["reasoning.encrypted_content"]}

    # ---------------------------------------------------------------- BaseBrain 구현

    def _error_kind(self, exc: Exception) -> str | None:
        if isinstance(exc, openai.AuthenticationError):
            return "auth"
        if isinstance(exc, openai.RateLimitError):
            return "rate"
        if isinstance(exc, openai.APIStatusError):
            log.error("OpenAI API 오류 %s: %s (request_id=%s)", exc.status_code, exc.message, exc.request_id)
            return "server"
        if isinstance(exc, openai.APIConnectionError):
            return "connection"
        if isinstance(exc, (openai.APIError, _ResponseFailed)):
            # 스트리밍 도중 서버가 보낸 오류 이벤트 (상태 코드 없음)
            return "server"
        return None

    def _complete(self, system: str, text: str) -> str:
        response = self.client.responses.create(
            model=self.cfg.model,
            instructions=system,
            input=[{"role": "user", "content": text}],
            max_output_tokens=4000,
            store=False,
            **({"reasoning": {"effort": "low"}} if self.reasoning else {}),
        )
        return response.output_text

    def _describe(self, message: Any) -> list[str]:
        if not isinstance(message, dict):
            return []
        kind = message.get("type")
        if kind == "message" and message.get("role") == "assistant":
            text = " ".join(
                part.get("text", "") for part in message.get("content") or []
                if isinstance(part, dict) and part.get("type") == "output_text"
            ).strip()
            return [f"스피커: {text}"] if text else []
        if kind == "function_call":
            args = message.get("arguments") or ""
            try:
                args = json.dumps(json.loads(args), ensure_ascii=False)
            except ValueError:
                pass
            return [f"(도구 {message.get('name')} {args})"]
        return []  # 생각, 웹 검색, 도구 결과

    def _run_turn(
        self,
        emit: Callable[[str], None],
        on_tool_start: Callable[[str], None] | None,
        cancel: threading.Event,
        started: float,
    ) -> None:
        splitter, said = self.timed(emit, started)
        announced_tool = False

        for _ in range(MAX_TOOL_ROUNDS):
            response = None
            done_items: dict[int, Any] = {}
            with self.client.responses.create(
                model=self.cfg.model,
                instructions=self.system_text(),
                input=self.messages,
                tools=self.tools,
                max_output_tokens=16000,
                store=False,
                stream=True,
                prompt_cache_key="mac-speaker",
                **self._reasoning_args(self.cfg.effort),
            ) as stream:
                for event in stream:
                    if cancel.is_set():
                        raise _Cancelled
                    kind = event.type
                    if kind == "response.output_text.delta":
                        splitter.feed(event.delta)
                    elif kind == "response.output_item.added":
                        item_type = getattr(event.item, "type", None)
                        if item_type in ("function_call", "web_search_call") and not announced_tool and not said():
                            # 검색이나 음악 재생은 몇 초 걸리므로, 시작할 때 알려서 조용히 기다리게 하지 않는다.
                            announced_tool = True
                            if on_tool_start:
                                on_tool_start(getattr(event.item, "name", None) or "web_search")
                    elif kind == "response.output_item.done":
                        done_items[event.output_index] = event.item
                        if getattr(event.item, "type", None) == "message":
                            splitter.flush()  # 말 한 덩어리가 끝났다
                    elif kind in ("response.completed", "response.incomplete", "response.failed"):
                        response = event.response
                    elif kind == "error":
                        raise _ResponseFailed(f"{getattr(event, 'code', None)}: {getattr(event, 'message', '')}")
            if cancel.is_set():
                raise _Cancelled
            if response is None:
                raise _ResponseFailed("응답이 끝나지 않은 채 스트림이 닫혔습니다.")
            if response.status == "failed":
                error = getattr(response, "error", None)
                raise _ResponseFailed(f"{getattr(error, 'code', None)}: {getattr(error, 'message', '')}")

            # 마지막 이벤트에 output 이 빠져 있으면 하나씩 끝난 항목들로 채운다.
            output = list(response.output or []) or [done_items[i] for i in sorted(done_items)]

            usage = response.usage
            if usage is not None:
                self._prompt_tokens = usage.input_tokens or 0  # OpenAI 는 캐시 토큰이 입력 토큰에 포함된다
                cached = getattr(usage.input_tokens_details, "cached_tokens", None)
                reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", None)
                log.info(
                    "응답 stop=%s 모델=%s 입력=%s(캐시 %s) 출력=%s(생각 %s)",
                    response.status, response.model, usage.input_tokens, cached,
                    usage.output_tokens, reasoning_tokens,
                )
            else:
                log.info("응답 stop=%s 모델=%s", response.status, response.model)

            # 생각(reasoning) 항목 등을 포함해 응답 전체를 순서대로 기록에 남긴다 (store=False 라 매번 다시 보낸다).
            self.messages.extend(_item_dict(item) for item in output)

            incomplete = response.status == "incomplete"
            reason = getattr(response.incomplete_details, "reason", None) if incomplete else None
            refused = reason == "content_filter" or any(
                getattr(part, "type", None) == "refusal"
                for item in output if getattr(item, "type", None) == "message"
                for part in item.content
            )
            if refused:
                splitter.flush()
                emit("죄송해요, 그 요청은 도와드리기 어려워요.")
                return

            calls = [item for item in output if getattr(item, "type", None) == "function_call"]
            if not calls:
                splitter.flush()
                if incomplete:
                    log.warning("답이 중간에 잘렸습니다 (%s).", reason)
                    if not said():
                        emit("답이 너무 길어져서 끊겼어요.")
                return

            splitter.flush()  # 도구를 부르기 전에 한 말은 따로 한 문장으로 끝낸다
            for call in calls:
                if incomplete:
                    # 잘린 도구 입력은 실행하지 않는다.
                    result = "입력이 중간에 잘렸습니다. 다시 호출해 주세요."
                else:
                    result = self._call_tool(call)
                self.messages.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
            if cancel.is_set():
                raise _Cancelled

        splitter.flush()
        log.warning("도구 호출이 %d번을 넘어 중단했습니다.", MAX_TOOL_ROUNDS)
        emit("처리가 너무 길어져서 여기서 멈출게요.")

    def _call_tool(self, call: Any) -> str:
        """함수 호출 하나를 실행해 모델에게 돌려줄 글을 만든다. 오류도 글로 알려 모델이 고쳐 부르게 한다."""
        try:
            args = json.loads(call.arguments or "{}")
        except ValueError as exc:
            return json.dumps({"error": f"INVALID_JSON: 입력을 JSON 으로 읽지 못했습니다 ({exc}). 다시 호출해 주세요.",
                               "arguments": call.arguments}, ensure_ascii=False)
        result, is_error = self.run_tool(call.name, args)
        return f"오류: {result}" if is_error else result
