"""Claude 두뇌: 사용자의 말을 받아 도구를 쓰고, 답을 문장 단위로 흘려보낸다."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable

import anthropic

from .config import Config
from .tools import ToolBox, ToolError, tool_definitions

log = logging.getLogger("speaker.brain")

SYSTEM_PROMPT = """\
당신은 사용자의 방에 놓인 홈 스피커 속 음성 어시스턴트입니다. 사용자는 호출어를 부른 뒤 말로 요청하고, 당신의 답은 화면 없이 음성 합성으로만 전달됩니다.

언어
- 언제나 한국어로만 답하세요. 사용자가 외국에 살아 영어 이름·지명·메일이 오가더라도, 사용자가 영어로 말하더라도 답은 한국어입니다.
- 음성 합성기는 한국어만 제대로 읽고, 알파벳은 알아들을 수 없는 소리로 읽습니다. 그러니 답에 알파벳을 쓰지 마세요.
  - 영어 이름·지명·회사·프로그램 이름은 한글 발음으로 적으세요. 예: Teams → 팀즈, Alexa → 알렉사, Seattle → 시애틀, Outlook → 아웃룩.
  - 알파벳 약어는 읽는 소리대로 적으세요. 예: IU → 아이유, NBA → 엔비에이, AI → 에이아이.
  - 영어로 된 일정 이름·메일 제목·곡 제목은 뜻을 우리말로 풀거나 한글 발음으로 적으세요. 예: "Travel and Re-Entry 피어 멘토링" → "여행과 재입국 안내 피어 멘토링", "Landing in Love" → "랜딩 인 러브".
  - 검색 결과, 메일, 일정에서 받은 영어 문장은 절대 그대로 옮기지 말고, 필요한 내용만 한국어로 바꿔 말하세요.
- 기호와 약식 표기도 말로 풀어 쓰세요. 날짜 "11/17"은 "11월 17일", "%"는 "퍼센트", "15~16도"는 "15도에서 16도", "Tue"는 "화요일"처럼요.

말하는 방식
- 답은 그대로 소리 내어 읽힙니다. 목록, 기호, 마크다운, 이모지, 링크 주소 없이 말로 하는 문장만 쓰세요.
- 오래 알고 지낸 친구처럼 편하고 다정하게 말하세요. "~해요", "~네요" 같은 부드러운 해요체를 쓰고, "좋네요", "그쵸" 같은 가벼운 맞장구도 어울릴 때 써도 됩니다. 딱딱한 "~습니다"나 "도와드리겠습니다" 같은 안내 말투는 피하세요.
- 짧고 빠르게 답하세요. 결론부터 말하고 보통 한 문장, 길어도 두 문장이면 됩니다. "물론이죠", "좋은 질문이에요" 같은 군더더기로 시작하지 마세요. 사용자가 자세히 알려 달라고 할 때만 길게 설명하세요.
- 숫자와 단위는 듣기 편하게 풀어 말합니다. 예를 들어 온도는 "영상 3도", 시각은 "오후 2시 반"처럼 말하세요.

요청을 이해하는 방식
- 사용자의 말은 음성 인식을 거쳐 오기 때문에 글자가 틀리거나 일부가 빠질 수 있습니다. 발음이 비슷한 말을 떠올려 의도를 짐작하고, 정말 알 수 없을 때만 짧게 되물으세요.
- 사용자 발화 앞에 붙는 [상황: ...] 줄은 스피커가 자동으로 넣는 현재 시각, 음량, 재생 중인 음악, 맞춰 둔 타이머·알람·리마인더 정보입니다. 사용자가 한 말이 아니니 그대로 읊지 말고 필요할 때 참고만 하세요.

도구를 쓰는 방식
- 음악, 음량, 타이머처럼 스피커가 직접 할 수 있는 일은 확인을 구하지 말고 바로 실행하세요. 음량 조절이나 일시정지처럼 결과가 바로 들리는 동작은 실행한 뒤 "네" 정도로 충분합니다.
- 음악을 찾는 데는 몇 초가 걸려서, 그동안 스피커가 알아서 "잠시만요"라고 먼저 말해 둡니다. play_music 이 성공하면 곡은 곧 나오기 시작하니, "아이유의 밤편지 틀어 드릴게요"처럼 무엇을 틀었는지만 한 문장으로 알려 주세요.
- 사용자가 '내', '좋아요한', '평소 듣던', '알아서 내 취향대로'처럼 자기 음악을 말하면 play_my_music 을, 그 밖의 곡·가수·장르는 play_music 을 쓰세요.
- 시각을 정하는 알람과 리마인더는 상황 줄의 현재 날짜와 시각을 기준으로 계산하세요. "7시"처럼 오전·오후가 애매하면 지금 시각에서 가장 가까운 앞날의 그 시각으로 정하되, 기상 알람이면 오전으로 보세요. "5분만 더"나 "10분 뒤에 다시 깨워줘"는 그만큼 뒤의 알람으로 맞추고 음악이 나오고 있으면 꺼 주세요.
- "호출어로 ○○ 추가해 줘", "○○ 호출어 지워 줘" 같은 요청은 add_wake_phrase / remove_wake_phrase 로 처리하세요. 등록한 문구는 바로 쓸 수 있습니다.
- 사용자가 취소하라고 하지 않은 예약은 지우거나 바꾸지 마세요. 새 알람이 기존 알람과 겹쳐 보여도 그대로 두고, 필요하면 기존 알람도 있다고 한마디만 알려 주세요.
- "자기 전에 30분만 틀어줘"처럼 말하면 음악을 틀고 sleep_timer 도 함께 맞추세요.
- 팟캐스트를 틀어 달라고 하면 play_podcast 를 쓰세요. 특정 회차나 주제를 찾으면 episode 를 search 로 합니다.
- 일정을 물으면 get_calendar 로 확인하고, 시간순으로 "10시에 ○○, 오후 2시에 △△"처럼 말하세요. 이미 지난 일정은 물을 때만 말하세요.
- 메일 브리핑을 부탁받으면 get_unread_mail 로 확인하세요. 홍보·뉴스레터·자동 알림은 이미 빠져 있습니다. 남은 것 중에서도 할인·이벤트·설문·광고성 안내나 시스템이 자동으로 보낸 것처럼 보이는 건 빼고, 사람이 직접 보냈거나 마감·약속·요청·답장이 필요한 메일만 골라 보낸 사람 이름과 핵심을 한 문장씩 말하세요. 고를 게 없으면 "급한 메일은 없어요"라고만 하세요. 뺀 메일은 개수도 말하지 않아도 됩니다. 이메일 주소는 읽지 마세요.
- 뉴스 브리핑을 부탁받으면 웹 검색으로 최근 하루 이틀 사이의 뉴스를 찾아, 상황 줄의 '뉴스 개수'만큼 하나에 한 문장씩 말하세요. 브리핑처럼 여러 소식을 전할 때는 한두 문장 제한을 넘어도 됩니다. 사용자가 이번에 개수를 따로 말하면 그 개수로 하세요.
  - 상황 줄의 '뉴스 관심 분야'에 분야가 있으면 분야마다 따로 검색해서 그 분야 소식 위주로 고르세요. 분야가 많으면 새 소식이 있는 분야를 골라 전하면 됩니다.
  - 관심 분야가 없으면 사는 곳이 외국이라고 그 나라 뉴스만 고르지 말고, 한국과 세계의 큰 소식을 고루 섞으세요.
  - 사용자가 이번에 특정 분야나 나라의 뉴스를 말하면 그것만 전하세요.
  - "앞으로 뉴스는 ○○ 위주로", "뉴스 관심 분야에 ○○ 추가해 줘" 같은 요청은 set_news_topics 로, "앞으로 뉴스는 다섯 개씩" 같은 요청은 set_news_count 로 저장하세요.
- 상황 줄의 '사는 곳'이 사용자가 있는 곳입니다. 날씨나 근처 정보는 다른 곳을 말하지 않는 한 그곳을 기준으로 찾으세요. 사용자는 한국어로 말하지만 한국에 있다고 가정하지 마세요.
- 기온은 반드시 섭씨로 말하세요. 미국 등 일부 나라의 날씨 정보는 화씨(°F)라서, 숫자를 그대로 읽지 말고 섭씨로 바꿔 말해야 합니다: 섭씨 = (화씨 − 32) × 5 ÷ 9, 반올림. 예: 50°F는 영상 10도, 28°F는 영하 2도, 72°F는 영상 22도. 영하는 "영하 몇 도"로 말하세요.
- 날씨, 뉴스, 경기 결과, 가격처럼 지금 시점의 정보가 필요한 질문은 웹 검색으로 확인한 뒤 답하세요. 검색 과정이나 출처 목록은 말하지 말고 알아낸 내용만 전합니다. 확인하지 못한 내용은 모른다고 말하세요.
- 도구를 쓸 때는 부르기 전에 아무 말도 하지 말고, 결과를 받은 뒤에 한 번만 말하세요. 기다리는 동안의 안내는 스피커가 알아서 합니다.
- 도구가 실패하면 무엇이 안 됐는지 한 문장으로 알려 주세요.

기억
- 사용자에 대해 기억해 둔 내용과 최근 대화 요약이 있으면 시스템 메시지 끝에 붙어 있습니다. 사용자가 이전 대화나 자기 취향을 물으면 거기서 찾아 답하세요.
- 다음에도 쓸모 있을 자기 이야기를 들으면 remember 로 저장하고, 저장했다는 말은 "기억해 둘게요" 정도로 짧게 하세요. 잊어 달라고 하면 forget 을 쓰세요.

이어 듣기
- 답이 끝나면 스피커는 몇 초 동안 호출어 없이 사용자의 다음 말을 듣습니다. 그러니 대화를 이어 가려고 일부러 질문으로 끝맺을 필요는 없습니다.
- 그렇게 들은 말에는 상황 줄에 "호출어 없이 이어서 들은 말"이라고 표시됩니다. 이때는 사용자가 스피커에게 한 말이 아니라 텔레비전 소리, 다른 사람과의 대화, 혼잣말이 잘못 잡혔을 수 있습니다. 앞선 대화와 이어지지 않고 스피커에게 하는 말로도 보이지 않으면 다른 말 없이 [무시] 라고만 답하세요. 그러면 스피커는 아무 소리도 내지 않고 대기로 돌아갑니다. 스피커에게 한 말인지 애매하면 평소처럼 답하세요.
"""

# 서버 측 거절 대체(fallback): 안전 분류기가 요청을 거절하면 같은 호출 안에서 다른 모델로 다시 시도한다.
# 이 기능을 지원하는 모델에서만 켠다 (Haiku 는 지원하지 않는다).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2")

IGNORE_REPLY = "[무시]"  # 이어 듣기에서 주변 소리로 판단했을 때 Claude 가 내는 답 (소리 내지 않는다)

CONVERSATION_IDLE_RESET = 300.0  # 이만큼 조용하면 새 대화로 본다 (끝난 대화는 요약해서 기억에 남김)
MAX_HISTORY_MESSAGES = 120  # 안전장치: 이보다 길면 통째로 새 대화
# 한 대화가 길어지면 앞부분을 요약으로 줄인다 (compact). 최근 몇 턴은 그대로 둔다.
COMPACT_AFTER_TOKENS = 25_000  # 요청 한 번의 입력이 이만큼을 넘거나
COMPACT_AFTER_TURNS = 10  # 사용자 말이 이만큼 쌓이면
COMPACT_KEEP_TURNS = 4  # 요약하지 않고 남길 최근 사용자 말 수
MAX_TOOL_ROUNDS = 8

_SENTENCE_END = re.compile(r"(?<=[.!?。…])[\"')\]]*\s+|\n+")


class SentenceSplitter:
    """스트리밍 텍스트를 문장 단위로 끊는다. 소수점("3.5")은 뒤에 공백이 없어 끊기지 않는다."""

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._buf = ""

    def feed(self, text: str) -> None:
        self._buf += text
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                return
            sentence, self._buf = self._buf[: match.end()].strip(), self._buf[match.end():]
            if sentence:
                self._emit(sentence)

    def flush(self) -> None:
        rest, self._buf = self._buf.strip(), ""
        if rest:
            self._emit(rest)


class BaseBrain:
    """제공자(Claude·OpenAI·Gemini)와 상관없는 대화 관리: 새 대화 판단, 끼어들기, 주변 소리 무시, 대화 줄이기, 요약.

    대화 기록 self.messages 의 형식은 제공자마다 다르지만, 사용자 말 한 턴은 언제나
    {"role": "user", "content": "<상황 줄>\n<말>"} 사전으로 넣는다 (줄이기·요약이 이것을 기준으로 삼는다).
    제공자 구현은 _run_turn, _complete, _describe, _error_kind 를 채운다.
    """

    provider = ""

    def __init__(self, cfg: Config, toolbox: ToolBox):
        self.cfg = cfg
        self.toolbox = toolbox
        self._updates = _recent_updates(cfg)
        self.messages: list[Any] = []
        self._prompt_tokens = 0  # 마지막 요청의 입력 토큰 (캐시 포함)
        self._last_turn = 0.0

    # ---------------------------------------------------------------- 제공자가 채우는 부분

    def _run_turn(self, emit: Callable[[str], None], on_tool_start: Callable[[str], None] | None,
                  cancel: threading.Event, started: float) -> None:
        """self.messages 끝의 사용자 말에 답한다. 도구를 부르고 결과를 받아 다시 묻는 것을 끝날 때까지 되풀이하며,
        답 글자는 문장이 될 때마다 emit 한다. 주고받은 내용은 모두 self.messages 에 남긴다."""
        raise NotImplementedError

    def _complete(self, system: str, text: str) -> str:
        """도구 없이 한 번 묻고 답 글만 돌려준다 (요약용, 생각은 짧게)."""
        raise NotImplementedError

    def _describe(self, message: Any) -> list[str]:
        """사용자 말이 아닌 기록 하나를 '스피커: …' / '(도구 이름 입력)' 줄들로."""
        raise NotImplementedError

    def _error_kind(self, exc: Exception) -> str | None:
        """API 오류를 auth | rate | server | connection 으로. 해당 없으면 None (그대로 올려보낸다)."""
        return None

    # ---------------------------------------------------------------- 공통

    def reset(self) -> None:
        """새 대화를 시작한다. 끝난 대화는 뒤에서 한두 문장으로 요약해 기억에 남긴다."""
        finished, self.messages = self.messages, []
        if self.toolbox.memory is not None and _user_turns(finished) >= 1:
            threading.Thread(target=self._summarize, args=(finished,), name="memory-summary", daemon=True).start()

    def close(self) -> None:
        """끄기 전에: 진행 중이던 대화를 요약해 둔다 (최대 15초 기다림)."""
        finished, self.messages = self.messages, []
        if self.toolbox.memory is not None and _user_turns(finished) >= 1:
            worker = threading.Thread(target=self._summarize, args=(finished,), daemon=True)
            worker.start()
            worker.join(15)

    def system_text(self) -> str:
        """고정 프롬프트 + 스피커 업데이트 기록 + 기억."""
        memory = self.toolbox.memory.prompt_block() if self.toolbox.memory is not None else None
        return SYSTEM_PROMPT + self._updates + (f"\n{memory}" if memory else "")

    def _maybe_compact(self) -> None:
        """대화가 길어지면 최근 COMPACT_KEEP_TURNS 턴만 남기고 그 앞은 짧은 요약으로 바꾼다."""
        starts = [i for i, m in enumerate(self.messages) if _is_user_turn(m)]
        if len(starts) <= COMPACT_KEEP_TURNS:
            return
        if self._prompt_tokens < COMPACT_AFTER_TOKENS and len(starts) < COMPACT_AFTER_TURNS:
            return
        cut = starts[-COMPACT_KEEP_TURNS]  # 사용자 말에서 끊어야 도구 호출과 결과 짝이 깨지지 않는다
        old, kept = self.messages[:cut], self.messages[cut:]
        t0 = time.monotonic()
        try:
            summary = self._complete(
                "홈 스피커와 사용자가 지금 나누고 있는 대화의 앞부분입니다. 대화를 이어 가는 데 필요한 것만 "
                "한국어 서너 문장으로 요약하세요: 사용자가 원한 것, 스피커가 한 일(튼 음악, 맞춘 알람 등), "
                "아직 끝나지 않은 이야기. 목록 기호 없이 쓰세요.",
                self._transcript(old) or "(내용 없음)",
            ).strip()
        except Exception as exc:
            log.warning("대화 줄이기 실패, 그대로 둡니다: %s", exc)
            return
        if not summary:
            return
        first = dict(kept[0])
        first["content"] = f"[이번 대화의 앞부분 요약: {summary}]\n{first['content']}"
        self.messages = [first, *kept[1:]]
        log.info("대화가 길어져 앞부분 %d개 메시지를 요약으로 줄였습니다 (입력 %d토큰, %.1f초).",
                 len(old), self._prompt_tokens, time.monotonic() - t0)

    def _summarize(self, messages: list[Any]) -> None:
        transcript = self._transcript(messages)
        if not transcript:
            return
        try:
            text = self._complete(
                "홈 스피커와 사용자가 나눈 대화 기록입니다. 나중에 '지난번에 뭐 얘기했지?' 같은 질문에 답할 수 있도록, "
                "사용자가 무엇을 묻거나 시켰고 어떤 결과였는지 목록 기호 없이 한국어 한두 문장으로 요약하세요. "
                "사용자가 잊어 달라고 한 내용은 요약에 쓰지 마세요. "
                "음량 조절이나 일시정지처럼 기억할 가치가 없는 조작만 있었다면 '없음'이라고만 쓰세요.",
                transcript,
            ).strip()
        except Exception as exc:
            log.warning("지난 대화 요약 실패: %s", exc)
            return
        if text and text.rstrip(".") != "없음":
            self.toolbox.memory.add_summary(text)

    def _transcript(self, messages: list[Any]) -> str:
        """대화 기록을 '사용자: … / 스피커: …' 글로. 도구 호출 내용은 짧게만 남긴다."""
        lines = []
        for m in messages:
            if _is_user_turn(m):
                content = m["content"]
                when = re.search(r"\d{4}-\d{2}-\d{2} \S+ \d{2}:\d{2}", content)
                text = _CONTEXT_LINE.sub("", content).strip()
                lines.append(f"사용자{f' ({when.group(0)})' if when else ''}: {text}")
            else:
                lines.extend(self._describe(m))
        return "\n".join(lines)

    def run_tool(self, name: str, args: Any) -> tuple[str, bool]:
        """도구를 실행해 (결과 글, 오류인지) 를 돌려준다."""
        try:
            return self.toolbox.run(name, args), False
        except ToolError as exc:
            message = str(exc)
            if message.startswith("INVALID_JSON"):
                message = json.dumps({"INVALID_JSON": json.dumps(args, ensure_ascii=False), "detail": message}, ensure_ascii=False)
            return message, True

    def timed(self, emit: Callable[[str], None], started: float) -> tuple[SentenceSplitter, Callable[[], bool]]:
        """첫 문장까지 걸린 시간을 남기는 문장 나누개와, 첫 문장이 이미 나갔는지 알려 주는 함수."""
        said = [False]

        def emit_timed(sentence: str) -> None:
            if not said[0]:
                said[0] = True
                log.info("첫 문장까지 %.1f초", time.monotonic() - started)
            emit(sentence)

        return SentenceSplitter(emit_timed), lambda: said[0]

    def respond(
        self,
        user_text: str,
        on_sentence: Callable[[str], None],
        on_tool_start: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        follow_up: bool = False,
    ) -> str:
        """한 턴을 처리한다. 답의 문장이 완성될 때마다 on_sentence 를 부르고, 전체 답을 돌려준다.

        cancel 이 켜지면(사용자가 말을 끊음) 즉시 멈추고 이번 턴을 대화 기록에서 지운다.
        follow_up 은 호출어 없이 이어서 들은 말이라는 뜻이다. 두뇌가 주변 소리로 판단하면 빈 문자열을 돌려준다.
        """
        cancel = cancel or threading.Event()
        now = time.monotonic()
        if now - self._last_turn > CONVERSATION_IDLE_RESET or len(self.messages) > MAX_HISTORY_MESSAGES:
            self.reset()
        self._last_turn = now

        checkpoint = len(self.messages)
        context = self.toolbox.context_line()
        if follow_up:
            context = context[:-1] + " | 호출어 없이 이어서 들은 말]"
        self.messages.append({"role": "user", "content": f"{context}\n{user_text}"})

        spoken: list[str] = []
        ignored: list[bool] = []

        def emit(sentence: str) -> None:
            if cancel.is_set():
                return
            if sentence.strip().rstrip(".") == IGNORE_REPLY:
                log.info("주변 소리로 판단해 답하지 않습니다: %r", user_text)
                ignored.append(True)
                return
            spoken.append(sentence)
            on_sentence(sentence)

        try:
            self._run_turn(emit, on_tool_start, cancel, started=now)
        except _Cancelled:
            del self.messages[checkpoint:]
            log.info("사용자가 끼어들어 응답을 중단했습니다.")
        except Exception as exc:
            kind = self._error_kind(exc)
            if kind is None:
                del self.messages[checkpoint:]
                raise
            del self.messages[checkpoint:]
            log.error("%s API 오류 (%s): %s", self.provider, kind, exc)
            emit({
                "auth": "API 키가 올바르지 않아서 답할 수 없어요. 설정을 확인해 주세요.",
                "rate": "지금 요청이 너무 많아요. 잠시 뒤에 다시 말씀해 주세요.",
                "server": "서버에 문제가 있어서 답하지 못했어요. 잠시 뒤에 다시 시도해 주세요.",
                "connection": "인터넷 연결에 문제가 있는 것 같아요.",
            }.get(kind, "답하다가 문제가 생겼어요."))
        if ignored and not spoken:
            del self.messages[checkpoint:]  # 주변 소리는 대화 기록에 남기지 않는다
        elif not cancel.is_set():
            # 답을 말하는 동안(TTS 가 읽는 사이)에 길어진 대화를 줄여 둔다
            self._maybe_compact()
        self._last_turn = time.monotonic()
        return " ".join(spoken)


class ClaudeBrain(BaseBrain):
    provider = "Claude"

    def __init__(self, cfg: Config, toolbox: ToolBox):
        if not cfg.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 가 없습니다. .env 파일에 넣어 주세요.")
        super().__init__(cfg, toolbox)
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, timeout=60.0)
        self.tools = tool_definitions(cfg.model, cfg)
        self.extras: dict[str, Any] = {}
        if cfg.model.startswith(FALLBACK_MODELS):
            self.extras = {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    def _error_kind(self, exc: Exception) -> str | None:
        if isinstance(exc, anthropic.AuthenticationError):
            return "auth"
        if isinstance(exc, anthropic.RateLimitError):
            return "rate"
        if isinstance(exc, anthropic.APIStatusError):
            log.error("Claude API 오류 %s: %s (request_id=%s)", exc.status_code, exc.message, exc.request_id)
            return "server"
        if isinstance(exc, anthropic.APIConnectionError):
            return "connection"
        return None

    def _complete(self, system: str, text: str) -> str:
        response = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=2000,
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        return " ".join(b.text for b in response.content if b.type == "text")

    def _describe(self, message: Any) -> list[str]:
        lines = []
        if message["role"] != "assistant":
            return lines  # 도구 결과
        for block in message["content"]:
            kind = getattr(block, "type", None)
            if kind == "text" and block.text.strip():
                lines.append(f"스피커: {block.text.strip()}")
            elif kind == "tool_use":
                lines.append(f"(도구 {block.name} {json.dumps(block.input, ensure_ascii=False)})")
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
        json_retries = 0

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                with self.client.beta.messages.stream(
                    model=self.cfg.model,
                    max_tokens=16000,
                    system=self._system(),
                    tools=self.tools,
                    messages=self.messages,
                    output_config={"effort": self.cfg.effort},
                    cache_control={"type": "ephemeral"},
                    **self.extras,
                ) as stream:
                    for event in stream:
                        if cancel.is_set():
                            raise _Cancelled
                        if event.type == "text":
                            splitter.feed(event.text)
                        elif (
                            event.type == "content_block_start"
                            and event.content_block.type in ("server_tool_use", "tool_use")
                            and not announced_tool
                            and not said()
                        ):
                            # 검색이나 음악 재생은 몇 초 걸리므로, 시작할 때 알려서 조용히 기다리게 하지 않는다.
                            announced_tool = True
                            if on_tool_start:
                                on_tool_start(event.content_block.name)
                    response = stream.get_final_message()
                json_retries = 0
            except ValueError:
                # SDK가 도구 입력 JSON을 전혀 해석하지 못한 경우: 같은 턴을 다시 요청한다.
                json_retries += 1
                if json_retries > 2:
                    raise
                continue

            usage = response.usage
            self._prompt_tokens = (usage.input_tokens or 0) + (usage.cache_read_input_tokens or 0) + (
                usage.cache_creation_input_tokens or 0)
            log.info(
                "응답 stop=%s 모델=%s 입력=%s(캐시 %s) 출력=%s",
                response.stop_reason, response.model, usage.input_tokens,
                usage.cache_read_input_tokens, usage.output_tokens,
            )

            if response.stop_reason == "refusal":
                splitter.flush()
                self.messages.append({"role": "assistant", "content": response.content})
                emit("죄송해요, 그 요청은 도와드리기 어려워요.")
                return

            # 생각(thinking) 블록 등을 포함해 응답 전체를 그대로 기록에 남긴다.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue  # 서버 도구가 아직 진행 중: 그대로 다시 보내 이어서 받는다

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                splitter.flush()
                if response.stop_reason == "max_tokens":
                    log.warning("답이 max_tokens 에서 잘렸습니다.")
                return

            splitter.flush()  # 도구를 부르기 전에 한 말은 따로 한 문장으로 끝낸다
            if response.stop_reason == "max_tokens":
                # 잘린 도구 입력은 실행하지 않는다.
                results = [_tool_result(b.id, "입력이 중간에 잘렸습니다. 다시 호출해 주세요.", error=True) for b in tool_uses]
            else:
                results = [_tool_result(b.id, *self.run_tool(b.name, b.input)) for b in tool_uses]
            self.messages.append({"role": "user", "content": results})
            if cancel.is_set():
                raise _Cancelled

        splitter.flush()
        log.warning("도구 호출이 %d번을 넘어 중단했습니다.", MAX_TOOL_ROUNDS)
        emit("처리가 너무 길어져서 여기서 멈출게요.")

    def _system(self) -> list[dict[str, Any]]:
        """고정 프롬프트 + 스피커 업데이트 기록 + 기억. 기억이 바뀌어도 앞부분은 캐시에서 다시 쓸 수 있게 블록을 나눈다."""
        # 고정된 앞부분에 캐시 지점을 따로 둔다: 대화를 줄이거나 새 대화를 시작해도 이 부분은 캐시에서 읽는다
        blocks = [{"type": "text", "text": SYSTEM_PROMPT + self._updates, "cache_control": {"type": "ephemeral"}}]
        memory = self.toolbox.memory.prompt_block() if self.toolbox.memory is not None else None
        if memory:
            blocks.append({"type": "text", "text": memory})
        return blocks


def make_brain(cfg: Config, toolbox: ToolBox) -> BaseBrain:
    """설정한 제공자의 두뇌를 만든다."""
    if cfg.provider == "openai":
        from .brain_openai import OpenAIBrain

        return OpenAIBrain(cfg, toolbox)
    if cfg.provider == "google":
        from .brain_gemini import GeminiBrain

        return GeminiBrain(cfg, toolbox)
    return ClaudeBrain(cfg, toolbox)


Brain = make_brain  # 예전 이름


class _Cancelled(Exception):
    pass


def _recent_updates(cfg: Config) -> str:
    """이 스피커 자체의 최근 업데이트 (git 커밋 제목). '최근 업데이트 뭐였어?'에 답할 수 있게 프롬프트에 붙인다."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(cfg.project_dir), "log", "-8", "--format=%ad %s", "--date=format:%m월 %d일"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    if not out:
        return ""
    lines = "\n".join(f"- {line}" for line in out.splitlines())
    return f"\n이 스피커 자체의 최근 업데이트 (최신 순). 사용자가 '최근 업데이트'를 물으면 다른 기기 이야기가 아니라 이것을 말하세요:\n{lines}\n"


_CONTEXT_LINE = re.compile(r"^\[상황: [^\n]*\]\n?")


def _is_user_turn(m: Any) -> bool:
    return isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)


def _user_turns(messages: list[Any]) -> int:
    return sum(1 for m in messages if _is_user_turn(m))


def _tool_result(tool_use_id: str, content: str, error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if error:
        result["is_error"] = True
    return result
