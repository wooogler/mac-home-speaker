"""Claude가 호출하는 도구들: 음악, 음량, 타이머·알람·리마인더, 기억. (웹 검색은 Anthropic 서버에서 실행된다.)"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import logging
import re
import subprocess
import time
from typing import Any

from .config import Config
from .schedule import Event, Scheduler, next_occurrence, spoken_duration

log = logging.getLogger("speaker.tools")

# 스트리밍 요청이므로 eager_input_streaming 을 켠다. 이 경우 서버가 입력을 검증하지 않으므로
# 실행 전에 ToolBox._validate 에서 스키마를 직접 확인한다.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "play_music",
        "description": (
            "YouTube Music에서 음악을 찾아 스피커로 재생한다. 사용자가 노래, 가수, 장르, 분위기를 "
            "말하며 틀어 달라고 하면 되묻지 말고 바로 호출한다. 이미 다른 곡이 나오고 있으면 교체된다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어. 예: '아이유 밤편지', '뉴진스', '비 오는 날 듣기 좋은 재즈'",
                },
                "kind": {
                    "type": "string",
                    "enum": ["song", "artist", "playlist"],
                    "description": (
                        "song: 특정 곡(이어서 비슷한 곡이 자동 재생됨). artist: 가수의 인기곡 모음. "
                        "playlist: 장르·분위기·상황에 맞는 플레이리스트."
                    ),
                },
            },
            "required": ["query", "kind"],
        },
    },
    {
        "name": "play_my_music",
        "description": (
            "사용자의 유튜브 뮤직 계정에 있는 음악을 재생한다. '좋아요한 노래', '내 ○○ 재생목록', "
            "'내 취향대로/알아서 틀어줘', '요즘/최근 듣던 거'처럼 사용자 자신의 음악을 말할 때 쓴다. "
            "계정이 연결되어 있지 않으면 실패하니, 그때는 play_music 으로 대신하지 말고 연결이 필요하다고 알린다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["liked", "playlist", "mix", "recent"],
                    "description": "liked: 좋아요한 곡. playlist: 보관함의 재생목록(name 필요). mix: 맞춤 추천. recent: 최근 들은 곡.",
                },
                "name": {"type": "string", "description": "source 가 playlist 일 때 재생목록 이름 (들은 그대로)"},
                "shuffle": {"type": "boolean", "description": "섞어서 재생할지. 좋아요한 곡은 보통 섞는 편이 낫다."},
            },
            "required": ["source"],
        },
    },
    {
        "name": "play_podcast",
        "description": (
            "유튜브 뮤직에서 팟캐스트를 찾아 재생한다. 프로그램 이름이나 주제를 query 로 넣는다. "
            "보통은 최신 회차(latest)부터 틀고, 특정 회차나 주제의 에피소드를 찾으면 episode 를 search 로 한다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "사용자가 말한 팟캐스트 이름이나 주제 그대로. '팟캐스트', '최신' 같은 말은 빼고, 이름을 늘려 쓰지 않는다. 예: '세바시', 'KBS 뉴스', '경제'"},
                "episode": {"type": "string", "enum": ["latest", "search"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_my_playlists",
        "description": "사용자의 유튜브 뮤직 보관함에 있는 재생목록 이름을 알려 준다. 어떤 재생목록이 있는지 물을 때 쓴다.",
        "eager_input_streaming": True,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "music_control",
        "description": "재생 중인 음악을 제어한다: 일시정지, 다시 재생, 완전히 끄기, 다음 곡, 이전 곡.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["pause", "resume", "stop", "next", "previous"]},
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_calendar",
        "description": "사용자의 캘린더 일정을 가져온다 (맥 캘린더에 연결된 학교 계정 등). 오늘·내일·이번 주 일정이나 특정 날짜 일정을 물을 때 쓴다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "enum": ["today", "tomorrow", "week"]},
                "date": {"type": "string", "description": "'YYYY-MM-DD'. 앞으로 일주일 안의 특정 날짜일 때"},
            },
            "required": [],
        },
    },
    {
        "name": "get_unread_mail",
        "description": "안 읽은 메일 목록(보낸 사람, 제목, 본문 앞부분)을 가져온다. 메일 브리핑이나 새 메일을 물을 때 쓴다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 15}},
            "required": [],
        },
    },
    {
        "name": "remember",
        "description": (
            "사용자에 대해 앞으로도 기억해 둘 만한 사실을 저장한다. 사용자가 '기억해 줘'라고 하거나, 이름·호칭, "
            "좋아하는 음악이나 가수, 생활 습관, 반복되는 일정, 알레르기처럼 다음 대화에서도 쓸모 있는 자기 이야기를 하면 쓴다. "
            "그날 한 번뿐인 일이나 잡담은 저장하지 않는다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "짧은 한 문장. 예: '사용자는 아침에 재즈를 즐겨 듣는다'"},
            },
            "required": ["fact"],
        },
    },
    {
        "name": "forget",
        "description": "저장한 기억을 지운다. about 에 지울 내용의 핵심 단어를 넣는다. 사용자가 전부 잊으라고 하면 about 을 비운다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"about": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "set_volume",
        "description": (
            "음량을 0~100 사이 값으로 맞춘다. target 으로 대상을 고른다. "
            "music: 음악·팟캐스트만 ('노래 소리 줄여', '음악 좀 키워'). voice: 스피커(너)의 목소리와 알림음만 ('네 목소리 너무 커', '말소리 키워'). "
            "system: 맥 전체 음량 ('소리 줄여'처럼 대상을 말하지 않았을 때). "
            "'조금 키워줘' 같은 상대적 요청은 상황 정보에 적힌 그 대상의 현재 음량에서 10~15 정도 더하거나 빼서 계산한다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "percent": {"type": "integer", "minimum": 0, "maximum": 100},
                "target": {"type": "string", "enum": ["system", "music", "voice"]},
            },
            "required": ["percent"],
        },
    },
    {
        "name": "add_wake_phrase",
        "description": (
            "스피커를 부르는 호출 문구를 새로 등록한다 ('호출어로 ○○ 추가해 줘'). phrase 에는 사용자가 정한 문구만 그대로 넣는다. "
            "너무 짧거나 평소 대화에 자주 나오는 말이면 등록은 하되 잘못 깨어날 수 있다고 한마디 해 준다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"phrase": {"type": "string", "description": "예: '자비스 일어나', '스피커야'"}},
            "required": ["phrase"],
        },
    },
    {
        "name": "remove_wake_phrase",
        "description": "등록된 호출 문구를 지운다. phrase 는 지울 문구 (들은 그대로, 비슷하면 됨).",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"phrase": {"type": "string"}},
            "required": ["phrase"],
        },
    },
    {
        "name": "list_wake_phrases",
        "description": "지금 스피커를 부를 수 있는 호출 문구 목록을 알려 준다.",
        "eager_input_streaming": True,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_voice_speed",
        "description": "스피커(너)의 말 속도를 바꾼다. 1.0 이 기본 모델 속도, 1.2 면 20% 빠르게. '좀 빠르게 말해' 같은 상대적 요청은 상황 줄의 현재 속도에서 0.1~0.15 정도 조절한다. 0.7~1.6 사이.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"speed": {"type": "number", "minimum": 0.7, "maximum": 1.6}},
            "required": ["speed"],
        },
    },
    {
        "name": "set_news_topics",
        "description": (
            "뉴스 브리핑에서 챙길 관심 분야를 정한다. topics 는 바꾼 뒤의 전체 목록이라, 하나를 더하거나 뺄 때는 "
            "상황 줄의 '뉴스 관심 분야'에 있는 지금 목록에서 고쳐서 넘긴다. 빈 목록이면 관심 분야를 지운다 (한국·세계 주요 뉴스). "
            "각 분야는 '인공지능', '한국 경제', '축구', '반도체'처럼 짧은 검색어로."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"topics": {"type": "array", "items": {"type": "string"}, "maxItems": 8}},
            "required": ["topics"],
        },
    },
    {
        "name": "set_news_count",
        "description": "뉴스 브리핑에서 전할 소식 개수를 정한다 (1~10, 기본 4). '뉴스 좀 더 많이', '뉴스는 두 개만' 같은 요청에 쓴다. 상대적 요청은 상황 줄의 지금 개수에서 조절한다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["count"],
        },
    },
    {
        "name": "turn_off_display",
        "description": "이 맥(스피커)의 화면을 끈다. 스피커 기능은 그대로 동작하고, 맥의 키보드나 트랙패드를 건드리면 화면이 다시 켜진다. '화면 꺼 줘' 같은 요청에 쓴다.",
        "eager_input_streaming": True,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_timer",
        "description": "몇 분 뒤처럼 남은 시간으로 정하는 타이머. 시간이 되면 알림음과 함께 이름을 말해 준다. 특정 시각에 깨우는 건 set_alarm 을 쓴다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "minimum": 1, "description": "지금부터 울릴 때까지의 초"},
                "label": {"type": "string", "description": "타이머 이름. 예: '라면', '10분'"},
            },
            "required": ["seconds", "label"],
        },
    },
    {
        "name": "set_alarm",
        "description": (
            "특정 시각에 울리는 알람 (기상 알람 등). time 은 24시간제 'HH:MM'. date 를 비우면 다음에 오는 그 시각. "
            "음악으로 깨워 달라고 하면 music 에 검색어를, 좋아요한 곡이면 'liked', 맞춤 추천이면 'mix' 를 넣는다. "
            "아침 알람에 날씨나 뉴스를 알려 달라고 하면 briefing 을 true 로."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "'07:30' 같은 24시간제 시각"},
                "date": {"type": "string", "description": "'YYYY-MM-DD'. 오늘이나 내일이 아니면 넣는다"},
                "repeat": {"type": "string", "enum": ["none", "daily", "weekdays", "weekends"]},
                "label": {"type": "string", "description": "알람 이름. 예: '기상', '출근'"},
                "music": {"type": "string", "description": "알람으로 틀 음악. 비우면 알림음"},
                "briefing": {"type": "boolean", "description": "울린 뒤 날씨·뉴스 브리핑을 할지"},
            },
            "required": ["time"],
        },
    },
    {
        "name": "set_reminder",
        "description": (
            "정해진 때에 말로 알려 줄 리마인더. '9시에 약 먹으라고 알려줘', '30분 뒤에 빨래 꺼내라고 해줘' 같은 요청에 쓴다. "
            "시각이면 time('HH:MM', 24시간제)과 필요하면 date 를, 남은 시간이면 in_minutes 를 넣는다."
        ),
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "알릴 내용. 예: '약 드실 시간이에요'"},
                "time": {"type": "string"},
                "date": {"type": "string"},
                "in_minutes": {"type": "integer", "minimum": 1},
                "repeat": {"type": "string", "enum": ["none", "daily", "weekdays", "weekends"]},
            },
            "required": ["message"],
        },
    },
    {
        "name": "sleep_timer",
        "description": "몇 분 뒤에 음악을 천천히 줄이다가 끈다 ('30분 뒤에 음악 꺼줘'). minutes 를 0 으로 하면 수면 타이머를 취소한다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"minutes": {"type": "integer", "minimum": 0, "maximum": 600}},
            "required": ["minutes"],
        },
    },
    {
        "name": "cancel_schedule",
        "description": "타이머·알람·리마인더·수면 타이머를 취소한다. kind 와 label(이름이나 내용 일부)로 고르고, 둘 다 비우면 전부 취소한다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["timer", "alarm", "reminder", "sleep"]},
                "label": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "list_schedule",
        "description": "맞춰 둔 타이머·알람·리마인더 목록과 남은 시간을 알려 준다.",
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["timer", "alarm", "reminder", "sleep"]}},
            "required": [],
        },
    },
]


def tool_definitions(model: str, cfg: Config | None = None) -> list[dict[str, Any]]:
    """Claude 에 보낼 도구 목록: 위의 도구들 + 웹 검색(서버 도구, 날씨·뉴스 같은 최신 정보용).

    결과를 걸러 주는 새 검색 도구는 Opus·Sonnet 급에서만 쓸 수 있어서 Haiku 에는 기본형을 준다.
    사는 곳이 설정돼 있으면 검색 결과도 그 지역 기준으로 받는다.
    """
    search_type = "web_search_20250305" if "haiku" in model else "web_search_20260209"
    search: dict[str, Any] = {"type": search_type, "name": "web_search", "max_uses": 3}
    if cfg is not None and (cfg.location_city or cfg.location_country):
        loc = {"type": "approximate"}
        for key, value in (("city", cfg.location_city), ("region", cfg.location_region),
                           ("country", cfg.location_country), ("timezone", cfg.timezone)):
            if value:
                loc[key] = value
        search["user_location"] = loc
    return [*TOOL_DEFINITIONS, search]


def _state_path(cfg: Config):
    import os
    from pathlib import Path

    return Path(os.environ.get("STATE_FILE") or cfg.project_dir / "state.json")


NEWS_COUNT_DEFAULT = 4  # 뉴스 브리핑에서 전할 소식 개수


def load_state(cfg: Config) -> dict[str, Any]:
    """다시 켜도 유지할 설정 (음악·목소리 음량, 마지막으로 알린 업데이트)."""
    try:
        return json.loads(_state_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(cfg: Config, **values: Any) -> None:
    state = load_state(cfg)
    state.update(values)
    path = _state_path(cfg)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def location_text(cfg: Config) -> str | None:
    parts = [x for x in (cfg.location_city, cfg.location_region, cfg.location_country) if x]
    return ", ".join(parts) if parts else None


class ToolError(Exception):
    """도구 실행 실패. 메시지는 Claude에게 그대로 전달된다."""


# ---------------------------------------------------------------- 시스템 음량


def get_system_volume() -> int | None:
    try:
        out = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(out)
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def set_system_volume(percent: int) -> None:
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {int(percent)}"],
        capture_output=True, timeout=5, check=True,
    )


# ---------------------------------------------------------------- 도구 실행기


class ToolBox:
    """도구 이름 → 실제 동작. music 은 speaker.music.MusicPlayer 인터페이스를 따른다."""

    def __init__(self, cfg: Config, music: Any, scheduler: Scheduler, memory: Any = None, pim: Any = None):
        self.cfg = cfg
        self.music = music
        self.scheduler = scheduler
        self.pim = pim  # speaker.pim.PersonalInfo (일정·메일). 없으면 해당 도구가 실패
        self.memory = memory  # speaker.memory.Memory (없으면 기억 도구가 실패)
        self.speaker = None  # speaker.tts.Speaker (음성 모드에서 main 이 넣어 준다. 목소리 음량용)
        self.listener = None  # speaker.audio_in.Listener (음성 모드에서 main 이 넣어 준다. 호출 문구용)
        # 이번 대화 턴에서 음악 상태를 건드렸는지. (호출어로 잠시 멈춘 음악을 다시 틀지 결정할 때 쓴다.)
        self.music_touched = False
        # 이번 턴에 음악을 틀거나 다시 재생했는지. (음악이 나오는 중에는 이어 듣기를 하지 않는다.)
        self.music_started = False

    # 매 턴 사용자 발화 앞에 붙는 상황 정보
    def context_line(self) -> str:
        now = time.localtime()
        weekday = "월화수목금토일"[now.tm_wday]
        parts = [time.strftime("%Y-%m-%d", now) + f" {weekday}요일 " + time.strftime("%H:%M", now)
                 + (f" ({self.cfg.timezone})" if self.cfg.timezone else "")]
        where = location_text(self.cfg)
        if where:
            parts.append(f"사는 곳: {where}")
        volumes = []
        volume = get_system_volume()
        if volume is not None:
            volumes.append(f"전체 {volume}%")
        try:
            volumes.append(f"음악 {self.music.get_volume()}%")
        except Exception:
            pass
        if self.speaker is not None:
            volumes.append(f"목소리 {self.speaker.get_voice_volume()}%")
            volumes.append(f"말 속도 {self.speaker.get_voice_speed():.2f}배")
        if volumes:
            parts.append("음량: " + " / ".join(volumes))
        try:
            playing = self.music.now_playing()
        except Exception:  # 음악 모듈 문제로 대화 자체가 막히면 안 된다
            playing = None
        parts.append(f"음악: {playing}" if playing else "음악: 없음")
        topics = self.news_topics()
        parts.append("뉴스 관심 분야: " + (", ".join(topics) if topics else "없음") + f" / 뉴스 개수: {self.news_count()}개")
        upcoming = self.scheduler.upcoming()
        if upcoming:
            now = time.time()
            parts.append("예약: " + ", ".join(e.describe(now) for e in upcoming[:5]))
        parts.append("답은 한국어로, 기온은 섭씨로")
        return "[상황: " + " | ".join(parts) + "]"

    def run(self, name: str, args: Any) -> str:
        """도구를 실행하고 결과 문자열을 돌려준다. 실패하면 ToolError."""
        args = self._validate(name, args)
        log.info("도구 실행: %s %s", name, args)
        try:
            return getattr(self, f"_tool_{name}")(**args)
        except ToolError:
            raise
        except Exception as exc:
            log.exception("도구 %s 실패", name)
            raise ToolError(f"{name} 실행 중 오류: {exc}") from exc

    @staticmethod
    def _validate(name: str, args: Any) -> dict[str, Any]:
        schema = next(
            (t["input_schema"] for t in TOOL_DEFINITIONS if t.get("name") == name and "input_schema" in t),
            None,
        )
        if schema is None:
            raise ToolError(f"알 수 없는 도구: {name}")
        if not isinstance(args, dict):
            raise ToolError("INVALID_JSON: 입력이 객체가 아닙니다.")
        clean: dict[str, Any] = {}
        for key, spec in schema["properties"].items():
            if key not in args or args[key] is None:
                if key in schema["required"]:
                    raise ToolError(f"INVALID_JSON: '{key}' 값이 없습니다.")
                continue
            value = args[key]
            if spec["type"] == "string":
                if not isinstance(value, str) or not value.strip():
                    raise ToolError(f"INVALID_JSON: '{key}'는 비어 있지 않은 문자열이어야 합니다.")
                if "enum" in spec and value not in spec["enum"]:
                    raise ToolError(f"INVALID_JSON: '{key}'는 {spec['enum']} 중 하나여야 합니다.")
                value = value.strip()
            elif spec["type"] == "boolean":
                if not isinstance(value, bool):
                    raise ToolError(f"INVALID_JSON: '{key}'는 true 나 false 여야 합니다.")
            elif spec["type"] == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ToolError(f"INVALID_JSON: '{key}'는 숫자여야 합니다.")
                value = float(value)
                if value < spec.get("minimum", value) or value > spec.get("maximum", value):
                    raise ToolError(f"INVALID_JSON: '{key}' 값이 허용 범위를 벗어났습니다.")
            elif spec["type"] == "integer":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                    raise ToolError(f"INVALID_JSON: '{key}'는 정수여야 합니다.")
                value = int(value)
                if value < spec.get("minimum", value) or value > spec.get("maximum", value):
                    raise ToolError(f"INVALID_JSON: '{key}' 값이 허용 범위를 벗어났습니다.")
            clean[key] = value
        return clean

    # --- 음악

    def _tool_play_music(self, query: str, kind: str) -> str:
        self.music_touched = True
        try:
            result = self.music.play(query, kind)
            self.music_started = True
            return result
        except Exception as exc:
            raise ToolError(str(exc) or "음악을 재생하지 못했습니다.") from exc

    def _tool_play_my_music(self, source: str, name: str | None = None, shuffle: bool = False) -> str:
        self.music_touched = True
        if source == "playlist" and not name:
            raise ToolError("재생목록 이름이 필요합니다.")
        try:
            result = self.music.play_library(source, name, shuffle)
            self.music_started = True
            return result
        except Exception as exc:
            raise ToolError(str(exc) or "음악을 재생하지 못했습니다.") from exc

    def _tool_play_podcast(self, query: str, episode: str = "latest") -> str:
        self.music_touched = True
        try:
            result = self.music.play_podcast(query, episode)
            self.music_started = True
            return result
        except Exception as exc:
            raise ToolError(str(exc) or "팟캐스트를 재생하지 못했습니다.") from exc

    def _tool_list_my_playlists(self) -> str:
        try:
            playlists = self.music.library_playlists()
        except Exception as exc:
            raise ToolError(str(exc) or "보관함을 불러오지 못했습니다.") from exc
        if not playlists:
            return "보관함에 재생목록이 없습니다."
        return "보관함 재생목록: " + ", ".join(
            f"{p.get('title')}({p.get('count')}곡)" if p.get("count") else str(p.get("title")) for p in playlists[:30]
        )

    def _tool_music_control(self, action: str) -> str:
        self.music_touched = True
        if action == "pause":
            self.music.pause()
            self.music_started = False
            return "일시정지했습니다."
        if action == "resume":
            if not self.music.now_playing():
                raise ToolError("이어서 재생할 음악이 없습니다.")
            self.music.resume()
            self.music_started = True
            return "다시 재생합니다."
        if action == "stop":
            self.music.stop()
            self.music_started = False
            return "음악을 껐습니다."
        if not self.music.now_playing():
            raise ToolError("재생 중인 음악이 없습니다.")
        title = self.music.next() if action == "next" else self.music.previous()
        # 호출어 때문에 멈춰 있었을 수 있으므로 곡을 넘긴 뒤에는 재생 상태로 둔다.
        self.music.resume()
        self.music_started = True
        return f"지금 곡: {title}" if title else "곡을 넘겼습니다."

    # --- 일정·메일

    def _tool_get_calendar(self, day: str = "today", date: str | None = None) -> str:
        if self.pim is None or not self.pim.calendar_enabled:
            raise ToolError("캘린더가 연결되어 있지 않습니다.")
        return self.pim.calendar_text(day, _parse_date(date))

    def _tool_get_unread_mail(self, limit: int = 10) -> str:
        if self.pim is None or not self.pim.mail_enabled:
            raise ToolError("메일이 연결되어 있지 않습니다.")
        return self.pim.mail_text(limit)

    # --- 기억

    def _tool_remember(self, fact: str) -> str:
        if self.memory is None:
            raise ToolError("기억 기능이 꺼져 있습니다.")
        return self.memory.remember(fact)

    def _tool_forget(self, about: str | None = None) -> str:
        if self.memory is None:
            raise ToolError("기억 기능이 꺼져 있습니다.")
        removed = self.memory.forget(about)
        if not removed:
            raise ToolError("지울 기억을 찾지 못했습니다.")
        return f"{len(removed)}개를 잊었습니다: " + "; ".join(removed[:5])

    # --- 음량

    def _tool_set_volume(self, percent: int, target: str = "system") -> str:
        if target == "music":
            self.music.set_volume(percent)
            save_state(self.cfg, music_volume=percent)
            return f"음악 음량을 {percent}%로 맞췄습니다."
        if target == "voice":
            if self.speaker is None:
                raise ToolError("목소리 음량은 음성 모드에서만 바꿀 수 있습니다.")
            self.speaker.set_voice_volume(percent)
            save_state(self.cfg, voice_volume=percent)
            return f"목소리 음량을 {percent}%로 맞췄습니다."
        set_system_volume(percent)
        return f"전체 음량을 {percent}%로 맞췄습니다."

    # --- 말 속도

    def _tool_set_voice_speed(self, speed: float) -> str:
        if self.speaker is None:
            raise ToolError("말 속도는 음성 모드에서만 바꿀 수 있습니다.")
        self.speaker.set_voice_speed(speed)
        save_state(self.cfg, voice_speed=round(speed, 2))
        return f"말 속도를 {speed:.2f}배로 바꿨습니다. 다음 문장부터 적용됩니다."

    # --- 화면

    def _tool_turn_off_display(self) -> str:
        r = subprocess.run(["pmset", "displaysleepnow"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise ToolError(f"화면을 끄지 못했습니다: {(r.stderr or r.stdout).strip()}")
        return "맥 화면을 껐습니다. 스피커는 계속 동작합니다."

    # --- 뉴스 관심 분야

    def news_topics(self) -> list[str]:
        return [str(t) for t in load_state(self.cfg).get("news_topics") or []]

    def news_count(self) -> int:
        try:
            return min(10, max(1, int(load_state(self.cfg).get("news_count") or NEWS_COUNT_DEFAULT)))
        except (TypeError, ValueError):
            return NEWS_COUNT_DEFAULT

    def _tool_set_news_count(self, count: int) -> str:
        count = int(count)
        if not 1 <= count <= 10:
            raise ToolError("뉴스 개수는 1개에서 10개 사이로 정해 주세요.")
        save_state(self.cfg, news_count=count)
        return f"뉴스 브리핑을 {count}개 소식으로 정했습니다."

    def _tool_set_news_topics(self, topics: list[str]) -> str:
        clean: list[str] = []
        for t in topics:
            t = " ".join(str(t).strip(" .,!?~'\"").split())
            if not t:
                continue
            if len(t) > 30:
                raise ToolError(f"'{t[:30]}…'은(는) 너무 깁니다. 분야는 짧은 검색어로 정해 주세요.")
            if t not in clean:
                clean.append(t)
        if len(clean) > 8:
            raise ToolError("뉴스 관심 분야는 여덟 개까지 정할 수 있습니다.")
        save_state(self.cfg, news_topics=clean)
        if not clean:
            return "뉴스 관심 분야를 지웠습니다. 이제 한국과 세계의 주요 뉴스를 전합니다."
        return f"뉴스 관심 분야: {', '.join(clean)}. 다음 뉴스 브리핑부터 이 분야 위주로 전합니다."

    # --- 호출 문구

    def _wake_phrases(self) -> list[str]:
        if self.listener is not None:
            return self.listener.phrases
        return list(load_state(self.cfg).get("wake_phrases") or ([self.cfg.wake_phrase] if self.cfg.wake_phrase else []))

    def _save_wake_phrases(self, phrases: list[str]) -> None:
        save_state(self.cfg, wake_phrases=phrases)
        if self.listener is not None:
            self.listener.set_phrases(phrases)

    def _tool_add_wake_phrase(self, phrase: str) -> str:
        phrase = " ".join(phrase.strip(" .,!?~'\"").split())
        letters = re.sub(r"[^\w]", "", phrase)
        if len(letters) < 3:
            raise ToolError("호출 문구는 세 글자 이상이어야 합니다. 짧으면 평소 말에도 자주 깨어납니다.")
        if len(letters) > 20:
            raise ToolError("호출 문구가 너무 깁니다. 스무 글자 안쪽으로 정해 주세요.")
        phrases = self._wake_phrases()
        if any(re.sub(r"[^\w]", "", p) == letters for p in phrases):
            return f"'{phrase}'은(는) 이미 등록되어 있습니다. 지금 호출 문구: {', '.join(phrases)}"
        if len(phrases) >= 5:
            raise ToolError(f"호출 문구는 다섯 개까지 등록할 수 있습니다. 지금: {', '.join(phrases)}")
        phrases.append(phrase)
        self._save_wake_phrases(phrases)
        return f"'{phrase}'을(를) 호출 문구로 등록했습니다. 지금 호출 문구: {', '.join(phrases)}"

    def _tool_remove_wake_phrase(self, phrase: str) -> str:
        phrases = self._wake_phrases()
        want = re.sub(r"[^\w]", "", phrase)
        best, best_score = None, 0.0
        for p in phrases:
            score = difflib.SequenceMatcher(None, want, re.sub(r"[^\w]", "", p)).ratio()
            if score > best_score:
                best, best_score = p, score
        if best is None or best_score < 0.6:
            raise ToolError(f"'{phrase}'에 맞는 호출 문구가 없습니다. 지금 호출 문구: {', '.join(phrases) or '없음'}")
        rest = [p for p in phrases if p != best]
        english = self.cfg.wake_word.lower() not in ("off", "none", "")
        if not rest and not english:
            raise ToolError("마지막 호출 문구라서 지우면 스피커를 부를 방법이 없어집니다. 다른 문구를 먼저 등록해 주세요.")
        self._save_wake_phrases(rest)
        left = ", ".join(rest) if rest else "없음 (헤이 자비스로만 부를 수 있음)"
        return f"'{best}'을(를) 호출 문구에서 지웠습니다. 남은 호출 문구: {left}"

    def _tool_list_wake_phrases(self) -> str:
        phrases = self._wake_phrases()
        english = self.cfg.wake_word.lower() not in ("off", "none", "")
        parts = phrases + (["헤이 자비스 (영어 호출어, 말로는 바꿀 수 없음)"] if english else [])
        return "호출 문구: " + (", ".join(parts) if parts else "없음")

    # --- 타이머

    def _tool_set_timer(self, seconds: int, label: str) -> str:
        e = self.scheduler.add(Event(kind="timer", fires_at=time.time() + seconds, label=label))
        return f"'{label}' 타이머를 {spoken_duration(seconds)} 뒤로 맞췄습니다. ({e.describe()})"

    def _tool_set_alarm(self, time: str, date: str | None = None, repeat: str = "none", label: str | None = None,
                        music: str | None = None, briefing: bool = False) -> str:
        hhmm = _parse_hhmm(time)
        date = _parse_date(date)
        try:
            fires_at = next_occurrence(hhmm, repeat, _now(), date)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        e = self.scheduler.add(Event(kind="alarm", fires_at=fires_at, label=label or "", repeat=repeat,
                                     hhmm=hhmm, music=music, briefing=briefing))
        extra = []
        if music:
            extra.append("음악으로 깨움")
        if briefing:
            extra.append("울린 뒤 브리핑")
        return f"알람을 맞췄습니다: {e.describe()}" + (f" ({', '.join(extra)})" if extra else "")

    def _tool_set_reminder(self, message: str, time: str | None = None, date: str | None = None,
                           in_minutes: int | None = None, repeat: str = "none") -> str:
        if in_minutes:
            if repeat != "none":
                raise ToolError("반복 리마인더는 시각(time)으로 정해야 합니다.")
            e = Event(kind="reminder", fires_at=_now() + in_minutes * 60, label="", message=message)
        elif time:
            hhmm = _parse_hhmm(time)
            try:
                fires_at = next_occurrence(hhmm, repeat, _now(), _parse_date(date))
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            e = Event(kind="reminder", fires_at=fires_at, label="", message=message, repeat=repeat, hhmm=hhmm)
        else:
            raise ToolError("언제 알려 줄지(time 또는 in_minutes)가 필요합니다.")
        self.scheduler.add(e)
        return f"리마인더를 맞췄습니다: {e.describe()}"

    def _tool_sleep_timer(self, minutes: int) -> str:
        if minutes == 0:
            cancelled = self.scheduler.cancel(kind="sleep")
            if not cancelled:
                raise ToolError("맞춰 둔 수면 타이머가 없습니다.")
            return "수면 타이머를 취소했습니다."
        self.scheduler.add(Event(kind="sleep", fires_at=_now() + minutes * 60, label="수면"))
        return f"{spoken_duration(minutes * 60)} 뒤에 음악을 천천히 줄여서 끕니다."

    def _tool_cancel_schedule(self, kind: str | None = None, label: str | None = None) -> str:
        cancelled = self.scheduler.cancel(kind, label)
        if not cancelled:
            raise ToolError("취소할 예약을 찾지 못했습니다.")
        return "취소했습니다: " + ", ".join(e.describe() for e in cancelled)

    def _tool_list_schedule(self, kind: str | None = None) -> str:
        upcoming = self.scheduler.upcoming(kind)
        if not upcoming:
            return "맞춰 둔 예약이 없습니다."
        now = _now()
        return "; ".join(e.describe(now) for e in upcoming)


def _now() -> float:
    return time.time()


def _parse_hhmm(text: str) -> str:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", text or "")
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ToolError("INVALID_JSON: time 은 'HH:MM' 24시간제여야 합니다. 예: '07:30', '21:00'")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _parse_date(text: str | None) -> str | None:
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text.strip()).isoformat()
    except ValueError as exc:
        raise ToolError("INVALID_JSON: date 는 'YYYY-MM-DD' 형식이어야 합니다.") from exc
