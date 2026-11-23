"""환경 변수(.env)에서 읽어오는 설정. 모든 모듈이 이 Config 하나만 받는다."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def _device(name: str) -> int | str | None:
    """sounddevice 장치 지정: 숫자면 인덱스, 아니면 이름 일부, 비어 있으면 시스템 기본값."""
    value = _env(name)
    if value is None:
        return None
    return int(value) if value.isdigit() else value


@dataclass(frozen=True)
class Config:
    project_dir: Path
    models_dir: Path

    # 두뇌 (Claude)
    anthropic_api_key: str | None
    claude_model: str
    claude_effort: str

    # 호출어 / 음성 인식
    wake_word: str  # openWakeWord 사전학습 모델 이름 (hey_jarvis, alexa, hey_mycroft ...)
    wake_threshold: float
    wake_threshold_music: float  # 음악이 나오는 동안의 호출어 기준 (낮을수록 잘 깨어나지만 잘못 깨어날 수 있음)
    wake_phrase: str | None  # 한국어 호출 문구 (음성 인식으로 확인). None 이면 끔
    whisper_model: str
    language: str
    input_device: int | str | None
    follow_up_seconds: float  # 답한 뒤 호출어 없이 이어서 듣는 시간. 0 이면 끔

    # 음성 합성
    openai_api_key: str | None
    tts_backend: str  # auto(=qwen) | qwen | openai | say
    qwen_tts_model: str
    qwen_tts_speaker: str
    qwen_tts_instruct: str | None  # 말투 지시문. 예: "차분하고 다정한 목소리로 말해 주세요."
    voice_speed: float  # 말 속도 배율 (음높이 유지). 1.0 = 모델 그대로
    openai_tts_model: str
    openai_tts_voice: str
    say_voice: str
    say_rate: int
    output_device: int | str | None

    # 음악 (mpv + yt-dlp)
    mpv_path: str | None
    ytdlp_path: str | None
    mpv_ao: str | None  # 테스트용: "null" 이면 소리 없이 재생
    music_volume: int
    duck_volume: int
    wake_music: str  # 호출됐을 때 음악을: duck(작게 줄임) | pause(멈춤)
    listen_duck_volume: int  # 듣는 동안의 음악 크기 (말할 때보다 더 작게)
    wake_duck_score: float  # 음악 중 호출어 점수가 이만큼만 올라도 음악을 잠깐 줄인다 (0 이면 끔)
    announce_tracks: bool  # 곡이 바뀔 때마다 제목을 말로 알려 줄지
    ytmusic_auth_path: Path  # YouTube Music 계정 연결 정보 (speaker.ytmusic_setup 으로 만든다)
    memory_path: Path | None  # 사용자 기억 파일. None 이면 기억 기능 끔
    schedule_path: Path  # 타이머·알람·리마인더 저장 파일
    location_city: str | None  # 사는 곳: 날씨·지역 검색의 기준
    location_region: str | None
    location_country: str | None  # 두 글자 국가 코드. 예: US, KR
    timezone: str | None  # 예: America/New_York (비우면 맥 설정에서 읽음)
    auto_reload: bool  # 코드가 바뀌면 한가할 때 스스로 다시 켤지
    remote_enabled: bool  # 휴대폰 브라우저 원격 조작 (같은 Wi-Fi, HTTPS + 비밀 토큰)
    remote_port: int
    calendar_enabled: bool  # 맥 캘린더 일정 읽기
    mail_enabled: bool  # 맥 메일 앱의 안 읽은 메일 읽기 (내용이 Claude 로 전송됨)


def _system_timezone() -> str | None:
    """맥의 시간대 이름 (/etc/localtime → .../zoneinfo/America/New_York)."""
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        return None
    return target.split("zoneinfo/", 1)[1] if "zoneinfo/" in target else None


def load_config() -> Config:
    load_dotenv(PROJECT_DIR / ".env")
    venv_ytdlp = PROJECT_DIR / ".venv" / "bin" / "yt-dlp"
    return Config(
        project_dir=PROJECT_DIR,
        models_dir=PROJECT_DIR / "models",
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        claude_model=_env("SPEAKER_MODEL", "claude-haiku-5-5"),
        claude_effort=_env("SPEAKER_EFFORT", "low"),
        wake_word=_env("WAKE_WORD", "hey_jarvis"),
        wake_threshold=float(_env("WAKE_THRESHOLD", "0.5")),
        wake_threshold_music=float(_env("WAKE_THRESHOLD_MUSIC", "0.25")),
        wake_phrase=None if _env("WAKE_PHRASE", "야 듣고 있어").lower() in ("off", "none") else _env("WAKE_PHRASE", "야 듣고 있어"),
        whisper_model=_env("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"),
        language=_env("LANGUAGE_CODE", "ko"),
        input_device=_device("INPUT_DEVICE"),
        follow_up_seconds=float(_env("FOLLOW_UP_SECONDS", "5")),
        openai_api_key=_env("OPENAI_API_KEY"),
        tts_backend=_env("TTS_BACKEND", "auto"),
        qwen_tts_model=_env("QWEN_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"),
        qwen_tts_speaker=_env("QWEN_TTS_SPEAKER", "ono_anna"),
        qwen_tts_instruct=_env("QWEN_TTS_INSTRUCT", "A clear, neutral voice."),
        voice_speed=float(_env("VOICE_SPEED", "1.2")),
        openai_tts_model=_env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        openai_tts_voice=_env("OPENAI_TTS_VOICE", "coral"),
        say_voice=_env("SAY_VOICE", "Yuna"),
        say_rate=int(_env("SAY_RATE", "185")),
        output_device=_device("OUTPUT_DEVICE"),
        mpv_path=_env("MPV_PATH") or shutil.which("mpv") or "/opt/homebrew/bin/mpv",
        ytdlp_path=_env("YTDLP_PATH") or (str(venv_ytdlp) if venv_ytdlp.exists() else shutil.which("yt-dlp")),
        mpv_ao=_env("MPV_AO"),
        music_volume=int(_env("MUSIC_VOLUME", "70")),
        duck_volume=int(_env("DUCK_VOLUME", "20")),
        wake_music=(_env("WAKE_MUSIC", "duck") or "duck").lower(),
        listen_duck_volume=int(_env("LISTEN_DUCK_VOLUME", "6")),
        wake_duck_score=float(_env("WAKE_DUCK_SCORE", "0.05")),
        announce_tracks=_env("ANNOUNCE_TRACKS", "0").lower() in ("1", "true", "yes", "on"),
        ytmusic_auth_path=Path(_env("YTMUSIC_AUTH", str(PROJECT_DIR / "ytmusic_auth.json"))).expanduser(),
        schedule_path=Path(_env("SCHEDULE_FILE", str(PROJECT_DIR / "schedule.json"))).expanduser(),
        location_city=_env("LOCATION_CITY"),
        location_region=_env("LOCATION_REGION"),
        location_country=_env("LOCATION_COUNTRY"),
        timezone=_env("TIMEZONE") or _system_timezone(),
        remote_enabled=_env("REMOTE", "on").lower() not in ("off", "0", "false", "no"),
        remote_port=int(_env("REMOTE_PORT", "8443")),
        auto_reload=_env("AUTO_RELOAD", "on").lower() not in ("off", "0", "false", "no"),
        calendar_enabled=_env("CALENDAR", "on").lower() not in ("off", "0", "false", "no"),
        mail_enabled=_env("MAIL", "on").lower() not in ("off", "0", "false", "no"),
        memory_path=None if _env("MEMORY", "on").lower() in ("off", "0", "false", "no")
        else Path(_env("MEMORY_FILE", str(PROJECT_DIR / "memory.json"))).expanduser(),
    )
