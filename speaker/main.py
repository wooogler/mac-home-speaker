"""전체 흐름: 호출어 → 녹음 → 음성 인식 → Claude → 음성 출력."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import queue
import signal
import sys
import threading
import time
import warnings

from .config import PROVIDER_KEYS, Config, load_config
from .schedule import Event, Scheduler, spoken_clock
from .tools import ToolBox, load_state, save_state

log = logging.getLogger("speaker.main")


def setup_logging(cfg: Config, verbose: bool) -> None:
    # 모델을 불러올 때마다 찍히는, 해가 없는 경고와 진행 막대를 끈다.
    warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    log_dir = cfg.project_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = logging.handlers.RotatingFileHandler(
        log_dir / "speaker.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    logfile.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers[:] = [console, logfile]
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_pim(cfg: Config):
    if not (cfg.calendar_enabled or cfg.mail_enabled):
        return None
    from .pim import PersonalInfo

    return PersonalInfo(calendar=cfg.calendar_enabled, mail=cfg.mail_enabled)


def _load_memory(cfg: Config):
    if cfg.memory_path is None:
        return None
    from .memory import Memory

    return Memory(cfg.memory_path)


FIRST_WORD_AFTER_S = 1.5  # 이만큼 지나도 첫 문장이 없으면 "네, 잠시만요"를 먼저 말한다
RESTART_EXIT_CODE = 75  # start.command 에게 '다시 켜 줘'라는 뜻
NEAR_DUCK_S = 3.0  # 호출어 비슷한 소리 뒤 음악을 줄여 두는 시간
SHUTDOWN_TIMEOUT_S = 15  # 종료 정리가 이보다 오래 걸리면 강제로 끝낸다
WATCHED_SUFFIXES = (".py",)


def _running_speaker(pid_file) -> int | None:
    """pid 파일의 스피커가 아직 살아 있으면 그 pid."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)  # 신호는 보내지 않고 살아 있는지만 확인
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    try:
        import subprocess

        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        cmd = ""
    return pid if "-m speaker" in cmd else None


def _git(cfg: Config, *args: str, timeout: float = 10) -> str | None:
    import subprocess

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # 로그인 창을 띄우며 멈추지 않게
    try:
        r = subprocess.run(["git", "-C", str(cfg.project_dir), *args], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _pull_update(cfg: Config) -> bool:
    """GitHub(원격 저장소)에 새 버전이 있으면 받아 온다. 받았으면 True.

    받아 온 뒤 다시 켜는 건 _check_code_change 가 한가할 때 한다. 패키지가 바뀌었으면 다시 켤 때
    start.command 의 `uv run` 이 알아서 맞춘다. 사용자가 코드를 고쳐 뒀거나 따로 커밋해 둔 경우에는
    덮어쓰지 않고 건너뛴다.
    """
    upstream = _git(cfg, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not upstream:
        return False
    if _git(cfg, "fetch", "--quiet", timeout=90) is None:
        log.info("새 버전을 확인하지 못했습니다 (인터넷 연결 확인).")
        return False
    head, remote = _git(cfg, "rev-parse", "HEAD"), _git(cfg, "rev-parse", "@{u}")
    if not head or not remote or head == remote:
        return False
    if _git(cfg, "merge-base", "--is-ancestor", "HEAD", "@{u}") is None:
        log.info("이 맥에만 있는 커밋이 있어 새 버전을 자동으로 받지 않습니다 (%s).", upstream)
        return False
    if _git(cfg, "status", "--porcelain", "--untracked-files=no"):
        log.info("고친 코드가 있어 새 버전을 자동으로 받지 않습니다. git status 로 확인해 주세요.")
        return False
    if _git(cfg, "merge", "--ff-only", "--quiet", "@{u}", timeout=60) is None:
        log.warning("새 버전을 받다가 실패했습니다.")
        return False
    log.info("새 버전을 받았습니다: %s → %s", head[:7], remote[:7])
    return True


def _code_stamp(cfg: Config) -> tuple:
    """다시 켤지 판단하는 기준: git 저장소면 현재 커밋(새 커밋이 '업데이트'), 아니면 코드 파일 수정 시각.
    .env 는 어느 쪽이든 수정 시각으로 본다."""
    env = cfg.project_dir / ".env"
    env_mtime = env.stat().st_mtime if env.exists() else 0.0
    head = _git(cfg, "rev-parse", "HEAD")
    if head:
        return (head, env_mtime)
    paths = cfg.project_dir.joinpath("speaker").glob("*.py")
    return (max((p.stat().st_mtime for p in paths), default=0.0), env_mtime)


def _as_sentence(text: str) -> str:
    text = text.strip()
    return text if text[-1:] in ".!?요" else text + "."


def _update_summary(cfg: Config, since: str, head: str) -> list[str]:
    """since 이후 커밋들의 제목 (오래된 것부터). 커밋 제목은 스피커가 그대로 읽는 한 문장으로 쓴다."""
    log_text = _git(cfg, "log", "--reverse", "--format=%s", f"{since}..{head}")
    if log_text is None:  # 기록에 없는 커밋 (기록을 다시 쓴 경우 등): 최신 커밋만
        log_text = _git(cfg, "log", "-1", "--format=%s", head) or ""
    return [line.strip() for line in log_text.splitlines() if line.strip()]


ALARM_BRIEFING_REQUEST = (
    "[스피커가 자동으로 보낸 요청: 아침 알람 브리핑] 방금 알람이 울렸어요. 웹 검색으로 사는 곳의 오늘 날씨와 "
    "뉴스(상황 줄의 뉴스 개수만큼, 관심 분야가 있으면 그 분야에서)를 찾고, get_calendar 로 오늘 일정과 get_unread_mail 로 중요해 보이는 새 메일을 확인해서, "
    "오늘 예정된 리마인더까지 묶어 밝게 브리핑해 주세요. 연결이 안 된 항목은 말하지 말고 건너뛰세요. "
    "시각은 방금 말했으니 다시 말하지 마세요."
)


class Assistant:
    """음성 모드의 상태 기계. 한 번에 한 가지 일만 한다: 대기 → 듣기 → 생각/말하기."""

    def __init__(self, cfg: Config):
        from .audio_in import Listener, Transcriber
        from .brain import Brain
        from .music import MusicPlayer
        from .tts import Speaker

        self.cfg = cfg
        self.speaker = Speaker(cfg)
        self.music = MusicPlayer(cfg)
        self.music.on_error = self.speaker.say  # 검색 뒤 곡 불러오기에 실패하면 말로 알린다
        self.events: queue.Queue[Event] = queue.Queue()  # 때가 된 타이머·알람·리마인더
        self.scheduler = Scheduler(cfg.schedule_path, self.events.put)
        self.pim = _load_pim(cfg)
        self.toolbox = ToolBox(cfg, self.music, self.scheduler, _load_memory(cfg), self.pim)
        self.toolbox.speaker = self.speaker
        state = load_state(cfg)
        if "music_volume" in state:
            self.music.set_volume(int(state["music_volume"]))
        if "voice_volume" in state:
            self.speaker.set_voice_volume(int(state["voice_volume"]))
        if "voice_speed" in state:
            self.speaker.set_voice_speed(float(state["voice_speed"]))
        self.brain = Brain(cfg, self.toolbox)
        self.listener = Listener(cfg)
        self.transcriber = Transcriber(cfg)

        self.stopping = threading.Event()
        self.restart_requested = False
        # 코드가 바뀌면 (원격 개발 중) 한가할 때 스스로 다시 켠다
        self._code_stamp = _code_stamp(cfg)
        self._next_code_check = 0.0
        self._next_update_check = time.monotonic() + 120  # 켜고 2분 뒤 처음 확인
        self._update_thread: threading.Thread | None = None
        self._brain_thread: threading.Thread | None = None
        # 호출어 때문에 잠시 멈춘 음악을 대화가 끝난 뒤 다시 틀어야 하는지
        self._resume_music = False
        # 호출어 때문에 음악 소리를 줄여 둔 상태인지 (WAKE_MUSIC=duck)
        self._music_ducked = False
        self._near_duck_until = 0.0  # 호출어 비슷한 소리로 음악을 잠깐 줄여 둔 동안 (0 이면 아님)
        self._filler_said = False  # 이번 턴에 "잠시만요" 같은 메움 말을 이미 했는지
        # 휴대폰에서 온 요청: (글, 소리 내기, 결과를 담을 리스트, 끝났음을 알릴 Event)
        self._remote_q: queue.Queue = queue.Queue()
        self.remote = None
        # 곡이 바뀐 것을 알아채기 위한 상태
        self._last_track: str | None = None
        self._next_track_check = 0.0

    # ------------------------------------------------------------------ 실행

    def run(self) -> None:
        log.info("음성 인식·음성 합성 모델을 불러오는 중입니다 (처음에는 내려받느라 오래 걸릴 수 있어요)...")
        self.transcriber.warmup()
        self.speaker.warmup()
        # 호출 문구: 말로 등록·삭제한 목록(state.json)이 있으면 그것, 없으면 WAKE_PHRASE
        saved = load_state(self.cfg).get("wake_phrases")
        phrases = list(saved) if isinstance(saved, list) else ([self.cfg.wake_phrase] if self.cfg.wake_phrase else [])
        # 스피커가 말하는 동안에는 자기 목소리를 받아쓰지 않도록 문구를 찾지 않는다
        self.listener.enable_phrase(phrases, self.transcriber, gate=lambda: not self.speaker.is_speaking())
        self.toolbox.listener = self.listener
        self.listener.start()
        if self.cfg.remote_enabled:
            try:
                from .remote import RemoteServer

                self.remote = RemoteServer(self, port=self.cfg.remote_port)
                self.remote.start()
                log.info("휴대폰 원격 조작 주소 (같은 Wi-Fi 에서): %s", self.remote.url)
            except Exception:
                log.exception("휴대폰 원격 조작 서버를 켜지 못했습니다 (스피커는 계속 동작)")
        names = [f"'{x}'" for x in phrases] + ([f"'{self.cfg.wake_word}'"] if self.cfg.wake_word.lower() != "off" else [])
        log.info("준비 완료. 호출어 %s 를 기다립니다. (TTS: %s)", " / ".join(names), self.speaker.backend)
        self.speaker.say("준비됐어요.")
        self._announce_update()
        if self.scheduler.missed:
            missed = ", ".join(e.message for e in self.scheduler.missed)
            self.speaker.say(f"꺼져 있는 동안 지나간 리마인더가 있어요. {missed}")
            self.scheduler.missed = []

        follow_up = False
        while not self.stopping.is_set():
            if not follow_up and not self.listener.wake_event.wait(0.2):
                self._handle_events()
                self._handle_remote()
                self._watch_track()
                self._watch_near_wake()
                self._check_remote_update()
                self._check_code_change()
                continue
            try:
                follow_up = self._handle_turn(follow_up)
            except Exception:
                log.exception("턴 처리 중 오류")
                self.speaker.chime("error")
                follow_up = False
                self._finish_conversation()

    def shutdown(self) -> None:
        self.stopping.set()
        if self.remote is not None:
            try:
                self.remote.stop()
            except Exception:
                log.exception("원격 서버 종료 중 오류")
        for closer in (self.listener.stop, self.speaker.shutdown, self.music.shutdown, self.scheduler.shutdown,
                       (self.pim.shutdown if self.pim else (lambda: None)),
                       self.brain.close):
            try:
                closer()
            except Exception:
                log.exception("종료 중 오류")

    # ------------------------------------------------------------------ 한 턴

    def _handle_turn(self, follow_up: bool) -> bool:
        """한 번 듣고 답한다. 답한 뒤 호출어 없이 곧바로 다시 들어야 하면 True."""
        self.speaker.stop()
        near_ducked, self._near_duck_until = self._near_duck_until > 0, 0.0  # 이제 대화가 음악 크기를 맡는다
        if near_ducked and not (self.music.is_playing() or self.music.is_starting()):
            self.music.unduck()
        if self.music.is_playing() or self.music.is_starting():
            if self.cfg.wake_music == "pause":
                self.music.pause()
                self._resume_music = True
            else:
                # 음악은 계속 틀되, 말소리가 잘 들리도록 듣는 동안은 아주 작게 줄인다
                self.music.duck(self.cfg.listen_duck_volume)
                self._music_ducked = True
        command = None if follow_up else self.listener.take_phrase_command()
        if command:
            # "야 듣고 있어? 음악 틀어줘"처럼 한 번에 말함: 다시 녹음하지 않고 바로 처리
            self.listener.clear_wake()
            self.speaker.chime("done")
            text = command
            log.info("호출 문구와 함께 들은 말: %r", text)
        else:
            if follow_up:
                time.sleep(0.25)  # 방금 한 말의 끝자락이 녹음에 섞이지 않게
            else:
                self.speaker.chime("wake")
            self.listener.clear_wake()

            # 이어 듣기에서는 설정한 시간만 기다린다. (질문으로 끝난 답 뒤에는 최소 5초)
            timeout = max(5.0, self.cfg.follow_up_seconds) if follow_up else 6.0
            audio = self.listener.record_utterance(start_timeout=timeout)
            if audio is None:
                log.info("말이 들리지 않아 대기로 돌아갑니다.")
                self._finish_conversation()
                return False
            self.speaker.chime("done")

            started = time.monotonic()
            text = self.transcriber.transcribe(audio)
            log.info("들은 말 (%.1f초 분량, 인식 %.1f초): %r", len(audio) / 16000, time.monotonic() - started, text)
        if not text:
            self._finish_conversation()
            return False

        reply = self._converse(text, follow_up)
        if reply is None:  # 사용자가 호출어로 끼어듦: wake_event 가 켜진 채로 돌아가 곧바로 다시 듣는다
            return False
        if self.toolbox.music_touched:
            self._resume_music = False  # 음악 명령이 있었으면 그 결과를 따른다
        log.info("답: %s", reply or "(스피커에게 한 말이 아니라고 보고 답하지 않음)")
        if reply and self._keep_listening(reply):
            return True
        self._finish_conversation()
        return False

    def _keep_listening(self, reply: str) -> bool:
        """답한 뒤 호출어 없이 이어서 들을지."""
        if reply.rstrip().endswith("?"):
            return True  # 스피커가 되물었으면 언제나 대답을 듣는다
        if self.cfg.follow_up_seconds <= 0:
            return False
        # 이번 턴에 음악을 틀었으면 듣지 않는다: 음악 소리를 말로 잘못 알아듣게 된다.
        return not self.toolbox.music_started

    def _converse(self, text: str, follow_up: bool = False, speak: bool = True) -> str | None:
        """Claude 에게 묻고 답을 말한다. 도중에 호출어가 들리면 멈추고 None.

        speak=False 면 소리 내지 않고 답 글만 돌려준다 (휴대폰에서 조용히 보낸 요청).
        """
        # 이전에 끼어들기로 중단된 요청이 아직 정리 중이면 끝나기를 기다린다 (대화 기록을 같이 쓰므로).
        if self._brain_thread and self._brain_thread.is_alive():
            self._brain_thread.join()

        self.toolbox.music_touched = False
        self.toolbox.music_started = False
        cancel = threading.Event()
        result: list[str] = []
        first_said = threading.Event()  # 첫 문장(또는 잠깐만요)이 나갔는지
        self._filler_said = not speak

        def on_sentence(sentence: str) -> None:
            first_said.set()
            if speak:
                self.speaker.say(sentence)

        def on_tool(name: str) -> None:
            if speak:
                self._announce_tool(name)

        def work() -> None:
            result.append(
                self.brain.respond(
                    text,
                    on_sentence=on_sentence,
                    on_tool_start=on_tool,
                    cancel=cancel,
                    follow_up=follow_up,
                )
            )

        thread = threading.Thread(target=work, name="brain", daemon=True)
        self._brain_thread = thread
        started = time.monotonic()
        thread.start()

        # 답하는 도중에 음악이 시작될 수 있으므로(재생 명령) 말하는 동안은 음악 소리를 낮춰 둔다.
        if speak:
            self.music.duck()
        try:
            while thread.is_alive() or self.speaker.is_speaking():
                # 빠른 첫마디: 답이 늦으면 (도구를 여러 번 쓰는 요청 등) 먼저 "네, 잠시만요"
                if (speak and not first_said.is_set() and not self._filler_said
                        and time.monotonic() - started > FIRST_WORD_AFTER_S):
                    self._filler_said = True
                    self.speaker.say("네, 잠시만요.")
                if self.stopping.is_set() or (speak and self.listener.wake_event.wait(0.05)):
                    cancel.set()
                    self.speaker.stop()
                    if self.toolbox.music_touched:
                        self._resume_music = False
                    return None
                if not speak:
                    time.sleep(0.05)
            return result[0] if result else ""
        finally:
            if not speak:
                pass
            elif self._music_ducked:
                self.music.duck(self.cfg.listen_duck_volume)  # 대화가 이어질 수 있으니 다시 듣는 크기로
            else:
                self.music.unduck()

    def _announce_tool(self, name: str) -> None:
        """시간이 걸리는 도구가 시작될 때 한마디 해서 침묵을 메운다."""
        # 웹 검색은 서버에서 web_search 나 code_execution(검색 결과 걸러내기) 이름으로 시작된다.
        filler = {
            "web_search": "찾아볼게요.",
            "web_fetch": "찾아볼게요.",
            "code_execution": "찾아볼게요.",
            "play_music": "네, 잠시만요.",
        }.get(name)
        if filler and not self._filler_said:
            self._filler_said = True
            self.speaker.say(filler)

    def _finish_conversation(self) -> None:
        """대화가 끝났을 때: 호출어 때문에 멈췄던 음악을 되살린다.

        대화 중에 음악 명령이 있었다면 _resume_music 이 이미 꺼져 있어 그 결과가 그대로 남는다.
        """
        if self._resume_music:
            try:
                self.music.resume()
            except Exception:
                log.exception("음악을 다시 재생하지 못했습니다.")
        if self._music_ducked:
            self.music.unduck()  # 줄여 뒀던 음악을 원래 크기로 (꺼졌거나 바뀌었어도 무해)
        self._resume_music = False
        self._music_ducked = False

    # ------------------------------------------------------------------ 휴대폰 원격 조작

    def remote_text(self, text: str, speak: bool) -> str:
        """(웹 서버 스레드) 휴대폰에서 보낸 글. 메인 루프가 차례대로 처리할 때까지 기다린다."""
        text = (text or "").strip()
        if not text:
            return ""
        box: list[str] = []
        done = threading.Event()
        self._remote_q.put((text, speak, box, done))
        if not done.wait(120):
            return "스피커가 바빠서 처리하지 못했어요. 잠시 후 다시 보내 주세요."
        return box[0] if box else ""

    def remote_audio(self, pcm16k, speak: bool) -> tuple[str, str]:
        """(웹 서버 스레드) 휴대폰 마이크로 녹음한 말. 받아쓴 뒤 글과 똑같이 처리한다."""
        heard = self.transcriber.transcribe(pcm16k)
        log.info("휴대폰에서 들은 말 (%.1f초): %r", len(pcm16k) / 16000, heard)
        if not heard:
            return "", "말소리를 알아듣지 못했어요. 조금 더 크게 말해 주세요."
        return heard, self.remote_text(heard, speak)

    def remote_status(self) -> dict:
        try:
            playing = self.music.now_playing()
        except Exception:
            playing = None
        now = time.time()
        try:
            paused = bool(playing) and not self.music.is_playing() and not self.music.is_starting()
        except Exception:
            paused = False
        return {
            "now_playing": playing,  # 리모컨 화면이 읽는 키
            "paused": paused,
            "art": self._safe(self.music.now_playing_art),
            "음악": playing or "없음",
            "음악 음량": f"{self.music.get_volume()}%",
            "목소리 음량": f"{self.speaker.get_voice_volume()}%",
            "예약": [e.describe(now) for e in self.scheduler.upcoming()[:5]] or ["없음"],
            "호출어": [*self.listener.phrases, *(["헤이 자비스"] if self.cfg.wake_word.lower() != "off" else [])],
        }

    # --- 휴대폰에서 Claude 를 거치지 않고 바로 하는 조작 (웹 서버 스레드에서 불림)

    def _tool(self, name: str, args: dict) -> dict:
        from .tools import ToolError

        try:
            return {"ok": True, "message": self.toolbox.run(name, args)}
        except ToolError as exc:
            return {"ok": False, "message": str(exc).replace("INVALID_JSON: ", "")}

    def remote_schedule(self) -> list[dict]:
        import datetime as dt
        from .schedule import KIND_NAMES, REPEAT_NAMES

        now = time.time()
        return [{
            "id": e.id, "kind": e.kind, "kind_name": KIND_NAMES.get(e.kind, e.kind),
            "label": e.label, "message": e.message, "when_text": e.when_text(now),
            "at": dt.datetime.fromtimestamp(e.fires_at).isoformat(timespec="minutes"),
            "repeat": e.repeat, "repeat_name": REPEAT_NAMES.get(e.repeat, ""),
            "music": e.music, "briefing": e.briefing, "description": e.describe(now),
            "enabled": e.enabled, "time": dt.datetime.fromtimestamp(e.fires_at).strftime("%H:%M"),
            "date": dt.datetime.fromtimestamp(e.fires_at).strftime("%Y-%m-%d"),
        } for e in self.scheduler.upcoming()]

    def remote_schedule_add(self, data: dict) -> dict:
        kind = data.get("kind")
        pick = lambda *keys: {k: data[k] for k in keys if data.get(k) not in (None, "")}  # noqa: E731
        if kind == "alarm":
            return self._tool("set_alarm", pick("time", "date", "repeat", "label", "music", "briefing"))
        if kind == "reminder":
            return self._tool("set_reminder", pick("message", "time", "date", "in_minutes", "repeat"))
        if kind == "timer":
            minutes = data.get("minutes")
            if not isinstance(minutes, (int, float)) or minutes <= 0:
                return {"ok": False, "message": "몇 분 뒤인지 알려 주세요."}
            return self._tool("set_timer", {"seconds": int(minutes * 60), "label": data.get("label") or "타이머"})
        if kind == "sleep":
            return self._tool("sleep_timer", {"minutes": int(data.get("minutes") or 0)})
        return {"ok": False, "message": "알 수 없는 종류예요."}

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    def remote_schedule_update(self, event_id: str, data: dict) -> dict:
        """예약 고치기. data 에 enabled 만 있으면 켜기/끄기, 아니면 add 와 같은 내용으로 새로 맞춘 뒤 옛것을 지운다."""
        old = self.scheduler.get(event_id)
        if old is None:
            return {"ok": False, "message": "이미 없는 예약이에요."}
        if set(data) <= {"enabled", "id"}:
            e = self.scheduler.set_enabled(event_id, bool(data.get("enabled")))
            return {"ok": True, "message": f"{'켰어요' if e.enabled else '껐어요'}: {e.describe()}"}
        data = {"kind": old.kind, **data}
        before = {e.id for e in self.scheduler.upcoming()}
        result = self.remote_schedule_add(data)
        if not result.get("ok"):
            return result  # 새로 맞추지 못했으면 옛 예약은 그대로 둔다
        self.scheduler.remove(event_id)
        new_ids = [e.id for e in self.scheduler.upcoming() if e.id not in before]
        if new_ids and data.get("enabled") is False:
            self.scheduler.set_enabled(new_ids[0], False)
        result["message"] = result["message"].replace("맞췄습니다", "바꿨습니다")
        return result

    def remote_schedule_delete(self, event_id: str) -> dict:
        e = self.scheduler.remove(event_id)
        return {"ok": bool(e), "message": f"지웠어요: {e.describe()}" if e else "이미 없는 예약이에요."}

    def remote_settings(self) -> dict:
        from .tools import get_system_volume

        return {
            "system_volume": get_system_volume(),
            "music_volume": self.music.get_volume(),
            "voice_volume": self.speaker.get_voice_volume(),
            "voice_speed": round(self.speaker.get_voice_speed(), 2),
            "wake_phrases": self.listener.phrases,
            "news_topics": self.toolbox.news_topics(),
            "news_count": self.toolbox.news_count(),
            "english_wake": "헤이 자비스" if self.cfg.wake_word.lower() != "off" else None,
            "follow_up_seconds": self.cfg.follow_up_seconds,
        }

    def remote_settings_set(self, data: dict) -> dict:
        """바꿀 항목만 담아 보낸다. 여러 개면 차례로 적용하고 결과 문장을 모아 돌려준다."""
        steps = []
        for key, target in (("system_volume", "system"), ("music_volume", "music"), ("voice_volume", "voice")):
            if key in data:
                steps.append(self._tool("set_volume", {"percent": int(data[key]), "target": target}))
        if "voice_speed" in data:
            steps.append(self._tool("set_voice_speed", {"speed": float(data["voice_speed"])}))
        if data.get("wake_phrase_add"):
            steps.append(self._tool("add_wake_phrase", {"phrase": str(data["wake_phrase_add"])}))
        if data.get("wake_phrase_remove"):
            steps.append(self._tool("remove_wake_phrase", {"phrase": str(data["wake_phrase_remove"])}))
        if "news_count" in data:
            steps.append(self._tool("set_news_count", {"count": int(data["news_count"])}))
        if data.get("news_topic_add") or data.get("news_topic_remove"):
            topics = self.toolbox.news_topics()
            add, remove = str(data.get("news_topic_add") or "").strip(), str(data.get("news_topic_remove") or "")
            topics = [t for t in topics if t != remove] + ([add] if add and add not in topics else [])
            steps.append(self._tool("set_news_topics", {"topics": topics}))
        if data.get("display_off"):
            steps.append(self._tool("turn_off_display", {}))
        if not steps:
            return {"ok": False, "message": "바꿀 항목이 없어요.", "settings": self.remote_settings()}
        return {"ok": all(x["ok"] for x in steps), "message": " ".join(x["message"] for x in steps),
                "settings": self.remote_settings()}

    def remote_music(self, action: str, query: str | None = None, kind: str = "song") -> dict:
        """pause|resume|stop|next|previous|play(query, kind)|liked|mix"""
        if action == "play":
            if not query:
                return {"ok": False, "message": "무엇을 틀지 적어 주세요."}
            r = self._tool("play_music", {"query": query, "kind": kind if kind in ("song", "artist", "playlist") else "song"})
        elif action in ("liked", "mix", "recent"):
            r = self._tool("play_my_music", {"source": action, "shuffle": action == "liked"})
        elif action in ("pause", "resume", "stop", "next", "previous"):
            r = self._tool("music_control", {"action": action})
        else:
            return {"ok": False, "message": "알 수 없는 동작이에요."}
        try:
            r["now_playing"] = self.music.now_playing()
        except Exception:
            r["now_playing"] = None
        return r

    def _handle_remote(self) -> None:
        try:
            text, speak, box, done = self._remote_q.get_nowait()
        except queue.Empty:
            return
        log.info("휴대폰에서 온 요청%s: %r", "" if speak else " (조용히)", text)
        try:
            reply = self._converse(text, speak=speak)
            log.info("답: %s", reply or "(없음)")
            box.append(reply or "")
            if speak:
                self._finish_conversation()
        except Exception:
            log.exception("휴대폰 요청 처리 중 오류")
            box.append("처리하다가 오류가 났어요.")
        finally:
            done.set()

    def _announce_update(self) -> None:
        """지난번 켰을 때 이후로 새 커밋이 있으면 무엇이 바뀌었는지 말해 준다."""
        head = _git(self.cfg, "rev-parse", "HEAD")
        if not head:
            return
        state = load_state(self.cfg)
        last = state.get("announced_commit")
        if last and last != head:
            changes = _update_summary(self.cfg, last, head)
            if changes:
                spoken = changes[:3]
                more = f" 그 밖에 {len(changes) - 3}가지가 더 바뀌었어요." if len(changes) > 3 else ""
                log.info("업데이트 브리핑: %s", " / ".join(changes))
                self.speaker.say("업데이트가 있었어요. " + " ".join(_as_sentence(c) for c in spoken) + more)
        save_state(self.cfg, announced_commit=head)

    def _check_remote_update(self) -> None:
        """AUTO_UPDATE=on 이면 가끔 GitHub 에 새 버전이 있는지 보고 받아 온다 (따로 도는 스레드에서)."""
        if not (self.cfg.auto_update and self.cfg.auto_reload):
            return
        now = time.monotonic()
        if now < self._next_update_check or (self._update_thread and self._update_thread.is_alive()):
            return
        self._next_update_check = now + self.cfg.auto_update_hours * 3600
        self._update_thread = threading.Thread(target=_pull_update, args=(self.cfg,), name="auto-update", daemon=True)
        self._update_thread.start()

    def _check_code_change(self) -> None:
        """대기 중에 새 커밋(또는 .env 변경)을 알아채면, 음악·말이 없을 때 다시 시작한다 (AUTO_RELOAD=off 로 끔)."""
        if not self.cfg.auto_reload:
            return
        now = time.monotonic()
        if now < self._next_code_check:
            return
        self._next_code_check = now + 3.0
        stamp = _code_stamp(self.cfg)
        if stamp == self._code_stamp:
            return
        if self.speaker.is_speaking() or self.music.is_playing() or self.music.is_starting():
            return  # 듣는 중인 건 끊지 않는다. 끝나면 다시 확인
        if isinstance(stamp[0], float) and time.time() - max(stamp) < 2.0:
            return  # (git 이 아닐 때) 파일을 여러 개 고치는 중일 수 있으니 잠깐 기다린다
        log.info("업데이트가 있어 다시 시작합니다.")
        self.speaker.say("업데이트가 있어서 다시 시작할게요.")
        self.speaker.wait()
        self.restart_requested = True
        self.stopping.set()

    def _watch_near_wake(self) -> None:
        """음악 중 호출어 점수가 조금 오르면(부르는 중일 수 있음) 음악을 잠깐 줄여 다음 말이 잘 들리게 한다."""
        now = time.monotonic()
        recent = now - self.listener.near_wake_at < NEAR_DUCK_S
        if recent and self._near_duck_until == 0.0:
            if not (self.listener.music_playing and self.music.is_playing()):
                return
            log.info("호출어 비슷한 소리가 들려 음악을 잠깐 줄입니다.")
            self.music.duck(self.cfg.duck_volume)
        if recent:
            self._near_duck_until = self.listener.near_wake_at + NEAR_DUCK_S
        elif self._near_duck_until and now >= self._near_duck_until:
            self._near_duck_until = 0.0
            self.music.unduck()

    def _watch_track(self) -> None:
        """대기 중에 곡이 바뀌면 화면(로그)에 보여 주고, 설정돼 있으면 말로도 알려 준다."""
        now = time.monotonic()
        if now < self._next_track_check:
            return
        self._next_track_check = now + 1.0
        try:
            playing = self.music.is_playing()
            self.listener.music_playing = playing  # 음악 중에는 호출어 기준을 낮춘다
            track = self.music.now_playing() if playing else None
        except Exception:
            return
        if track is None:
            if self.music.now_playing() is None:
                self._last_track = None  # 음악이 꺼짐: 다음에 트는 곡은 다시 첫 곡
            return
        if track == self._last_track:
            return
        first = self._last_track is None
        self._last_track = track
        log.info("♪ 지금 곡: %s", track)
        # 방금 틀어 달라고 한 첫 곡은 Claude 가 이미 말했으므로, 이어지는 곡부터 알려 준다.
        if self.cfg.announce_tracks and not first:
            title, _, artist = track.rpartition(" - ")
            self.music.duck()
            try:
                self.speaker.say(f"{artist}의 {title}." if title else f"{track}.", keep_english=True)
                self.speaker.wait()
            finally:
                self.music.unduck()

    # ------------------------------------------------------------------ 예약 (타이머·알람·리마인더·수면 타이머)

    def _handle_events(self) -> None:
        try:
            event = self.events.get_nowait()
        except queue.Empty:
            return
        log.info("예약 실행: %s", event.describe())
        if event.kind == "timer":
            self.music.duck()
            self.speaker.chime("timer")
            if not self._wait_or_wake(3.0):
                self._say_and_wait(f"{event.label} 타이머가 끝났어요.")
            self.music.unduck()
        elif event.kind == "reminder":
            self.music.duck()
            self.speaker.chime("wake")
            self._say_and_wait(f"알려 드릴게요. {event.message}")
            self.music.unduck()
        elif event.kind == "sleep":
            self._fade_out_music()
        elif event.kind == "alarm":
            self._ring_alarm(event)

    def _wait_or_wake(self, seconds: float) -> bool:
        """seconds 동안 기다린다. 그사이 호출어가 들리면 True (호출어 처리는 메인 루프가 이어서 한다)."""
        return self.listener.wake_event.wait(seconds)

    def _say_and_wait(self, text: str) -> None:
        self.speaker.say(text)
        while self.speaker.is_speaking():
            if self.listener.wake_event.wait(0.05):
                self.speaker.stop()  # 말하는 중에 호출어: 끊고 사용자 말을 듣는다
                return

    def _fade_out_music(self) -> None:
        """수면 타이머: 20초에 걸쳐 음악 소리를 줄이다가 끈다. 다음에 틀 때는 원래 음량으로."""
        if not (self.music.is_playing() or self.music.is_starting()):
            log.info("수면 타이머: 재생 중인 음악이 없습니다.")
            return
        volume = self.music.get_volume()
        steps = 20
        try:
            for i in range(1, steps + 1):
                self.music.set_volume(round(volume * (1 - i / steps)))
                if self._wait_or_wake(1.0):
                    log.info("수면 타이머: 호출어가 들려 줄이기를 멈춥니다.")
                    return
            self.music.stop()
            log.info("수면 타이머: 음악을 껐습니다.")
        finally:
            self.music.set_volume(volume)

    def _ring_alarm(self, event: Event) -> None:
        now = time.localtime()
        clock = spoken_clock(now.tm_hour, now.tm_min)
        music_on = False
        if event.music:
            try:
                if event.music in ("liked", "mix") and self.music.has_account():
                    self.music.play_library(event.music, shuffle=event.music == "liked")
                elif event.music in ("liked", "mix"):
                    self.music.play("아침에 듣기 좋은 노래", "playlist")
                else:
                    self.music.play(event.music, "playlist")
                music_on = True
            except Exception as exc:
                log.warning("알람 음악을 틀지 못해 알림음으로 대신합니다: %s", exc)
        if not music_on:
            # 알림음: 브리핑이 있으면 세 번 울리고 브리핑으로 넘어가고, 없으면 호출어가 들릴 때까지 (최대 약 3분)
            rings = 3 if event.briefing else 45
            for _ in range(rings):
                self.speaker.chime("timer")
                if self._wait_or_wake(4.0):
                    return  # 일어남: 사용자의 말은 메인 루프가 듣는다 ("5분만 더" 등)
        greeting = "좋은 아침이에요! " if 4 <= now.tm_hour < 11 else ""
        name = f"{event.label} 알람이에요. " if event.label else ""
        self.music.duck()
        try:
            self._say_and_wait(f"{greeting}{name}{clock}예요.")
        finally:
            self.music.unduck()
        if event.briefing and not self.listener.wake_event.is_set():
            log.info("브리핑: %s", self._converse(ALARM_BRIEFING_REQUEST) or "(없음)")


# ---------------------------------------------------------------------- 글자 모드


def run_text(cfg: Config, speak: bool) -> None:
    """마이크 없이 키보드로 대화한다 (두뇌와 도구를 시험할 때 쓴다)."""
    from .brain import Brain
    from .music import MusicPlayer

    music = MusicPlayer(cfg)
    speaker = None
    if speak:
        from .tts import Speaker

        speaker = Speaker(cfg)

    def on_fire(event: Event) -> None:
        print(f"\n[예약] {event.kind}: {event.label or event.message}")
        if speaker:
            speaker.chime("timer")

    timers = Scheduler(None, on_fire)  # 글자 모드의 예약은 저장하지 않는다
    pim = _load_pim(cfg)
    brain = Brain(cfg, ToolBox(cfg, music, timers, _load_memory(cfg), pim))

    def on_sentence(sentence: str) -> None:
        print(f"  {sentence}", flush=True)
        if speaker:
            speaker.say(sentence)

    print("글자 모드입니다. 빈 줄이나 Ctrl-D 로 끝냅니다.")
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                break
            brain.respond(text, on_sentence=on_sentence, on_tool_start=lambda name: print(f"  ({name} 실행 중...)", flush=True))
            if speaker:
                speaker.wait()
    finally:
        brain.close()
        timers.shutdown()
        music.shutdown()
        if speaker:
            speaker.shutdown()


# ---------------------------------------------------------------------- 점검


def _brain_key(cfg: Config) -> str | None:
    return {"anthropic": cfg.anthropic_api_key, "openai": cfg.openai_api_key, "google": cfg.gemini_api_key}[cfg.provider]


def run_check(cfg: Config) -> int:
    """설정과 장치를 점검해 무엇이 빠졌는지 알려 준다."""
    import shutil

    import sounddevice as sd

    ok = True

    def line(good: bool, label: str, detail: str = "", required: bool = True) -> None:
        nonlocal ok
        mark = "✓" if good else ("✗" if required else "–")
        print(f" {mark} {label}" + (f": {detail}" if detail else ""))
        if not good and required:
            ok = False

    key_name = PROVIDER_KEYS[cfg.provider]
    line(bool(_brain_key(cfg)), key_name, f"두뇌 {cfg.provider}" if _brain_key(cfg) else ".env 에 넣어 주세요")
    if cfg.tts_backend == "openai":
        line(bool(cfg.openai_api_key), "OPENAI_API_KEY", "" if cfg.openai_api_key else "없음 → macOS 내장 음성 사용", required=False)
    line(bool(cfg.mpv_path and os.path.exists(cfg.mpv_path)), "mpv", cfg.mpv_path or "brew install mpv")
    line(bool(cfg.ytdlp_path and os.path.exists(cfg.ytdlp_path)), "yt-dlp", cfg.ytdlp_path or "uv sync")
    line(bool(shutil.which("node")), "node (yt-dlp 가 사용)", shutil.which("node") or "brew install node")
    linked = cfg.ytmusic_auth_path.exists()
    line(linked, "유튜브 뮤직 계정", "연결됨" if linked else "연결 안 됨 → uv run python -m speaker.ytmusic_setup", required=False)
    try:
        mic = sd.query_devices(cfg.input_device, "input")
        line(True, "마이크", mic["name"])
    except Exception as exc:
        line(False, "마이크", str(exc))
    try:
        out = sd.query_devices(cfg.output_device, "output")
        line(True, "스피커", out["name"])
    except Exception as exc:
        line(False, "스피커", str(exc))
    voice = {"auto": cfg.qwen_tts_model, "qwen": cfg.qwen_tts_model, "openai": cfg.openai_tts_model}.get(cfg.tts_backend, cfg.say_voice)
    print(f"\n두뇌: {cfg.provider} {cfg.model} (effort {cfg.effort}) / 호출어: {cfg.wake_word}")
    print(f"음성 인식: {cfg.whisper_model} / 음성 합성: {voice}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="speaker", description="맥북 홈 스피커 어시스턴트")
    parser.add_argument("--text", action="store_true", help="마이크 대신 키보드로 대화")
    parser.add_argument("--speak", action="store_true", help="--text 모드에서도 답을 소리 내어 읽기")
    parser.add_argument("--check", action="store_true", help="설정과 장치 점검만 하고 끝내기")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.check:
        return run_check(cfg)
    setup_logging(cfg, args.verbose)
    if not _brain_key(cfg):
        print(f"{PROVIDER_KEYS[cfg.provider]} 가 없습니다. ./setup.sh 를 실행하거나 .env 에 키를 넣어 주세요.", file=sys.stderr)
        return 1
    if args.text:
        run_text(cfg, args.speak)
        return 0

    pid_file = cfg.project_dir / "logs" / "speaker.pid"
    other = _running_speaker(pid_file)
    if other:
        # 로그인 항목으로 켜진 상태에서 또 켜면 마이크·스피커를 두 개가 나눠 쓰게 된다
        log.warning("스피커가 이미 켜져 있습니다 (pid %s). 이번 실행은 끝냅니다.", other)
        return 0

    assistant = Assistant(cfg)

    def on_signal(signum: int, _frame: object) -> None:
        log.info("종료 신호(%s)를 받았습니다.", signal.Signals(signum).name)
        assistant.stopping.set()

    def on_restart(_signum: int, _frame: object) -> None:
        log.info("재시작 요청을 받았습니다.")
        assistant.restart_requested = True
        assistant.stopping.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_restart)  # ./restart.sh 가 보낸다
    pid_file.write_text(str(os.getpid()))
    try:
        assistant.run()
    finally:
        # 마이크·스피커 스트림을 닫다가 CoreAudio 안에서 멈추는 일이 있다 (네이티브 잠금이라 신호도 안 먹힌다).
        # 정리가 오래 걸리면 그냥 끝내서, 업데이트 재시작이 멈춘 채로 남지 않게 한다.
        def _force_exit() -> None:
            time.sleep(SHUTDOWN_TIMEOUT_S)
            log.error("종료 정리가 %d초 넘게 멈춰 있어 강제로 끝냅니다.", SHUTDOWN_TIMEOUT_S)
            logging.shutdown()
            os._exit(RESTART_EXIT_CODE if assistant.restart_requested else 0)

        threading.Thread(target=_force_exit, name="shutdown-watchdog", daemon=True).start()
        assistant.shutdown()
        try:
            if pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass
    # start.command 는 이 코드를 받으면 곧바로 다시 켠다
    return RESTART_EXIT_CODE if assistant.restart_requested else 0
