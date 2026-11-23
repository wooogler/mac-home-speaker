"""음성 출력: 문장 단위 TTS(로컬 Qwen3-TTS / OpenAI / macOS say) + 효과음(chime).

구조
- say(text)  -> 텍스트 큐 -> 합성 워커(문장 N+1 을 N 재생 중에 미리 합성) -> 믹서 FIFO
- 믹서는 sounddevice OutputStream 콜백(또는 테스트용 null 싱크)에서 블록 단위로 PCM 을 꺼내고,
  효과음은 그 위에 섞는다. 그래서 효과음이 말을 끊지도, 말에 막히지도 않는다.
- stop() 은 세대(generation) 번호를 올려 큐/FIFO 를 비운다. 진행 중이던 합성 결과는 버려진다.
"""

from __future__ import annotations

import collections
import io
import logging
import re
import subprocess
import threading
import time
import wave
from typing import Callable, Iterator

import numpy as np

from speaker.config import Config

log = logging.getLogger("speaker.tts")

SAMPLE_RATE = 24000  # OpenAI pcm 출력과 같은 24 kHz mono
BLOCK = 480  # 20 ms
# 출력 스트림은 계속 열어 둔다. 조용할 때마다 닫았다 열면 맥북 스피커에서 '툭' 하는 소리가 났다.
# (장치가 사라지는 등 스트림이 멈추면 _monitor 가 다시 연다)
IDLE_CLOSE_S = float("inf")
OPENAI_COOLDOWN_S = 60.0  # OpenAI 실패 후 이 시간 동안은 바로 say 사용

OPENAI_INSTRUCTIONS = (
    "자연스럽고 따뜻한 한국어 대화체로 말하세요. 친근한 홈 어시스턴트처럼, "
    "평소보다 살짝 빠르고 경쾌한 속도로, 또박또박 말하세요."
)

QWEN_SILENCE_RMS = 0.012  # 20 ms 프레임의 RMS 가 이보다 작으면 조용한 것으로 본다
QWEN_MAX_PAUSE_S = float("inf")  # 문장 안 긴 쉼 줄이기: 꺼 둠 (줄이면 말이 잘리는 느낌이 났다). 켜려면 0.3
QWEN_PAUSE_FLOOR_S = 0.22  # 줄여도 이만큼은 남기고, 넘는 부분은 QWEN_PAUSE_RATIO 만큼만 남긴다
QWEN_PAUSE_RATIO = 0.35


def _shorten_pause(held: list, frame: int) -> list:
    """문장 안의 긴 쉼을 줄인다 (모델이 '바람은… 약하니'처럼 이상한 곳에서 길게 쉬는 버릇).

    통째로 자르지 않고 길이에 비례해 줄이며, 앞뒤를 남기고 이음매를 겹쳐(크로스페이드) 이어 붙인다.
    """
    length = len(held) * frame / SAMPLE_RATE
    target = QWEN_PAUSE_FLOOR_S + (length - QWEN_PAUSE_FLOOR_S) * QWEN_PAUSE_RATIO
    keep = max(2, int(target * SAMPLE_RATE / frame))
    head, tail = held[: keep // 2 + 1], held[-(keep - keep // 2):]
    a, b = np.concatenate(head), np.concatenate(tail)
    n = frame  # 20 ms 겹치기
    fade = np.linspace(1.0, 0.0, n, dtype=np.float32)
    joined = np.concatenate([a[:-n], a[-n:] * fade + b[:n] * (1 - fade), b[n:]])
    return [joined]
QWEN_RAMBLE_GAP_S = 1.2  # 문장을 다 읽었을 시점 이후에 이만큼 조용하면 합성을 끝낸다

class _Stretcher:
    """음높이는 그대로 두고 말 속도만 바꾼다 (WSOLA). 스트리밍 조각을 차례로 넣고 끝에 flush."""

    def __init__(self, speed: float) -> None:
        from audiotsm import wsola

        self.tsm = wsola(1, speed=speed)

    def process(self, pcm: np.ndarray) -> np.ndarray:
        from audiotsm.io.array import ArrayReader, ArrayWriter

        reader, writer = ArrayReader(pcm.reshape(1, -1).astype(np.float32)), ArrayWriter(1)
        while not reader.empty:
            self.tsm.read_from(reader)
            self.tsm.write_to(writer)
        return writer.data[0].astype(np.float32)

    def flush(self) -> np.ndarray:
        from audiotsm.io.array import ArrayWriter

        writer = ArrayWriter(1)
        finished = False
        while not finished:
            _, finished = self.tsm.flush_to(writer)
        return writer.data[0].astype(np.float32)


# 스피커가 정해진 문구로 자주 하는 말 (clean_text 를 거친 형태와 같아야 한다)
QWEN_FIXED_PHRASES = ("네.", "네, 잠시만요.", "찾아볼게요.", "준비됐어요.")

SynthFn = Callable[[str], Iterator[np.ndarray]]
Sink = Callable[[np.ndarray], None]

# ---------------------------------------------------------------- 텍스트 정리

_RE_FENCE = re.compile(r"```.*?(```|$)", re.S)
_RE_MDLINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_RE_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_RE_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_RE_QUOTE = re.compile(r"^\s*>+\s?", re.M)
_RE_BULLET = re.compile(r"^\s*(?:[-*+•·▪●◦]|\d{1,3}[.)])\s+", re.M)
_RE_HRULE = re.compile(r"^\s*([-*_=])\1{2,}\s*$", re.M)
_RE_EMPH = re.compile(r"\*+|__+|~~|`+")
_RE_TABLE = re.compile(r"\s*\|\s*")
_RE_SPACE = re.compile(r"\s+")
_RE_SPEAKABLE = re.compile(r"[^\W_]")  # 글자나 숫자가 하나라도 있어야 말할 거리가 있음


def clean_text(text: str) -> str:
    """마크다운/URL 을 걷어내고 공백을 정리한다. 말할 게 없으면 빈 문자열."""
    if not text:
        return ""
    s = _RE_FENCE.sub(" ", text)
    s = _RE_MDLINK.sub(r"\1", s)
    s = _RE_URL.sub("링크", s)
    s = _RE_HRULE.sub(" ", s)
    s = _RE_HEADING.sub("", s)
    s = _RE_QUOTE.sub("", s)
    s = _RE_BULLET.sub("", s)
    s = _RE_EMPH.sub("", s)
    s = _RE_TABLE.sub(" ", s)
    s = _RE_SPACE.sub(" ", s).strip()
    return s if _RE_SPEAKABLE.search(s) else ""


_RE_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_HANGUL = re.compile(r"[가-힣]")
# 알파벳 단어 4개 이상이 이어진 부분 (사이에 공백·쉼표·숫자 정도만 끼는 경우)
_RE_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'’\-]*(?:[\s,;:]+(?:\d+[\s,;:]+)?[A-Za-z][A-Za-z'’\-]*){3,}[.!?]?")
_RE_FAHRENHEIT = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*[Ff](?![A-Za-z])")
_RE_CELSIUS = re.compile(r"°\s*C(?![A-Za-z])|℃|°")
_RE_DATE = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])")
_RE_RANGE = re.compile(r"(\d)\s*~\s*(\d)")


def _english_like(run: str) -> bool:
    """알파벳 단어 묶음이 문장처럼 보이는지 (소문자로 시작하는 단어가 절반 이상). 곡 제목 같은 이름은 남긴다."""
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", run)
    return sum(w[0].islower() for w in words) * 2 >= len(words)


def _fahrenheit(m: re.Match) -> str:
    c = round((float(m.group(1)) - 32) * 5 / 9)
    return f"영하 {-c}도" if c < 0 else f"영상 {c}도" if c > 0 else "0도"


def _date(m: re.Match) -> str:
    month, day = int(m.group(1)), int(m.group(2))
    return f"{month}월 {day}일" if 1 <= month <= 12 and 1 <= day <= 31 else m.group(0)


def speakable(text: str, keep_english: bool = False) -> str:
    """음성 합성기가 한국어로 읽을 수 있게 다듬는다.

    이 합성기는 한국어 모드라 영어 문장이나 기호를 만나면 알아들을 수 없는 소리를 낸다.
    거의 영어로만 된 문장과 문장처럼 이어진 영어 구절은 빼고, 화씨·날짜·범위·퍼센트 같은 표기는 말로 바꾼다.
    keep_english=True 면 영어는 그대로 둔다 (곡 제목 안내처럼 원래 이름을 말해야 할 때).
    """
    if not keep_english:
        kept = []
        for sent in _RE_SENTENCES.split(text):
            latin, hangul = len(_RE_LATIN.findall(sent)), len(_RE_HANGUL.findall(sent))
            if latin >= 6 and hangul <= 2:  # 한글이 거의 없는 영어 문장
                log.info("영어 문장이라 읽지 않습니다: %r", sent)
                continue
            for run in _RE_LATIN_RUN.findall(sent):
                if _english_like(run):
                    log.info("영어 구절이라 읽지 않습니다: %r", run)
                    sent = sent.replace(run, " ", 1)
            kept.append(sent)
        text = " ".join(kept)
    text = _RE_FAHRENHEIT.sub(_fahrenheit, text)
    text = _RE_CELSIUS.sub("도", text)
    text = _RE_DATE.sub(_date, text)
    text = _RE_RANGE.sub(r"\1에서 \2", text)
    text = text.replace("%", " 퍼센트")
    text = _RE_SPACE.sub(" ", text).strip()
    return text if _RE_SPEAKABLE.search(text) else ""


# ---------------------------------------------------------------- 효과음


def _tone(freq: float, dur: float, amp: float = 0.2, fade: float = 0.012,
          freq_end: float | None = None) -> np.ndarray:
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    if freq_end is None:
        phase = 2 * np.pi * freq * t
    else:  # 선형 글라이드
        phase = 2 * np.pi * (freq * t + (freq_end - freq) * t * t / (2 * dur))
    y = np.sin(phase)
    # 부드럽게: 짧은 선형 페이드 인/아웃 + 약한 지수 감쇠
    f = min(int(SAMPLE_RATE * fade), n // 2)
    env = np.ones(n)
    if f > 0:
        env[:f] = np.linspace(0, 1, f)
        env[-f:] = np.linspace(1, 0, f)
    env *= np.exp(-t * 2.0)
    return (amp * y * env).astype(np.float32)


def _gap(dur: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * dur), dtype=np.float32)


def make_chime(kind: str) -> np.ndarray:
    """효과음 PCM (float32 mono, 24 kHz, peak ~0.2)."""
    if kind == "wake":  # 올라가는 두 음: 듣고 있어요
        parts = [_tone(660, 0.09), _gap(0.03), _tone(990, 0.13)]
    elif kind == "done":  # 부드러운 한 음: 알겠어요
        parts = [_tone(880, 0.16, amp=0.16, fade=0.02)]
    elif kind == "error":  # 낮게 내려가는 음
        parts = [_tone(440, 0.16), _gap(0.03), _tone(300, 0.26, freq_end=240)]
    elif kind == "timer":  # 알람: (삐삐) x4, 약 3초
        beep = [_tone(1046, 0.12, fade=0.008), _gap(0.08), _tone(1046, 0.12, fade=0.008)]
        parts = []
        for _ in range(4):
            parts += beep + [_gap(0.6)]
        parts = parts[:-1]
    else:
        log.warning("알 수 없는 효과음 종류: %r (done 으로 대체)", kind)
        return make_chime("done")
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------- 믹서

_END = object()  # 문장 끝 표시 (FIFO 안에서 재생 순서대로 소비됨)


class _Mixer:
    """말 FIFO + 효과음 목록을 블록 단위로 섞는다. 모든 상태는 하나의 lock 으로 보호."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.gen = 0
        self.pending = 0  # 큐에 있거나 합성 중이거나 재생 중인 문장 수
        self.fifo: collections.deque = collections.deque()  # [arr, pos] 또는 _END
        self.chimes: list[list] = []  # [arr, pos]
        self.last_active = time.monotonic()
        self.on_work: Callable[[], None] = lambda: None

    # --- 생산자 쪽
    def begin(self) -> int:
        with self.lock:
            self.pending += 1
            self.last_active = time.monotonic()
            return self.gen

    def current(self, gen: int) -> bool:
        return gen == self.gen

    def push(self, gen: int, pcm: np.ndarray) -> bool:
        if pcm.size == 0:
            return gen == self.gen
        with self.lock:
            if gen != self.gen:
                return False
            self.fifo.append([pcm, 0])
        self.on_work()
        return True

    def end(self, gen: int) -> None:
        with self.lock:
            if gen != self.gen:
                return
            self.fifo.append(_END)
        self.on_work()

    def add_chime(self, pcm: np.ndarray) -> None:
        with self.lock:
            self.chimes.append([pcm, 0])
        self.on_work()

    def reset(self) -> None:
        with self.cond:
            self.gen += 1
            self.fifo.clear()
            self.chimes.clear()
            self.pending = 0
            self.cond.notify_all()

    # --- 소비자 쪽 (오디오 콜백 스레드)
    def has_audio(self) -> bool:
        return bool(self.fifo) or bool(self.chimes)

    def busy(self) -> bool:
        return self.pending > 0 or bool(self.chimes)

    def render(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self.cond:
            i = 0
            fifo = self.fifo
            while fifo:
                item = fifo[0]
                if item is _END:
                    fifo.popleft()
                    self.pending = max(0, self.pending - 1)
                    self.cond.notify_all()
                    continue
                if i >= n:
                    break
                arr, pos = item
                k = min(n - i, arr.size - pos)
                out[i:i + k] = arr[pos:pos + k]
                i += k
                pos += k
                if pos >= arr.size:
                    fifo.popleft()
                else:
                    item[1] = pos
            if self.chimes:
                alive = []
                for c in self.chimes:
                    arr, pos = c
                    k = min(n, arr.size - pos)
                    out[:k] += arr[pos:pos + k]
                    c[1] = pos + k
                    if c[1] < arr.size:
                        alive.append(c)
                self.chimes = alive
            if self.fifo or self.chimes or self.pending:
                self.last_active = time.monotonic()
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def drain(self) -> None:
        """출력 장치를 못 열 때: 쌓인 소리를 버리되 문장 완료 처리는 해서 wait() 이 멈추지 않게."""
        while True:
            with self.lock:
                if not self.fifo and not self.chimes:
                    return
            self.render(SAMPLE_RATE)


# ---------------------------------------------------------------- 출력 드라이버


class _DeviceDriver:
    """실제 sounddevice OutputStream. 필요할 때 열고, 한동안 조용하면 닫는다."""

    def __init__(self, mixer: _Mixer, device: int | str | None) -> None:
        self.mixer = mixer
        self.device = device
        self.lock = threading.Lock()
        self.stream = None
        self.failed_at = 0.0
        self.closed = False
        self.monitor = threading.Thread(target=self._monitor, name="tts-idle", daemon=True)
        self.monitor.start()

    def _callback(self, outdata, frames, _time, status) -> None:  # 오디오 스레드
        outdata[:, 0] = self.mixer.render(frames)

    def ensure(self) -> None:
        drain = False
        with self.lock:
            if self.closed:
                drain = True
            elif self.stream is None:
                if time.monotonic() - self.failed_at < 2.0:
                    drain = True
                else:
                    try:
                        import sounddevice as sd

                        s = sd.OutputStream(
                            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=BLOCK, latency="low", device=self.device,
                            callback=self._callback,
                        )
                        s.start()
                        self.stream = s
                        log.debug("출력 스트림 열림 (device=%r)", self.device)
                    except Exception as e:  # 장치 사용 중/없음 등
                        self.failed_at = time.monotonic()
                        log.warning("출력 장치를 열 수 없습니다: %s", e)
                        drain = True
        if drain:
            self.mixer.drain()

    def _close_stream(self) -> None:
        s, self.stream = self.stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception as e:
                log.debug("스트림 닫기 실패: %s", e)

    def _monitor(self) -> None:
        while not self.closed:
            time.sleep(0.5)
            with self.lock:
                if (self.stream is not None and not self.mixer.busy()
                        and time.monotonic() - self.mixer.last_active > IDLE_CLOSE_S):
                    self._close_stream()
                    log.debug("출력 스트림 닫힘 (유휴)")
                elif self.stream is not None and not self.stream.active:
                    # 장치가 사라진 경우 등: 다음 요청 때 다시 연다
                    log.warning("출력 스트림이 멈춰 있어 다시 엽니다")
                    self._close_stream()
                    if self.mixer.has_audio():
                        threading.Thread(target=self.ensure, daemon=True).start()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self._close_stream()


class _NullDriver:
    """테스트용: 소리를 내지 않고 sink(block) 로 PCM 을 넘긴다. realtime=True 면 실제 속도로."""

    def __init__(self, mixer: _Mixer, sink: Sink, realtime: bool) -> None:
        self.mixer = mixer
        self.sink = sink
        self.realtime = realtime
        self.wake = threading.Event()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="tts-null", daemon=True)
        self.thread.start()

    def ensure(self) -> None:
        self.wake.set()

    def _run(self) -> None:
        block_s = BLOCK / SAMPLE_RATE
        while not self.closed:
            self.wake.wait(0.05)
            self.wake.clear()
            t_next = time.monotonic()
            while not self.closed and self.mixer.has_audio():
                self.sink(self.mixer.render(BLOCK))
                if self.realtime:
                    t_next += block_s
                    d = t_next - time.monotonic()
                    if d > 0:
                        time.sleep(d)

    def close(self) -> None:
        self.closed = True
        self.wake.set()


# ---------------------------------------------------------------- 합성 백엔드


def _list_voices() -> list[tuple[str, str]]:
    """[(이름, 로캘)] — `say -v '?'` 출력 파싱."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        log.warning("say 음성 목록을 읽지 못했습니다: %s", e)
        return []
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Za-z0-9]+)\s+#", line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))
    return voices


def pick_say_voice(wanted: str) -> str | None:
    voices = _list_voices()
    if not voices:
        return wanted or None
    ko = [n for n, loc in voices if loc == "ko_KR"]

    def match(names: list[str], w: str) -> str | None:
        wl = w.lower()
        for n in names:  # "Eddy" -> "Eddy (Korean (South Korea))"
            if n.lower() == wl or n.lower().startswith(wl + " ("):
                return n
        return None

    if wanted:
        v = match(ko, wanted) or match([n for n, _ in voices], wanted)
        if v:
            return v
        log.warning("say 음성 %r 이(가) 없습니다", wanted)
    v = match(ko, "Yuna") or (ko[0] if ko else None)
    if v:
        log.info("say 음성 대체: %s", v)
    return v


def _wav_to_float(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        raw = w.readframes(w.getnframes())
        sr, ch = w.getframerate(), w.getnchannels()
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:  # 이론상 안 일어나지만 안전하게 선형 보간
        x = np.linspace(0, len(pcm), int(len(pcm) * SAMPLE_RATE / sr), endpoint=False)
        pcm = np.interp(x, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


class _SayBackend:
    def __init__(self, voice: str | None, rate: int) -> None:
        self.voice = voice
        self.rate = rate

    def synth(self, text: str, alive: Callable[[], bool]) -> np.ndarray | None:
        """한 문장을 PCM 으로. alive() 가 False 가 되면 즉시 중단(None)."""
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="speaker-tts-")
        os.close(fd)
        cmd = ["say", "-r", str(self.rate), "-o", path,
               "--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}"]
        if self.voice:
            cmd[1:1] = ["-v", self.voice]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE)
            p.stdin.write(text.encode("utf-8"))  # 텍스트는 stdin 으로 ('-' 로 시작해도 안전)
            p.stdin.close()
            deadline = time.monotonic() + 30
            while p.poll() is None:
                if not alive() or time.monotonic() > deadline:
                    p.kill()
                    p.wait()
                    return None
                time.sleep(0.02)
            if p.returncode != 0:
                log.error("say 실패 (%s): %s", p.returncode, p.stderr.read().decode(errors="replace").strip())
                return None
            with open(path, "rb") as f:
                return _wav_to_float(f.read())
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class _OpenAIBackend:
    def __init__(self, cfg: Config, *, base_url: str | None = None, timeout: float = 15.0) -> None:
        import openai

        kw = {}
        if base_url:
            kw["base_url"] = base_url
        try:
            to = openai.Timeout(timeout, connect=min(4.0, timeout))
        except Exception:
            to = timeout
        self.client = openai.OpenAI(api_key=cfg.openai_api_key, timeout=to, max_retries=0, **kw)
        self.model = cfg.openai_tts_model
        self.voice = cfg.openai_tts_voice
        # instructions 는 gpt-4o 계열 TTS 만 지원 (tts-1 / tts-1-hd 는 미지원)
        self.instructions = OPENAI_INSTRUCTIONS if not self.model.startswith("tts-1") else None

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """24 kHz 16-bit mono PCM 을 받는 대로 float32 조각으로 내보낸다."""
        kw = dict(model=self.model, voice=self.voice, input=text, response_format="pcm")
        if self.instructions:
            kw["instructions"] = self.instructions
        with self.client.audio.speech.with_streaming_response.create(**kw) as resp:
            rest = b""
            for chunk in resp.iter_bytes(4800):
                buf = rest + chunk
                cut = len(buf) & ~1
                rest = buf[cut:]
                if cut:
                    yield np.frombuffer(buf[:cut], dtype="<i2").astype(np.float32) / 32768.0


class _QwenBackend:
    """로컬 Qwen3-TTS (mlx-audio). 모델 로드와 합성은 모두 합성 워커 스레드 한 곳에서만 한다."""

    def __init__(self, cfg: Config) -> None:
        self.repo = cfg.qwen_tts_model
        self.speaker = cfg.qwen_tts_speaker
        self.instruct = cfg.qwen_tts_instruct  # 말투 지시문 (1.7B 모델만 반영)
        self.model = None
        self._cache: dict[str, np.ndarray] = {}

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model

        from speaker.mlx_lock import MLX_LOCK

        t0 = time.monotonic()
        with MLX_LOCK:
            model = load_model(self.repo)
        if model.sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"샘플레이트 {model.sample_rate}Hz 는 지원하지 않습니다")
        speakers = [s.lower() for s in model.get_supported_speakers()]
        if self.speaker.lower() not in speakers:
            raise RuntimeError(f"화자 {self.speaker!r} 가 없습니다 (가능: {speakers})")
        self.model = model
        # 자주 쓰는 고정 문구는 세 번 만들어 길이가 가운데인 것을 저장해 둔다.
        # 이 모델은 수십 번에 한 번꼴로 문장 뒤에 뜻 없는 소리를 덧붙이는데, 그런 결과를 걸러내는 동시에
        # 예열도 겸한다. 저장된 문구는 합성 없이 곧바로 재생된다.
        for phrase in QWEN_FIXED_PHRASES:
            takes = [np.concatenate(list(self._generate(phrase, lambda: True))) for _ in range(3)]
            self._cache[phrase] = sorted(takes, key=len)[1]
        log.info("Qwen 음성 모델 준비 완료 (%.1f초): %s, 화자 %s", time.monotonic() - t0, self.repo, self.speaker)

    @staticmethod
    def _token_cap(text: str) -> int:
        """글자 수에 맞춘 생성 길이 상한 (12.5 토큰이 1초).

        정상 합성은 글자당 2~3토큰쯤이라 넉넉히 통과하고, 군말이 붙기 시작하면 여기서 잘린다.
        숫자는 읽으면 여러 음절이 되므로 더 쳐준다.
        """
        return int(8 + 3.5 * _QwenBackend._weighted_len(text))

    @staticmethod
    def _weighted_len(text: str) -> float:
        return len(text) + 1.5 * sum(ch.isdigit() for ch in text)

    def stream(self, text: str, alive: Callable[[], bool]) -> Iterator[np.ndarray]:
        """24 kHz float32 조각을 만들어지는 대로 내보낸다. alive() 가 False 가 되면 그만둔다."""
        cached = self._cache.get(text)
        if cached is not None:
            yield cached
            return
        yield from self._generate(text, alive)

    def _generate(self, text: str, alive: Callable[[], bool]) -> Iterator[np.ndarray]:
        from speaker.mlx_lock import MLX_LOCK

        it = self.model.generate_custom_voice(
            text=text,
            speaker=self.speaker,
            language="korean",
            instruct=self.instruct,
            max_tokens=self._token_cap(text),
            stream=True,
            streaming_interval=0.32,
        )
        # 군말 차단: 이 모델은 가끔 문장을 다 읽고 1.3초 넘게 조용하다가 뜻 없는 소리를 덧붙인다.
        # 그래서 조용한 구간은 일단 쥐고 있다가 뒤에 소리가 이어질 때만 내보내고, 긴 침묵을 만나면 끝낸다.
        # 다만 쉼표 뒤의 쉼도 1초를 넘을 때가 있으므로, 글자 수로 보아 문장을 다 읽었을 만큼
        # (글자당 1.3토큰, 정상 합성의 최소치보다 조금 아래) 소리가 나온 뒤에만 이 규칙을 적용한다.
        frame = BLOCK  # 20 ms
        held: list[np.ndarray] = []  # 아직 내보내지 않은 조용한 프레임
        spoke = False
        emitted = 0  # 지금까지 내보낸 샘플 수 (= 쥐고 있는 침묵이 시작된 위치)
        complete_after = int(1.3 * self._weighted_len(text) / 12.5 * SAMPLE_RATE)
        rest = np.zeros(0, dtype=np.float32)
        try:
            while alive():
                with MLX_LOCK:
                    result = next(it, None)
                    pcm = None if result is None else np.array(result.audio, dtype=np.float32).reshape(-1)
                if pcm is None:
                    break
                buf = np.concatenate([rest, pcm])
                cut = len(buf) - len(buf) % frame
                rest = buf[cut:]
                out: list[np.ndarray] = []
                for i in range(0, cut, frame):
                    piece = buf[i:i + frame]
                    if float(np.sqrt(np.mean(piece * piece))) < QWEN_SILENCE_RMS:
                        held.append(piece)
                        if (spoke and emitted >= complete_after
                                and len(held) * frame >= QWEN_RAMBLE_GAP_S * SAMPLE_RATE):
                            if out:
                                yield np.concatenate(out)
                            log.debug("Qwen: 긴 침묵에서 합성을 끝냄: %r", text)
                            return
                    else:
                        emitted += (len(held) + 1) * frame
                        if spoke and len(held) * frame > QWEN_MAX_PAUSE_S * SAMPLE_RATE:
                            held = _shorten_pause(held, frame)
                        out.extend(held)
                        held.clear()
                        out.append(piece)
                        spoke = True
                if out:
                    yield np.concatenate(out)
            # 정상 종료: 끝의 침묵은 짧게만 남긴다
            tail = held[: int(0.25 * SAMPLE_RATE / frame)]
            if spoke and (tail or rest.size):
                yield np.concatenate(tail + [rest])
        finally:
            it.close()


# ---------------------------------------------------------------- Speaker


class Speaker:
    """어시스턴트의 목소리. say/stop/chime 은 어느 스레드에서 불러도 안전하다.

    테스트용 키워드 인자:
      sink=callable  -> 실제 장치 대신 float32 블록을 이 함수로 넘김 (소리 없음)
      realtime=False -> sink 를 실시간 속도가 아니라 즉시 소비
      synth=callable -> text -> Iterator[np.ndarray] 합성 함수 대체
      openai_base_url / openai_timeout -> OpenAI 클라이언트 설정(폴백 테스트용)
    """

    def __init__(self, cfg: Config, *, sink: Sink | None = None, realtime: bool = True,
                 synth: SynthFn | None = None, openai_base_url: str | None = None,
                 openai_timeout: float = 15.0) -> None:
        self.cfg = cfg
        self._mixer = _Mixer()
        self._gain = 1.0  # 목소리·알림음 음량 (시스템 음량·음악 음량과 따로)
        self._speed = cfg.voice_speed  # 말 속도 배율 (Qwen 음성에만 적용)
        self._say = _SayBackend(pick_say_voice(cfg.say_voice), cfg.say_rate)
        self._openai: _OpenAIBackend | None = None
        self._openai_fail_until = 0.0
        self._synth_override = synth

        self._qwen: _QwenBackend | None = None
        self._ready = threading.Event()  # Qwen 모델 로드가 끝났는지 (다른 백엔드는 곧바로 켜짐)

        want = (cfg.tts_backend or "auto").lower()
        if want not in ("auto", "qwen", "openai", "say"):
            log.warning("알 수 없는 TTS_BACKEND=%r, auto 로 처리", want)
            want = "auto"
        if want in ("auto", "qwen"):
            if synth is None:  # 합성 함수를 바꿔 끼운 테스트에서는 모델을 불러오지 않는다
                self._qwen = _QwenBackend(cfg)
        elif want == "openai" and cfg.openai_api_key:
            try:
                self._openai = _OpenAIBackend(cfg, base_url=openai_base_url, timeout=openai_timeout)
            except Exception as e:
                log.error("OpenAI TTS 초기화 실패, say 사용: %s", e)
        elif want == "openai":
            log.warning("OPENAI_API_KEY 가 없어 say 로 말합니다")
        self.backend = "qwen" if self._qwen else "openai" if self._openai else "say"
        log.info("TTS 백엔드: %s (실패 시 say 음성: %s)", self.backend, self._say.voice)

        if sink is not None:
            self._driver = _NullDriver(self._mixer, sink, realtime)
        else:
            self._driver = _DeviceDriver(self._mixer, cfg.output_device)
        self._mixer.on_work = self._driver.ensure

        self._queue: collections.deque = collections.deque()
        self._qcond = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._work, name="tts-synth", daemon=True)
        self._worker.start()

    # --- 공개 API
    def say(self, text: str, keep_english: bool = False) -> None:
        clean = speakable(clean_text(text), keep_english)
        if not clean or self._closed:
            return
        with self._qcond:
            gen = self._mixer.begin()
            self._queue.append((gen, clean))
            self._qcond.notify()

    def warmup(self) -> None:
        """음성 모델이 준비될 때까지 기다린다 (Qwen 은 처음에 모델을 불러오는 데 시간이 걸린다)."""
        self._ready.wait()

    def wait(self) -> None:
        m = self._mixer
        with m.cond:
            m.cond.wait_for(lambda: m.pending == 0 or self._closed)

    def stop(self) -> None:
        with self._qcond:
            self._queue.clear()
            self._mixer.reset()

    def set_voice_speed(self, speed: float) -> None:
        """말 속도 배율 (1.0 이 모델 그대로). 음높이는 바뀌지 않는다. 다음 문장부터 적용."""
        self._speed = max(0.7, min(1.6, float(speed)))

    def get_voice_speed(self) -> float:
        return self._speed

    def set_voice_volume(self, percent: int) -> None:
        """목소리·알림음 음량 0~100. 다음 문장부터 적용된다. 100 이 원래 크기."""
        self._gain = max(0, min(100, int(percent))) / 100.0

    def get_voice_volume(self) -> int:
        return round(self._gain * 100)

    def _apply_gain(self, pcm: np.ndarray) -> np.ndarray:
        return pcm if self._gain == 1.0 else (pcm * self._gain).astype(np.float32)

    def is_speaking(self) -> bool:
        return self._mixer.pending > 0

    def chime(self, kind: str) -> None:
        try:
            self._mixer.add_chime(self._apply_gain(make_chime(kind)))
        except Exception as e:  # 효과음은 절대 예외를 내지 않는다
            log.warning("효과음 재생 실패 (%s): %s", kind, e)

    def shutdown(self) -> None:
        if self._closed:
            return
        self.stop()
        with self._qcond:
            self._closed = True
            self._qcond.notify_all()
        with self._mixer.cond:
            self._mixer.cond.notify_all()
        self._worker.join(timeout=2)
        self._driver.close()

    # --- 합성 워커
    def _work(self) -> None:
        if self._qwen is not None:
            try:
                self._qwen.load()
            except Exception:
                log.exception("Qwen 음성 모델을 불러오지 못해 macOS 내장 음성으로 말합니다")
                self._qwen = None
                self.backend = "say"
        self._ready.set()
        while True:
            with self._qcond:
                while not self._queue and not self._closed:
                    self._qcond.wait()
                if self._closed:
                    return
                gen, text = self._queue.popleft()
            if not self._mixer.current(gen):
                continue
            t0 = time.monotonic()
            try:
                self._synthesize(gen, text)
            except Exception:
                log.exception("TTS 합성 중 오류: %r", text)
            finally:
                self._mixer.end(gen)  # 실패해도 문장 완료 처리 (stale 이면 무시됨)
            log.debug("합성 %.2fs: %s", time.monotonic() - t0, text)

    def _synthesize(self, gen: int, text: str) -> None:
        m = self._mixer
        alive = lambda: m.current(gen)  # noqa: E731
        if self._synth_override is not None:
            for pcm in self._synth_override(text):
                if not m.push(gen, self._apply_gain(pcm)):
                    return
            return
        if self._qwen is not None:
            pushed = False
            stretch = _Stretcher(self._speed) if abs(self._speed - 1.0) > 0.01 else None
            try:
                for pcm in self._qwen.stream(text, alive):
                    if stretch is not None:
                        pcm = stretch.process(pcm)
                    if pcm.size and not m.push(gen, self._apply_gain(pcm)):
                        return
                    pushed = True
                if stretch is not None and alive():
                    m.push(gen, self._apply_gain(stretch.flush()))
            except Exception:
                log.exception("Qwen 합성 실패, say 로 대체: %r", text)
            if pushed or not alive():
                return  # 이미 일부 재생됨: 같은 문장을 반복하지 않음
        if self._openai and time.monotonic() >= self._openai_fail_until:
            pushed = False
            try:
                for pcm in self._openai.stream(text):
                    if not m.push(gen, self._apply_gain(pcm)):
                        return
                    pushed = True
                return
            except Exception as e:
                self._openai_fail_until = time.monotonic() + OPENAI_COOLDOWN_S
                log.warning("OpenAI TTS 실패, say 로 대체 (%ds간): %s: %s",
                            OPENAI_COOLDOWN_S, type(e).__name__, e)
                if pushed or not alive():
                    return  # 이미 일부 재생됨: 같은 문장을 반복하지 않음
        pcm = self._say.synth(text, alive)
        if pcm is not None:
            m.push(gen, self._apply_gain(pcm))


# ---------------------------------------------------------------- 자체 테스트


def _selftest(audible: bool) -> int:  # pragma: no cover - 수동 실행용
    import argparse  # noqa: F401
    import dataclasses

    from speaker.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ok = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK ' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")

    if audible:
        spk = Speaker(cfg)
        print("백엔드:", spk.backend)
        spk.chime("wake")
        time.sleep(0.4)
        spk.say("안녕하세요. 맥 스피커 음성 테스트입니다.")
        spk.wait()
        spk.chime("done")
        time.sleep(0.5)
        spk.shutdown()
        return 0

    # 1) 텍스트 정리
    cases = {
        "**굵게** 그리고 `코드`": "굵게 그리고 코드",
        "## 제목\n- 첫째\n- 둘째": "제목 첫째 둘째",
        "1. 사과\n2) 배": "사과 배",
        "자세한 건 https://example.com/a?b=1 참고": "자세한 건 링크 참고",
        "[문서](https://x.y) 보세요": "문서 보세요",
        "  여러   줄\n\n공백  ": "여러 줄 공백",
        "...!?": "",
        "😀🎉": "",
        "```py\nprint(1)\n```": "",
        "3~5개 정도": "3~5개 정도",
    }
    for src, want in cases.items():
        got = clean_text(src)
        check(f"clean_text {src!r}", got == want, f"-> {got!r}")

    spoken = {
        "Tue 11/17 Low 28°f, Low 28F.": "",
        "Winds light and variable.라는 예보가 있었어요.": "라는 예보가 있었어요.",
        "낮 최고는 72°F예요.": "낮 최고는 영상 22도예요.",
        "11/17에 15~16도, 습도 40%예요.": "11월 17일에 15에서 16도, 습도 40 퍼센트예요.",
        "첫 곡은 'Landing in Love'예요.": "첫 곡은 'Landing in Love'예요.",
        "Alexa와 하는 Teams 미팅이 있어요.": "Alexa와 하는 Teams 미팅이 있어요.",
        "섭씨 3°C예요.": "섭씨 3도예요.",
    }
    for src, want in spoken.items():
        got = speakable(src)
        check(f"speakable {src!r}", got == want, f"-> {got!r}")
    check("speakable keep_english", speakable("Coldplay의 Viva La Vida And The Rest.", True)
          == "Coldplay의 Viva La Vida And The Rest.")

    # 2) 효과음
    for kind in ("wake", "done", "error", "timer"):
        c = make_chime(kind)
        peak = float(np.abs(c).max())
        check(f"chime {kind}", c.ndim == 1 and c.dtype == np.float32 and 0.1 < peak <= 0.21
              and abs(c[0]) < 0.01 and abs(c[-1]) < 0.01,
              f"shape={c.shape} dur={c.size / SAMPLE_RATE:.2f}s peak={peak:.3f}")

    # 3) 파이프라인 (가짜 합성 + null 싱크, 실시간 속도)
    say_cfg = dataclasses.replace(cfg, tts_backend="say", openai_api_key=None)
    got: list[float] = []
    lock = threading.Lock()

    def sink(block: np.ndarray) -> None:
        with lock:
            for v in np.unique(block[block > 0]):
                v = round(float(v), 3)
                if not got or got[-1] != v:
                    got.append(v)

    def fake(text: str) -> Iterator[np.ndarray]:
        val, dur = text.split()
        time.sleep(0.05)  # 합성 지연 흉내
        n = int(float(dur) * SAMPLE_RATE)
        for i in range(0, n, 2400):
            yield np.full(min(2400, n - i), float(val), dtype=np.float32)

    spk = Speaker(say_cfg, sink=sink, synth=fake)
    check("is_speaking 초기 False", not spk.is_speaking())
    t0 = time.monotonic()
    for v in ("0.1", "0.2", "0.3"):
        spk.say(f"{v} 0.3")
    check("say() 즉시 반환", time.monotonic() - t0 < 0.01, f"{(time.monotonic() - t0) * 1000:.1f} ms")
    check("is_speaking True", spk.is_speaking())
    spk.wait()
    el = time.monotonic() - t0
    check("wait() 이 재생 끝까지 대기 (~0.9s)", 0.85 < el < 1.3, f"{el:.2f}s")
    check("순서 0.1→0.2→0.3", got == [0.1, 0.2, 0.3], str(got))
    check("is_speaking 끝나면 False", not spk.is_speaking())

    got.clear()
    spk.say("0.4 2.0")
    spk.say("0.5 2.0")
    time.sleep(0.3)
    t0 = time.monotonic()
    spk.stop()
    st = time.monotonic() - t0
    time.sleep(0.12)
    with lock:
        n_after = len(got)
    time.sleep(0.5)
    check("stop() 빠르게 반환", st < 0.05, f"{st * 1000:.1f} ms")
    check("stop() 후 이전 오디오 없음", len(got) == n_after and 0.5 not in got, str(got))
    check("stop() 후 is_speaking False", not spk.is_speaking())
    spk.say("0.6 0.2")
    spk.wait()
    check("stop() 후 say() 정상", got[-1:] == [0.6], str(got))
    # 효과음은 말과 동시에 섞인다
    spk.say("0.7 0.5")
    spk.chime("done")
    spk.wait()
    spk.chime("nonsense")  # 예외 없어야 함
    spk.shutdown()

    # 4) say 실제 합성 (소리 없이)
    sb = _SayBackend(pick_say_voice(cfg.say_voice), cfg.say_rate)
    sent = "오늘 서울 날씨는 맑고 기온은 이십 도예요."  # ~16음절
    t0 = time.monotonic()
    pcm = sb.synth(sent, lambda: True)
    st = time.monotonic() - t0
    dur = (pcm.size / SAMPLE_RATE) if pcm is not None else 0
    rms = float(np.sqrt(np.mean(pcm ** 2))) if pcm is not None else 0
    t1 = time.monotonic()
    sb.synth(sent, lambda: True)
    warm = time.monotonic() - t1
    check("say 합성", pcm is not None and 1.0 < dur < 8.0 and rms > 0.01,
          f"voice={sb.voice} synth={st:.2f}s (2회째 {warm:.2f}s) audio={dur:.2f}s rms={rms:.3f}")

    # say 백엔드 end-to-end (null 싱크, 즉시 소비)
    total = [0]
    spk = Speaker(say_cfg, sink=lambda b: total.__setitem__(0, total[0] + b.size), realtime=False)
    spk.say("첫 번째 문장입니다.")
    spk.say("두 번째 문장입니다.")
    spk.wait()
    check("say 백엔드 end-to-end", total[0] > SAMPLE_RATE, f"{total[0] / SAMPLE_RATE:.2f}s 출력")
    spk.shutdown()

    # 5) OpenAI 실패 -> say 폴백 (잘못된 키 + 접속 불가 주소)
    bad = dataclasses.replace(cfg, tts_backend="openai", openai_api_key="sk-invalid-test")
    total = [0]
    t0 = time.monotonic()
    spk = Speaker(bad, sink=lambda b: total.__setitem__(0, total[0] + b.size), realtime=False,
                  openai_base_url="http://127.0.0.1:9/v1", openai_timeout=2.0)
    check("OpenAI 백엔드 선택", spk.backend == "openai")
    spk.say("오픈에이아이 실패 시 대체 음성입니다.")
    spk.wait()
    check("OpenAI 실패 → say 폴백", total[0] > SAMPLE_RATE,
          f"{total[0] / SAMPLE_RATE:.2f}s 출력, {time.monotonic() - t0:.2f}s 소요")
    spk.shutdown()

    # 6) 실제 OutputStream 열고 닫기 (디지털 무음만)
    try:
        import sounddevice as sd

        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             device=cfg.output_device) as s:
            s.write(np.zeros(BLOCK, dtype=np.float32))
        check("sounddevice OutputStream 열기/무음/닫기", True)
    except Exception as e:
        check("sounddevice OutputStream 열기/무음/닫기", False, repr(e))

    print("전체 결과:", "통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_selftest("--audible" in sys.argv[1:]))
