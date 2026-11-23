"""마이크 입력: 호출어 감지(openWakeWord) + 발화 녹음(Silero VAD) + 음성 인식(mlx-whisper).

Listener
    마이크 콜백은 프레임을 큐에 넣기만 하고, 워커 스레드 하나가 소비한다.
    - 호출어 모드: 1280샘플(80ms) 단위로 openWakeWord 에 넣고 점수가 임계값을 넘으면 wake_event 설정.
    - 녹음 모드: record_utterance() 가 요청하면 프레임을 VAD/녹음기로 보낸다.
    feed() 는 마이크 없이 같은 경로(큐 -> 워커)로 샘플을 밀어 넣는 테스트용 입구.

Transcriber
    mlx-whisper 는 전용 스레드 하나에서만 실행된다(어느 스레드에서 호출해도 안전).
"""

from __future__ import annotations

import collections
import difflib
import glob
import logging
import os
import queue
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from speaker.config import Config

log = logging.getLogger("speaker.audio_in")

SR = 16000
WAKE_CHUNK = 1280  # 80 ms
VAD_FRAME = 512  # 32 ms (Silero 16 kHz 권장 크기)
REFRACTORY_S = 2.0  # 감지 후 재감지 금지 시간
PREROLL_S = 0.3
RING_S = 0.5  # 녹음 시작 전에 보관하는 오디오


# ---------------------------------------------------------------------------
# openWakeWord 도우미
# ---------------------------------------------------------------------------

def _resolve_wake_model(name: str) -> str:
    """호출어 이름(hey_jarvis 등) 또는 .onnx 경로 -> 모델 파일 경로."""
    if name.endswith((".onnx", ".tflite")) and os.path.exists(name):
        return name
    import openwakeword

    entry = openwakeword.models.get(name)
    if entry and os.path.exists(entry["model_path"]):
        return entry["model_path"]
    res_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    found = sorted(glob.glob(os.path.join(res_dir, f"{name}_v*.onnx")))
    if found:
        return found[-1]
    # 최신 openwakeword 는 모델을 따로 내려받는다 (0.4.0 은 패키지에 포함)
    dl = getattr(__import__("openwakeword.utils", fromlist=["x"]), "download_models", None)
    if dl is not None:
        log.info("openWakeWord 모델 다운로드 중...")
        dl()
        found = sorted(glob.glob(os.path.join(res_dir, f"{name}_v*.onnx")))
        if found:
            return found[-1]
    raise FileNotFoundError(f"호출어 모델을 찾을 수 없습니다: {name}")


class _WakeDetector:
    def __init__(self, cfg: Config):
        from openwakeword.model import Model

        path = _resolve_wake_model(cfg.wake_word)
        self.model = Model(wakeword_model_paths=[path], inference_framework="onnx") \
            if _accepts_framework(Model) else Model(wakeword_model_paths=[path])
        self.key = next(iter(self.model.models.keys()))
        pp = self.model.preprocessor
        self._init_features = pp.feature_buffer.copy()
        self.threshold = cfg.wake_threshold
        self.last_score = 0.0
        self.max_score = 0.0  # 디버그/테스트용 최고 점수
        log.info("호출어 모델 로드: %s (임계값 %.2f)", os.path.basename(path), self.threshold)

    def reset(self) -> None:
        pp = self.model.preprocessor
        pp.raw_data_buffer.clear()
        pp.melspectrogram_buffer = np.ones((76, 32))
        pp.accumulated_samples = 0
        pp.feature_buffer = self._init_features.copy()
        self.model.reset()
        self.last_score = 0.0

    def score(self, chunk: np.ndarray) -> float:
        pred = self.model.predict(chunk)
        s = float(pred.get(self.key, max(pred.values()) if pred else 0.0))
        self.last_score = s
        self.max_score = max(self.max_score, s)
        return s


def _accepts_framework(model_cls) -> bool:
    import inspect

    try:
        return "inference_framework" in inspect.signature(model_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

class _Vad:
    """Silero VAD (openwakeword 동봉). 실패하면 적응형 에너지 임계값으로 대체."""

    def __init__(self):
        self.silero = None
        try:
            from openwakeword.vad import VAD

            self.silero = VAD()
        except Exception as e:  # noqa: BLE001
            log.warning("Silero VAD 로드 실패, 에너지 기반 VAD 사용: %s", e)
        self.noise_rms = 200.0

    def reset(self) -> None:
        if self.silero is not None:
            self.silero.reset_states()

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        if rms < 30.0:  # 사실상 무음
            if self.silero is not None:
                self.silero.predict(frame, frame_size=len(frame))  # 상태는 계속 갱신
            return 0.0
        if self.silero is not None:
            return float(self.silero.predict(frame, frame_size=len(frame)))
        # 에너지 대체: 잡음 바닥을 천천히 추적
        ratio = rms / max(self.noise_rms, 1.0)
        if ratio < 2.0:
            self.noise_rms = 0.95 * self.noise_rms + 0.05 * rms
        return float(np.clip((ratio - 1.5) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 한국어 호출 문구 ("야 듣고 있어?") — 말소리 구간을 잘라 음성 인식으로 확인
# ---------------------------------------------------------------------------

PHRASE_MIN_S = 0.4  # 이보다 짧은 말소리는 문구일 수 없다
PHRASE_MAX_S = 6.0  # 문구 + 바로 이어 말한 명령까지. 이보다 긴 건 대화로 보고 무시
PHRASE_END_SILENCE_S = 0.5
PHRASE_MATCH = 0.75


def match_phrase(text: str, phrase: str) -> tuple[bool, str]:
    """text 가 호출 문구로 시작하면 (True, 문구 뒤에 이어 말한 명령). 음성 인식 오차를 감안해 느슨하게 맞춘다."""
    def norm(t: str) -> str:
        return "".join(re.findall(r"\w+", t))

    target, said = norm(phrase), norm(text)
    if not target or not said:
        return False, ""
    best, best_len = 0.0, 0
    for n in range(max(2, len(target) - 2), len(target) + 3):
        r = difflib.SequenceMatcher(None, target, said[:n]).ratio()
        if r > best:
            best, best_len = r, n
    if best < PHRASE_MATCH:
        return False, ""
    # 정규화된 글자 수(best_len)만큼 원문에서 건너뛰고 남은 부분이 명령
    count, pos = 0, len(text)
    for i, ch in enumerate(text):
        if re.match(r"\w", ch):
            count += 1
            if count == best_len:
                pos = i + 1
                break
    rest = text[pos:].strip(" \t,.?!~…")
    return True, rest if len(norm(rest)) >= 2 else ""


class _PhraseSpotter:
    """대기 중 들리는 짧은 말소리를 잘라 두었다가 별도 스레드에서 음성 인식으로 문구인지 확인한다."""

    def __init__(self, phrases: list[str], transcriber: "Transcriber", on_match, gate=None):
        self.phrases = list(phrases)
        self.transcriber = transcriber
        self.on_match = on_match  # (명령 텍스트) -> None
        self.gate = gate or (lambda: True)  # False 를 돌려주는 동안은 듣지 않는다 (예: 스피커가 말하는 중)
        self.vad = _Vad()
        self.pending = np.zeros(0, np.int16)
        self.pre: collections.deque = collections.deque(maxlen=6)  # 말 시작 직전 ~0.2초
        self.seg: list[np.ndarray] = []
        self.silence = 0
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)  # 밀리면 버린다 (음악 가사 등)
        threading.Thread(target=self._run, name="phrase-spotter", daemon=True).start()

    def reset(self) -> None:
        self.seg, self.silence, self.pending = [], 0, np.zeros(0, np.int16)
        self.pre.clear()
        self.vad.reset()

    def feed(self, frames: np.ndarray) -> None:
        buf = np.concatenate([self.pending, frames])
        i = 0
        active = self.gate()
        while len(buf) - i >= VAD_FRAME:
            frame = buf[i:i + VAD_FRAME]
            i += VAD_FRAME
            speech = self.vad.prob(frame) >= 0.5 if active else False
            if self.seg:
                self.seg.append(frame)
                self.silence = 0 if speech else self.silence + 1
                n = len(self.seg) * VAD_FRAME
                if n > PHRASE_MAX_S * SR:
                    self.seg = []  # 너무 길다: 대화나 음악으로 보고 버린다
                elif self.silence * VAD_FRAME >= PHRASE_END_SILENCE_S * SR:
                    voiced = n - self.silence * VAD_FRAME
                    if voiced >= PHRASE_MIN_S * SR:
                        audio = np.concatenate(self.seg).astype(np.float32) / 32768.0
                        try:
                            self.q.put_nowait(audio)
                        except queue.Full:
                            log.debug("호출 문구 확인이 밀려 말소리 하나를 건너뜀")
                    self.seg = []
            elif speech:
                self.seg = list(self.pre) + [frame]
                self.silence = 0
            else:
                self.pre.append(frame)
        self.pending = buf[i:]

    def _run(self) -> None:
        while True:
            audio = self.q.get()
            try:
                text = self.transcriber.transcribe(audio)
            except Exception:  # noqa: BLE001
                log.exception("호출 문구 인식 실패")
                continue
            if not text:
                continue
            ok, command = False, ""
            for phrase in list(self.phrases):
                ok, command = match_phrase(text, phrase)
                if ok:
                    break
            log.debug("호출 문구 후보: %r → %s", text, ok)
            if ok:
                log.info("호출 문구 감지: %r%s", text, f" (이어서 한 말: {command!r})" if command else "")
                self.on_match(command)


@dataclass
class _RecordJob:
    start_timeout: float
    max_seconds: float
    end_silence: float
    buf: list = field(default_factory=list)  # int16 조각
    n: int = 0  # 누적 샘플 수
    pre_n: int = 0  # 앞에 붙인 링버퍼 샘플 수
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))
    flags: collections.deque = field(default_factory=lambda: collections.deque(maxlen=5))
    onset: int | None = None  # 발화 시작 샘플 위치
    last_speech_end: int = 0
    silence_run: int = 0
    result: np.ndarray | None = None
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------

class Listener:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.wake_event = threading.Event()
        # WAKE_WORD=off 면 영어 호출어 모델을 쓰지 않는다 (한국어 호출 문구만)
        self._wake = _WakeDetector(cfg) if cfg.wake_word.lower() not in ("off", "none", "") else None
        self._phrase: _PhraseSpotter | None = None
        self._phrase_command: str | None = None
        self.music_playing = False  # 메인 루프가 1초마다 갱신. 음악 중에는 호출어 기준을 낮춘다
        self._near_miss_logged = 0.0
        self.near_wake_at = 0.0  # 음악 중 호출어 점수가 조금 오른 마지막 시각 (메인 루프가 음악을 잠깐 줄인다)
        self._wake_lock = threading.Lock()
        self._vad = _Vad()
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1000)  # ~80초
        self._wake_pending = np.zeros(0, np.int16)
        self._ring: collections.deque = collections.deque()
        self._ring_n = 0
        self._samples_seen = 0
        self._refractory_until = 0  # 샘플 위치 기준
        self._job: _RecordJob | None = None
        self._job_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._stream = None
        self._dropped = 0

    # --- 수명 주기 --------------------------------------------------------
    def start(self) -> None:
        import sounddevice as sd

        self._ensure_worker()
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="int16", blocksize=WAKE_CHUNK,
            device=self.cfg.input_device, callback=self._callback,
        )
        self._stream.start()
        log.info("마이크 시작 (장치: %s)", self.cfg.input_device or "기본값")

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
            log.info("마이크 중지")
        self._running.clear()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        job = self._job
        if job is not None:
            job.cancelled = True
            job.done.set()

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._running.set()
        self._worker = threading.Thread(target=self._run, name="listener-worker", daemon=True)
        self._worker.start()

    def _callback(self, indata, frames, time_info, status) -> None:  # 오디오 스레드
        if status:
            log.debug("입력 상태: %s", status)
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                log.warning("오디오 큐가 가득 차서 프레임을 버렸습니다 (%d)", self._dropped)

    # --- 공개 API --------------------------------------------------------
    def feed(self, frames: np.ndarray) -> None:
        """테스트용: int16 16 kHz 모노 샘플을 마이크와 같은 경로로 처리. 처리 완료까지 블록."""
        frames = np.asarray(frames)
        if frames.dtype != np.int16:
            frames = np.clip(frames, -32768, 32767).astype(np.int16)
        frames = frames.reshape(-1)
        self._ensure_worker()
        for i in range(0, len(frames), WAKE_CHUNK):
            self._q.put(frames[i:i + WAKE_CHUNK].copy())
        self._q.join()

    def enable_phrase(self, phrases: list[str] | str, transcriber: "Transcriber", gate=None) -> None:
        """호출 문구(들)를 켠다. gate() 가 False 인 동안(스피커가 말하는 중 등)은 문구를 찾지 않는다."""
        phrases = [phrases] if isinstance(phrases, str) else list(phrases)
        self._phrase = _PhraseSpotter(phrases, transcriber, self._on_phrase, gate)
        log.info("호출 문구 사용: %s", ", ".join(repr(x) for x in phrases))

    def set_phrases(self, phrases: list[str]) -> None:
        """호출 문구 목록을 바로 바꾼다 (다시 켤 필요 없음)."""
        if self._phrase is not None:
            self._phrase.phrases = list(phrases)
            log.info("호출 문구 변경: %s", ", ".join(repr(x) for x in phrases) or "(없음)")

    @property
    def phrases(self) -> list[str]:
        return list(self._phrase.phrases) if self._phrase is not None else []

    def _on_phrase(self, command: str) -> None:
        if self.wake_event.is_set():
            return
        self._phrase_command = command or None
        self.wake_event.set()

    def take_phrase_command(self) -> str | None:
        """호출 문구 뒤에 바로 이어 말한 명령 ("야 듣고 있어? 음악 틀어줘" → "음악 틀어줘"). 없으면 None."""
        command, self._phrase_command = self._phrase_command, None
        return command

    def clear_wake(self) -> None:
        self.wake_event.clear()
        with self._wake_lock:
            if self._wake is not None:
                self._wake.reset()
            self._wake_pending = np.zeros(0, np.int16)
        if self._phrase is not None:
            self._phrase.reset()

    def record_utterance(self, start_timeout: float = 6.0, max_seconds: float = 20.0,
                         end_silence: float = 0.9) -> np.ndarray | None:
        job = _RecordJob(start_timeout, max_seconds, end_silence)
        self._ensure_worker()
        with self._job_lock:
            self._vad.reset()
            if self._ring:
                pre = np.concatenate(list(self._ring))
                job.buf.append(pre)
                job.n = job.pre_n = len(pre)
            self._job = job
        log.info("녹음 시작 (말씀하세요)")
        # 시간 판정은 샘플 수 기준; 오디오가 안 들어올 때를 대비한 벽시계 안전장치
        if not job.done.wait(timeout=start_timeout + max_seconds + 5.0):
            log.warning("오디오 입력이 없어 녹음을 중단합니다")
            with self._job_lock:
                if not job.done.is_set():
                    job.result = self._finish_audio(job) if job.onset is not None else None
                    job.done.set()
                self._job = None
        with self._job_lock:
            if self._job is job:
                self._job = None
        if job.cancelled:
            return None
        return job.result

    @property
    def last_wake_score(self) -> float:
        return self._wake.last_score if self._wake is not None else 0.0

    # --- 워커 -------------------------------------------------------------
    def _run(self) -> None:
        while self._running.is_set():
            try:
                frames = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process(frames)
            except Exception:  # noqa: BLE001
                log.exception("오디오 처리 중 오류")
            finally:
                self._q.task_done()

    def _process(self, frames: np.ndarray) -> None:
        with self._job_lock:
            job = self._job
            if job is not None and not job.done.is_set():
                self._record(job, frames)
                if job.done.is_set():
                    self._job = None
                    self._after_record()
                return
        self._push_ring(frames)
        self._wake_step(frames)
        if self._phrase is not None and self._phrase.phrases and not self.wake_event.is_set():
            self._phrase.feed(frames)

    def _push_ring(self, frames: np.ndarray) -> None:
        self._ring.append(frames)
        self._ring_n += len(frames)
        while self._ring and self._ring_n - len(self._ring[0]) >= int(RING_S * SR):
            self._ring_n -= len(self._ring.popleft())

    def _wake_step(self, frames: np.ndarray) -> None:
        if self._wake is None:
            return
        with self._wake_lock:
            buf = np.concatenate([self._wake_pending, frames])
            i = 0
            while len(buf) - i >= WAKE_CHUNK:
                chunk = buf[i:i + WAKE_CHUNK]
                i += WAKE_CHUNK
                self._samples_seen += WAKE_CHUNK
                score = self._wake.score(chunk)
                # 음악이 나오는 중에는 음악이 목소리를 덮으므로 기준을 낮춘다
                threshold = self.cfg.wake_threshold_music if self.music_playing else self.cfg.wake_threshold
                near = self.cfg.wake_duck_score if self.cfg.wake_duck_score > 0 else threshold * 0.5
                if self.music_playing and near <= score < threshold:
                    now = time.monotonic()
                    if self.cfg.wake_duck_score > 0:
                        self.near_wake_at = now
                    if now - self._near_miss_logged > 2.0:  # 조정용 기록 (2초에 한 번)
                        self._near_miss_logged = now
                        log.info("음악 중 호출어 비슷한 소리 (점수 %.2f, 기준 %.2f)", score, threshold)
                if (score >= threshold
                        and self._samples_seen >= self._refractory_until
                        and not self.wake_event.is_set()):
                    self._refractory_until = self._samples_seen + int(REFRACTORY_S * SR)
                    log.info("호출어 감지 (점수 %.2f%s)", score, ", 음악 재생 중" if self.music_playing else "")
                    self.wake_event.set()
            self._wake_pending = buf[i:]

    def _after_record(self) -> None:
        # 녹음 중 끊긴 호출어 스트림 초기화 (재감지 방지)
        with self._wake_lock:
            if self._wake is not None:
                self._wake.reset()
            self._wake_pending = np.zeros(0, np.int16)
            self._refractory_until = self._samples_seen + int(REFRACTORY_S * SR)
        self._ring.clear()
        if self._phrase is not None:
            self._phrase.reset()
        self._ring_n = 0

    def _record(self, job: _RecordJob, frames: np.ndarray) -> None:
        job.buf.append(frames)
        pend = np.concatenate([job.pending, frames])
        base = job.n - len(job.pending)  # pend[0] 의 샘플 위치
        job.n += len(frames)
        i = 0
        while len(pend) - i >= VAD_FRAME:
            fr = pend[i:i + VAD_FRAME]
            pos = base + i
            i += VAD_FRAME
            p = self._vad.prob(fr)
            speech = p >= 0.5
            job.flags.append(speech)
            if job.onset is None:
                if sum(job.flags) >= 3:  # 최근 5프레임 중 3프레임 이상 말소리
                    first = pos - (len(job.flags) - 1) * VAD_FRAME
                    job.onset = max(0, first)
                    job.last_speech_end = pos + VAD_FRAME
                    job.silence_run = 0
                    log.debug("발화 시작 감지 (%.2fs)", job.onset / SR)
                elif pos + VAD_FRAME >= int(job.start_timeout * SR) + job.pre_n:
                    log.info("발화가 없어 녹음을 종료합니다")
                    job.result = None
                    job.done.set()
                    return
            else:
                if speech:
                    job.last_speech_end = pos + VAD_FRAME
                    job.silence_run = 0
                elif p < 0.35:
                    job.silence_run += VAD_FRAME
                else:
                    job.silence_run = 0  # 애매한 구간은 침묵으로 세지 않음
                if job.silence_run >= int(job.end_silence * SR):
                    self._complete(job, "침묵")
                    return
                if pos + VAD_FRAME - job.onset >= int(job.max_seconds * SR):
                    self._complete(job, "최대 길이")
                    return
        job.pending = pend[i:]

    def _complete(self, job: _RecordJob, why: str) -> None:
        job.result = self._finish_audio(job)
        log.info("녹음 종료 (%s, %.1f초)", why, len(job.result) / SR)
        job.done.set()

    @staticmethod
    def _finish_audio(job: _RecordJob) -> np.ndarray:
        audio = np.concatenate(job.buf)
        start = max(0, (job.onset or 0) - int(PREROLL_S * SR))
        end = min(len(audio), job.last_speech_end + int(0.3 * SR))
        return audio[start:end].astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

# 무음/잡음에서 Whisper 가 흔히 지어내는 문구
_HALLUCINATION_PATTERNS = [
    r"시청\s*해\s*주셔서", r"시청\s*감사", r"구독", r"좋아요", r"알림\s*설정",
    r"(MBC|KBS|SBS|YTN|JTBC|EBS)\s*뉴스", r"뉴스\s*[가-힣]{2,4}\s*입니다\.?$",
    r"[가-힣]{2,4}\s*기자\s*입니다", r"다음\s*(영상|시간)에", r"자막\s*(제공|by|제작)",
    r"영상\s*편집", r"한글\s*자막", r"Thank you for watching", r"Subtitles by",
    r"^\W*(you|Thank you\.?)\W*$",
]
_HALLU_RE = re.compile("|".join(_HALLUCINATION_PATTERNS), re.IGNORECASE)
# 짧은 단독 문구: 확신이 낮을 때만 버림
_WEAK_PHRASES = {"감사합니다", "고맙습니다", "네", "아", "음", "어", "예", "안녕하세요", "bye"}


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._q: queue.Queue = queue.Queue()
        self._vad = _Vad()
        self._vad_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="whisper", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, fut = self._q.get()
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)

    def _call(self, fn):
        fut: Future = Future()
        self._q.put((fn, fut))
        return fut.result()

    def warmup(self) -> None:
        t = time.perf_counter()
        log.info("음성 인식 모델 준비 중: %s", self.cfg.whisper_model)
        self._call(lambda: self._raw(np.zeros(SR, np.float32)))
        log.info("음성 인식 모델 준비 완료 (%.1f초)", time.perf_counter() - t)

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < int(0.3 * SR):
            return ""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.002:  # 약 -54 dBFS: 사실상 무음
            log.debug("음량이 너무 작아 인식을 건너뜀 (rms=%.4f)", rms)
            return ""
        # turbo 모델의 no_speech_prob 는 믿기 어려워서 VAD 로 먼저 거른다
        speech = self._speech_seconds(audio)
        if speech < 0.15:  # "네" 한 마디가 0.2초쯤이라 이보다 높이면 짧은 대답이 버려진다
            log.info("말소리가 감지되지 않아 인식을 건너뜀 (%.2f초)", speech)
            return ""
        result = self._call(lambda: self._raw(audio))
        text = self._filter(result)
        log.info("인식 결과: %r", text)
        return text

    def _speech_seconds(self, audio: np.ndarray) -> float:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        n = 0
        with self._vad_lock:
            self._vad.reset()
            for i in range(0, len(pcm) - VAD_FRAME + 1, VAD_FRAME):
                if self._vad.prob(pcm[i:i + VAD_FRAME]) >= 0.5:
                    n += 1
        return n * VAD_FRAME / SR

    def _raw(self, audio: np.ndarray) -> dict:
        import mlx_whisper

        from speaker.mlx_lock import MLX_LOCK

        with MLX_LOCK:  # 음성 합성(Qwen)과 GPU 를 동시에 쓰지 않는다
            return mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.cfg.whisper_model,
                language=self.cfg.language,
                condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4),
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                verbose=None,
            )

    @staticmethod
    def _filter(result: dict) -> str:
        kept = []
        for seg in result.get("segments", []):
            t = seg.get("text", "").strip()
            if not t:
                continue
            nsp = seg.get("no_speech_prob", 0.0)
            lp = seg.get("avg_logprob", 0.0)
            cr = seg.get("compression_ratio", 1.0)
            log.debug("세그먼트 %r nsp=%.2f lp=%.2f cr=%.2f", t, nsp, lp, cr)
            if nsp > 0.6 and lp < -0.5:
                continue
            if nsp > 0.8 or lp < -1.0 or cr > 2.4:
                continue
            if _HALLU_RE.search(t):
                log.info("환각으로 보이는 문구 제거: %r", t)
                continue
            core = re.sub(r"[\s\.\,\!\?…~]+", "", t)
            if core in _WEAK_PHRASES and (nsp > 0.3 or lp < -0.6):
                continue
            kept.append(t)
        text = " ".join(kept).strip()
        if not re.search(r"\w", text):
            return ""
        return text


# ---------------------------------------------------------------------------
# 자체 테스트 (마이크 불필요): python -m speaker.audio_in
# ---------------------------------------------------------------------------

def _say_wav(text: str, voice: str, path: str) -> np.ndarray:
    import subprocess
    import wave

    subprocess.run(["say", "-v", voice, "-o", path, "--data-format=LEI16@16000", text], check=True)
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def _self_test() -> None:
    import difflib
    import sys

    from speaker.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    import tempfile

    tmp = os.environ.get("AUDIO_IN_TMP") or os.path.join(tempfile.gettempdir(), "mac-home-speaker-audio-in-test")
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(0)

    def sil(sec: float, level: float = 0.0) -> np.ndarray:
        n = int(sec * SR)
        if level <= 0:
            return np.zeros(n, np.int16)
        return (rng.normal(0, level * 32767, n)).clip(-32768, 32767).astype(np.int16)

    ko_text = "오늘 서울 날씨 어때?"
    ko = _say_wav(ko_text, "Yuna", os.path.join(tmp, "ko.wav"))
    print(f"한국어 문장 길이: {len(ko) / SR:.2f}s")

    failures = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # (1) 호출어
    lis = Listener(cfg)
    voices = ["Samantha", "Daniel", "Karen", "Moira", "Fred"]
    detected_by = []
    for v in voices:
        try:
            hj = _say_wav("hey jarvis", v, os.path.join(tmp, f"hj_{v}.wav"))
        except Exception as e:  # noqa: BLE001
            print(f"  voice {v} 사용 불가: {e}")
            continue
        lis.clear_wake()
        lis._wake.max_score = 0.0
        lis.feed(sil(1.5, 0.001))
        lis.feed(hj)
        lis.feed(sil(1.0, 0.001))
        print(f"  voice={v:9s} 최대 점수={lis._wake.max_score:.3f} 감지={lis.wake_event.is_set()}")
        if lis.wake_event.is_set():
            detected_by.append(v)
    check(bool(detected_by), f"'hey jarvis' 호출어 감지 (감지된 목소리: {detected_by})")

    lis.clear_wake()
    lis._wake.max_score = 0.0
    lis.feed(sil(3.0, 0.001))
    check(not lis.wake_event.is_set(), f"무음에서 미감지 (최대 {lis._wake.max_score:.3f})")
    lis.clear_wake()
    lis._wake.max_score = 0.0
    lis.feed(sil(1.0, 0.001))
    lis.feed(ko)
    lis.feed(sil(1.0, 0.001))
    check(not lis.wake_event.is_set(), f"한국어 문장에서 미감지 (최대 {lis._wake.max_score:.3f})")

    # 감지 후 2초 내 재감지 금지
    if detected_by:
        hj = _say_wav("hey jarvis", detected_by[0], os.path.join(tmp, "hj_re.wav"))
        lis.clear_wake()
        lis.feed(sil(1.0, 0.001))
        lis.feed(hj)
        lis.feed(sil(0.2, 0.001))
        first = lis.wake_event.is_set()
        lis.wake_event.clear()  # 모델 상태는 그대로 두고 이벤트만 해제
        lis.feed(sil(1.0, 0.001))
        check(first and not lis.wake_event.is_set(), "재감지 방지(불응기)")

    # 추론 시간
    lis.clear_wake()
    chunks = [sil(0.08, 0.01) for _ in range(100)]
    t = time.perf_counter()
    for c in chunks:
        lis._wake.score(c)
    wake_ms = (time.perf_counter() - t) / len(chunks) * 1000
    print(f"호출어 추론 시간: {wake_ms:.2f} ms / 80ms 청크")

    # (2) 녹음
    def run_record(feed_parts, **kw):
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("a", lis.record_utterance(**kw)))
        th.start()
        while lis._job is None and th.is_alive():
            time.sleep(0.005)
        for p in feed_parts:
            if not th.is_alive():
                break
            lis.feed(p)
        th.join(timeout=30)
        return out.get("a")

    lis.clear_wake()
    rec = run_record([sil(1.0, 0.003), ko, sil(1.5, 0.003)])
    ok = rec is not None and abs(len(rec) / SR - len(ko) / SR) < 1.0
    check(ok, f"녹음 길이 {(len(rec) / SR if rec is not None else -1):.2f}s ≈ 문장 {len(ko) / SR:.2f}s")

    # 잡음 환경 (약 -30 dBFS 백색잡음)
    rec_noisy = run_record([sil(1.0, 0.03), (ko + sil(len(ko) / SR, 0.03)[:len(ko)]).astype(np.int16),
                            sil(1.5, 0.03)])
    ok = rec_noisy is not None and abs(len(rec_noisy) / SR - len(ko) / SR) < 1.0
    check(ok, f"잡음 속 녹음 길이 {(len(rec_noisy) / SR if rec_noisy is not None else -1):.2f}s")

    none = run_record([sil(2.0, 0.003)], start_timeout=1.0)
    check(none is None, "무음만 있을 때 None 반환")

    # 녹음 후 호출어 모드 복귀 확인
    lis.clear_wake()
    if detected_by:
        lis.feed(sil(2.5, 0.001))
        lis.feed(hj)
        lis.feed(sil(1.0, 0.001))
        check(lis.wake_event.is_set(), "녹음 후 호출어 모드 복귀")

    # (3) 음성 인식
    tr = Transcriber(cfg)
    t = time.perf_counter()
    tr.warmup()
    print(f"warmup: {time.perf_counter() - t:.1f}s")
    if rec is not None:
        t = time.perf_counter()
        text = tr.transcribe(rec)
        dt = time.perf_counter() - t
        norm = lambda s: re.sub(r"[\s\.\,\!\?]", "", s)  # noqa: E731
        ratio = difflib.SequenceMatcher(None, norm(text), norm(ko_text)).ratio()
        print(f"인식: {text!r}  유사도={ratio:.2f}  시간={dt:.2f}s (오디오 {len(rec) / SR:.2f}s)")
        check(ratio >= 0.7, "한국어 인식 결과가 원문과 유사")
    if rec_noisy is not None:
        text_n = tr.transcribe(rec_noisy)
        print(f"잡음 속 인식: {text_n!r}")

    s0 = tr.transcribe(np.zeros(3 * SR, np.float32))
    check(s0 == "", f"무음 3초 -> {s0!r}")
    for lvl in (0.003, 0.01, 0.03):
        sn = tr.transcribe(rng.normal(0, lvl, 3 * SR).astype(np.float32))
        check(sn == "", f"잡음(std={lvl}) 3초 -> {sn!r}")
    # 다른 스레드에서 호출해도 동작하는지
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("t", tr.transcribe(ko.astype(np.float32) / 32768)))
    th.start()
    th.join()
    print(f"다른 스레드에서 인식: {out.get('t')!r}")

    # 마이크 start/stop
    try:
        lis.start()
        time.sleep(1.0)
        lis.stop()
        print("마이크 start()/stop() 정상")
    except Exception as e:  # noqa: BLE001
        print(f"마이크 start() 실패: {type(e).__name__}: {e}")
        lis.stop()

    print("\n요약: " + ("모두 통과" if not failures else f"실패 {len(failures)}건: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _self_test()
