"""타이머·알람·리마인더·수면 타이머. schedule.json 에 저장해 스피커를 껐다 켜도 남는다."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("speaker.schedule")

KINDS = ("timer", "alarm", "reminder", "sleep")
KIND_NAMES = {"timer": "타이머", "alarm": "알람", "reminder": "리마인더", "sleep": "수면 타이머"}
REPEATS = ("none", "daily", "weekdays", "weekends")
REPEAT_NAMES = {"none": "", "daily": "매일", "weekdays": "평일마다", "weekends": "주말마다"}

LATE_GRACE_S = 15 * 60  # 꺼져 있던 동안 지나간 알람·리마인더를 늦게라도 울릴 시간
MISSED_REMINDER_S = 12 * 3600  # 이보다 오래되지 않은 리마인더는 '놓친 리마인더'로 알린다


@dataclass
class Event:
    kind: str
    fires_at: float  # time.time() 기준
    label: str = ""
    message: str = ""
    repeat: str = "none"
    hhmm: str | None = None  # 반복 알람·리마인더의 시각 "07:30"
    music: str | None = None  # 알람: 틀 음악 (검색어, 또는 "liked" / "mix")
    briefing: bool = False  # 알람: 울린 뒤 날씨·뉴스 브리핑
    enabled: bool = True  # 꺼 둔 예약은 울리지 않는다 (반복 알람은 다음 회차로 넘어가며 남는다)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def when_text(self, now: float | None = None) -> str:
        """말로 하기 좋은 시각 표현: '3분 20초 뒤', '오늘 오후 7시 30분', '내일 오전 7시', '11월 20일 오전 9시'."""
        now = time.time() if now is None else now
        left = self.fires_at - now
        if self.kind in ("timer", "sleep") or left < 3600:
            return f"{spoken_duration(max(0, round(left)))} 뒤"
        at = dt.datetime.fromtimestamp(self.fires_at)
        today = dt.date.fromtimestamp(now)
        if at.date() == today:
            day = "오늘"
        elif at.date() == today + dt.timedelta(days=1):
            day = "내일"
        else:
            day = f"{at.month}월 {at.day}일"
        return f"{day} {spoken_clock(at.hour, at.minute)}"

    def describe(self, now: float | None = None) -> str:
        repeat = REPEAT_NAMES.get(self.repeat, "")
        name = KIND_NAMES.get(self.kind, self.kind)
        text = f"{self.label} {name}" if self.kind in ("timer", "alarm") and self.label else name
        if self.kind == "reminder":
            text = f"리마인더 '{self.message}'"
        extra = f" ({repeat})" if repeat else ""
        off = " [꺼 둠]" if not self.enabled else ""
        return f"{text}: {self.when_text(now)}{extra}{off}"


def spoken_duration(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}시간")
    if m:
        parts.append(f"{m}분")
    if s or not parts:
        parts.append(f"{s}초")
    return " ".join(parts)


def spoken_clock(hour: int, minute: int) -> str:
    ampm = "오전" if hour < 12 else "오후"
    h = hour % 12 or 12
    return f"{ampm} {h}시" + (f" {minute}분" if minute else "")


def next_occurrence(hhmm: str, repeat: str, after: float, date: str | None = None) -> float:
    """hhmm 시각의 다음 발생 시점. date 가 있으면 그날부터, repeat 에 맞는 요일만."""
    hour, minute = (int(x) for x in hhmm.split(":"))
    base = dt.datetime.fromtimestamp(after)
    day = dt.date.fromisoformat(date) if date else base.date()
    for _ in range(400):
        candidate = dt.datetime.combine(day, dt.time(hour, minute))
        weekday = candidate.weekday()
        ok_day = (
            repeat in ("none", "daily")
            or (repeat == "weekdays" and weekday < 5)
            or (repeat == "weekends" and weekday >= 5)
        )
        if ok_day and candidate.timestamp() > after:
            return candidate.timestamp()
        day += dt.timedelta(days=1)
    raise ValueError("다음 시각을 계산하지 못했습니다.")


class Scheduler:
    """예약을 저장하고, 때가 되면 on_fire(event) 를 부른다 (별도 스레드에서)."""

    def __init__(self, path: Path | None, on_fire: Callable[[Event], None]):
        self.path = path
        self._on_fire = on_fire
        self._events: list[Event] = []
        self._cond = threading.Condition()
        self._closed = False
        self.missed: list[Event] = []  # 꺼져 있는 동안 지나간 리마인더 (시작할 때 한 번 알린다)
        self._load()
        self._thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ 저장

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            events = [Event(**e) for e in raw.get("events", [])]
        except (OSError, ValueError, TypeError) as exc:
            log.error("예약 파일을 읽지 못했습니다 (%s). 빈 목록으로 시작합니다.", exc)
            return
        now = time.time()
        for e in events:
            if e.fires_at > now:
                self._events.append(e)
            elif e.repeat != "none" and e.hhmm:
                e.fires_at = next_occurrence(e.hhmm, e.repeat, now)  # 반복은 다음 회차로
                self._events.append(e)
            elif e.kind in ("alarm", "reminder") and now - e.fires_at <= LATE_GRACE_S:
                e.fires_at = now + 5  # 조금 늦었지만 지금 울린다
                self._events.append(e)
            elif e.kind == "reminder" and now - e.fires_at <= MISSED_REMINDER_S:
                self.missed.append(e)
            else:
                log.info("꺼져 있는 동안 지나간 예약을 버립니다: %s", e.describe(now))
        self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"events": [asdict(e) for e in self._events]}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ 공개 API

    def add(self, event: Event) -> Event:
        if event.kind not in KINDS:
            raise ValueError(f"알 수 없는 종류: {event.kind}")
        with self._cond:
            if event.kind == "sleep":  # 수면 타이머는 하나만
                self._events = [e for e in self._events if e.kind != "sleep"]
            self._events.append(event)
            self._save()
            self._cond.notify_all()
        log.info("예약 추가: %s", event.describe())
        return event

    def cancel(self, kind: str | None = None, label: str | None = None) -> list[Event]:
        """kind 와 label(이름이나 리마인더 내용의 일부)에 맞는 예약을 지운다. 둘 다 없으면 전부."""
        key = (label or "").replace(" ", "")

        def match(e: Event) -> bool:
            if kind and e.kind != kind:
                return False
            if key:
                text = f"{e.label}{e.message}".replace(" ", "")
                return key in text or (text and text in key)
            return True

        with self._cond:
            hit = [e for e in self._events if match(e)]
            self._events = [e for e in self._events if e not in hit]
            if hit:
                self._save()
            self._cond.notify_all()
        for e in hit:
            log.info("예약 취소: %s", e.describe())
        return hit

    def remove(self, event_id: str) -> Event | None:
        with self._cond:
            hit = next((e for e in self._events if e.id == event_id), None)
            if hit:
                self._events.remove(hit)
                self._save()
            self._cond.notify_all()
        if hit:
            log.info("예약 삭제: %s", hit.describe())
        return hit

    def set_enabled(self, event_id: str, enabled: bool) -> Event | None:
        with self._cond:
            hit = next((e for e in self._events if e.id == event_id), None)
            if hit:
                hit.enabled = enabled
                self._save()
        if hit:
            log.info("예약 %s: %s", "켬" if enabled else "끔", hit.describe())
        return hit

    def get(self, event_id: str) -> Event | None:
        with self._cond:
            return next((e for e in self._events if e.id == event_id), None)

    def upcoming(self, kind: str | None = None) -> list[Event]:
        with self._cond:
            events = [e for e in self._events if not kind or e.kind == kind]
        return sorted(events, key=lambda e: e.fires_at)

    def shutdown(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join(2)

    # ------------------------------------------------------------ 실행

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._closed:
                    return
                now = time.time()
                due = [e for e in self._events if e.fires_at <= now]
                for e in due:
                    if e.repeat != "none" and e.hhmm:
                        e.fires_at = next_occurrence(e.hhmm, e.repeat, now + 1)
                    else:
                        self._events.remove(e)
                if due:
                    self._save()
                else:
                    nxt = min((e.fires_at for e in self._events), default=now + 60)
                    # 맥이 잠들었다 깨는 경우를 대비해 길게 자지 않는다
                    self._cond.wait(timeout=max(0.05, min(nxt - now, 30.0)))
                    continue
            for e in sorted(due, key=lambda e: e.fires_at):
                if not e.enabled:
                    log.info("꺼 둔 예약이라 울리지 않습니다: %s", e.describe())
                    continue
                log.info("예약 시각 도래: %s", e.describe())
                try:
                    self._on_fire(e)
                except Exception:
                    log.exception("예약 처리 중 오류")
