"""맥 캘린더·메일 앱에 추가한 계정(학교 Exchange 등)의 일정과 안 읽은 메일을 읽는다.

- 일정: EventKit(빠르고 반복 일정도 정확). 스피커를 실행하는 앱에 캘린더 '전체 접근' 권한이 있어야 한다.
  (캘린더 앱 스크립팅은 계정이 크면 2분 넘게 걸려서 쓰지 않는다.)
- 메일: 메일 앱 스크립팅 (osascript).
둘 다 몇 초~수십 초 걸리므로 뒤에서 미리 읽어 두고(캐시), 묻는 순간에는 저장된 결과를 쓴다.
"""

from __future__ import annotations

import datetime as dt
import logging
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

log = logging.getLogger("speaker.pim")

REFRESH_S = 5 * 60  # 뒤에서 다시 읽는 주기 (메일 앱 읽기는 수십 초 걸릴 수 있어 미리 읽어 둔다)
_PROJECT_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
FIELD, RECORD = "\x1f", "\x1e"


@dataclass
class CalEvent:
    start: dt.datetime
    end: dt.datetime
    title: str
    location: str
    all_day: bool
    calendar: str


@dataclass
class MailItem:
    received: dt.datetime | None
    sender: str
    subject: str
    preview: str
    bulk: bool = False  # 홍보·뉴스레터·자동 알림처럼 여러 사람에게 한꺼번에 보낸 메일


# 자동 발송 주소에 흔한 이름 (헤더로 못 거른 것을 한 번 더 거른다)
_BULK_SENDER = re.compile(
    r"no-?reply|do-?not-?reply|newsletter|marketing|promo|notification|notify|mailer|bounce|digest|"
    r"updates?@|news@|info@|hello@|team@|support@|alerts?@",
    re.I,
)


# ---------------------------------------------------------------- 일정


_access_asked = False


def _eventkit_store():
    """권한이 있으면 EKEventStore, 없으면 None.

    아직 한 번도 묻지 않은 상태면, 처음 한 번 권한 창을 띄운다 (터미널 앱에서 실행했을 때 뜬다).
    """
    global _access_asked
    try:
        import EventKit
    except ImportError:
        return None
    status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)
    if status == 0 and not _access_asked:  # 0=아직 묻지 않음
        _access_asked = True
        done = threading.Event()
        store = EventKit.EKEventStore.alloc().init()
        log.info("캘린더 접근 허용 창을 띄웁니다. 맥 화면에서 '허용'을 눌러 주세요.")
        if hasattr(store, "requestFullAccessToEventsWithCompletion_"):
            store.requestFullAccessToEventsWithCompletion_(lambda granted, error: done.set())
        else:
            store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeEvent, lambda granted, error: done.set())
        done.wait(300)
        status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)
    if status != 3:  # 3=전체 접근만 일정 내용을 읽을 수 있다 (4=쓰기 전용)
        return None
    return EventKit.EKEventStore.alloc().init()


def _events_eventkit(store, start: dt.datetime, end: dt.datetime) -> list[CalEvent]:
    import EventKit
    from Foundation import NSDate

    pred = store.predicateForEventsWithStartDate_endDate_calendars_(
        NSDate.dateWithTimeIntervalSince1970_(start.timestamp()),
        NSDate.dateWithTimeIntervalSince1970_(end.timestamp()),
        store.calendarsForEntityType_(EventKit.EKEntityTypeEvent),
    )
    out = []
    for e in store.eventsMatchingPredicate_(pred) or []:
        out.append(CalEvent(
            start=dt.datetime.fromtimestamp(e.startDate().timeIntervalSince1970()),
            end=dt.datetime.fromtimestamp(e.endDate().timeIntervalSince1970()),
            title=str(e.title() or "(제목 없음)"),
            location=str(e.location() or ""),
            all_day=bool(e.isAllDay()),
            calendar=str(e.calendar().title()),
        ))
    return out


def _events_fresh(start: dt.datetime, end: dt.datetime) -> list[CalEvent]:
    """새 프로세스에서 일정을 읽는다.

    오래 켜 둔 프로세스 안에서는 캘린더 연결이 어느 순간 끊겨, 새 EKEventStore 를 만들어도
    오류 없이 일정 0개만 돌려주는 일이 있다 (밤새 10개 → 0개). 매번 짧게 사는 프로세스로 읽으면
    항상 새 연결이라 이 문제가 없다. 그 프로세스가 실패할 때만 이 프로세스 안에서 읽는다.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "speaker.pim", "--events", str(start.timestamp()), str(end.timestamp())],
            capture_output=True, text=True, timeout=60, cwd=str(_PROJECT_DIR),
        )
        if out.returncode == 0:
            return [
                CalEvent(
                    start=dt.datetime.fromtimestamp(e["start"]), end=dt.datetime.fromtimestamp(e["end"]),
                    title=e["title"], location=e["location"], all_day=e["all_day"], calendar=e["calendar"],
                )
                for e in json.loads(out.stdout)
            ]
        log.warning("일정 읽기 프로세스 실패 (%s): %s", out.returncode, out.stderr.strip()[-300:])
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        log.warning("일정 읽기 프로세스 실패: %s", exc)
    store = _eventkit_store()
    if store is None:
        raise RuntimeError("캘린더 '전체 접근' 권한이 없습니다.")
    return _events_eventkit(store, start, end)


def _print_events(start_ts: float, end_ts: float) -> int:
    store = _eventkit_store()
    if store is None:
        print("캘린더 '전체 접근' 권한이 없습니다.", file=sys.stderr)
        return 2
    events = _events_eventkit(store, dt.datetime.fromtimestamp(start_ts), dt.datetime.fromtimestamp(end_ts))
    json.dump([
        {"start": e.start.timestamp(), "end": e.end.timestamp(), "title": e.title, "location": e.location,
         "all_day": e.all_day, "calendar": e.calendar}
        for e in events
    ], sys.stdout, ensure_ascii=False)
    return 0


# ---------------------------------------------------------------- 메일

_MAIL_SCRIPT = """
on run argv
    set sinceD to (current date) - ((item 1 of argv) as integer) * hours
    set maxN to (item 2 of argv) as integer
    set out to ""
    tell application "Mail"
        set msgs to (messages of inbox whose read status is false and date received > sinceD)
        set i to 0
        repeat with m in msgs
            set i to i + 1
            if i > maxN then exit repeat
            set bulk to "0"
            try
                set hdr to all headers of m
                if hdr contains "List-Unsubscribe" or hdr contains "Precedence: bulk" or hdr contains "List-Id:" then set bulk to "1"
            end try
            set body to ""
            if bulk is "0" then
                try
                    set body to content of m
                    if (length of body) > 400 then set body to text 1 thru 400 of body
                end try
            end if
            set out to out & ((date received of m) as «class isot» as string) & (ASCII character 31) & ¬
                (sender of m) & (ASCII character 31) & (subject of m) & (ASCII character 31) & body & (ASCII character 31) & bulk & (ASCII character 30)
        end repeat
    end tell
    return out
end run
"""


def _unread_mail(since_hours: int, limit: int) -> tuple[int, list[MailItem]]:
    r = subprocess.run(["osascript", "-e", _MAIL_SCRIPT, str(since_hours), str(limit)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "메일 앱을 읽지 못했습니다.")
    items = []
    for rec in r.stdout.split(RECORD):
        parts = rec.strip("\n").split(FIELD)
        if len(parts) != 5:
            continue
        when, sender, subject, body, bulk = parts
        try:
            received = dt.datetime.fromisoformat(when)
        except ValueError:
            received = None
        preview = " ".join(body.split())[:300]
        sender = sender.strip()
        is_bulk = bulk == "1" or bool(_BULK_SENDER.search(sender))
        items.append(MailItem(received, sender, subject.strip() or "(제목 없음)", preview, is_bulk))
    items.sort(key=lambda m: m.received or dt.datetime.min, reverse=True)
    return len(items), items


# ---------------------------------------------------------------- 캐시 + 공개 API


class PersonalInfo:
    """일정과 메일을 뒤에서 주기적으로 읽어 두고, 도구가 물으면 바로 돌려준다."""

    def __init__(self, calendar: bool = True, mail: bool = True):
        self.calendar_enabled = calendar
        self.mail_enabled = mail
        self._lock = threading.Lock()
        self._events: list[CalEvent] | None = None
        self._events_at = 0.0
        self._events_source = ""
        self._mail: tuple[int, list[MailItem]] | None = None
        self._mail_at = 0.0
        self._errors: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pim-refresh", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(REFRESH_S)

    def refresh(self) -> None:
        if self.calendar_enabled:
            self._refresh_events()
        if self.mail_enabled:
            self._refresh_mail()

    def _refresh_events(self) -> None:
        t0 = time.monotonic()
        try:
            store = _eventkit_store()
            if store is None:
                raise RuntimeError(
                    "캘린더 '전체 접근' 권한이 없습니다. 시스템 설정 → 개인정보 보호 및 보안 → 캘린더에서 "
                    "스피커를 실행하는 앱을 '전체 접근'으로 바꿔 주세요."
                )
            today = dt.datetime.combine(dt.date.today(), dt.time())
            events, source = _events_fresh(today, today + dt.timedelta(days=8)), "eventkit"
            events.sort(key=lambda e: e.start)
            with self._lock:
                self._events, self._events_at, self._events_source = events, time.time(), source
                self._errors.pop("calendar", None)
            log.info("일정 %d개 읽음 (%s, %.1f초)", len(events), source, time.monotonic() - t0)
        except Exception as exc:
            with self._lock:
                self._errors["calendar"] = str(exc)
            log.warning("일정을 읽지 못했습니다: %s", exc)

    def _refresh_mail(self) -> None:
        t0 = time.monotonic()
        try:
            result = _unread_mail(since_hours=48, limit=30)
            with self._lock:
                self._mail, self._mail_at = result, time.time()
                self._errors.pop("mail", None)
            log.info("안 읽은 메일 %d통 중 최근 %d통 읽음 (%.1f초)", result[0], len(result[1]), time.monotonic() - t0)
        except Exception as exc:
            with self._lock:
                self._errors["mail"] = str(exc)
            log.warning("메일을 읽지 못했습니다: %s", exc)

    # --- 도구가 쓰는 함수 (문자열로 돌려준다)

    def calendar_text(self, day: str = "today", date: str | None = None) -> str:
        if not self.calendar_enabled:
            return "일정 기능이 꺼져 있습니다."
        with self._lock:
            events, at, source, err = self._events, self._events_at, self._events_source, self._errors.get("calendar")
        if events is None or time.time() - at > 3 * REFRESH_S:
            self._refresh_events()  # 아직 못 읽었거나 오래됨: 지금 읽는다
            with self._lock:
                events, source, err = self._events, self._events_source, self._errors.get("calendar")
        if events is None:
            return f"일정을 읽지 못했습니다: {err}"
        today = dt.date.today()
        if date:
            first = dt.date.fromisoformat(date)
            last = first
        elif day == "tomorrow":
            first = last = today + dt.timedelta(days=1)
        elif day == "week":
            first, last = today, today + dt.timedelta(days=6)
        else:
            first = last = today
        picked = [e for e in events if first <= e.start.date() <= last]
        if not picked:
            label = "그날" if date else {"tomorrow": "내일", "week": "이번 주"}.get(day, "오늘")
            return f"{label} 일정이 없습니다."
        now = dt.datetime.now()
        lines = []
        for e in picked:
            when = "하루 종일" if e.all_day else f"{e.start:%H:%M}~{e.end:%H:%M}"
            if first != last:
                when = f"{e.start:%m/%d}({'월화수목금토일'[e.start.weekday()]}) {when}"
            done = " (지남)" if not e.all_day and e.end < now else ""
            loc = f" @ {e.location}" if e.location else ""
            lines.append(f"- {when} {e.title}{loc}{done}")
        return f"일정 {len(picked)}개:\n" + "\n".join(lines)

    def mail_text(self, limit: int = 10) -> str:
        if not self.mail_enabled:
            return "메일 기능이 꺼져 있습니다."
        with self._lock:
            mail, at, err = self._mail, self._mail_at, self._errors.get("mail")
        if mail is None or time.time() - at > 3 * REFRESH_S:  # 평소에는 뒤에서 읽어 둔 것을 쓴다
            self._refresh_mail()
            with self._lock:
                mail, err = self._mail, self._errors.get("mail")
        if mail is None:
            return f"메일을 읽지 못했습니다: {err}"
        total, items = mail
        if total == 0:
            return "최근 48시간 동안 온 안 읽은 메일이 없습니다."
        personal = [m for m in items if not m.bulk]
        bulk = len(items) - len(personal)
        lines = [f"최근 48시간에 온 안 읽은 메일 {len(items)}통 중 홍보·뉴스레터·자동 알림 {bulk}통은 뺐고, "
                 f"나머지 {len(personal)}통:"]
        for m in personal[:limit]:
            when = f"{m.received:%m/%d %H:%M}" if m.received else ""
            lines.append(f"- [{when}] 보낸 사람: {_display_name(m.sender)} / 제목: {m.subject} / 앞부분: {m.preview}")
        if not personal:
            lines.append("- 직접 보낸 것으로 보이는 메일은 없음")
        return "\n".join(lines)


def _display_name(sender: str) -> str:
    """'홍길동 <gildong@x.edu>' → '홍길동'. 이름이 없으면 주소의 @ 앞부분."""
    m = re.match(r'\s*"?([^"<]+?)"?\s*<', sender)
    if m:
        return m.group(1).strip()
    return sender.split("@")[0].strip("<> ")


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--events":
    sys.exit(_print_events(float(sys.argv[2]), float(sys.argv[3])))
