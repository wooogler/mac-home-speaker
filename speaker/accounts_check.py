"""맥 캘린더·메일 앱에 추가한 계정을 스피커가 읽을 수 있는지 확인한다.

    uv run python -m speaker.accounts_check

macOS 는 이 권한을 '실행한 앱'(터미널)에 준다. 스피커를 켤 때 쓰는 터미널 앱에서 실행해야
처음 한 번 뜨는 캘린더·메일 접근 허용 창에 '허용'을 누를 수 있다.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import threading


def check_calendar() -> bool:
    import EventKit
    from Foundation import NSDate

    status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)
    store = EventKit.EKEventStore.alloc().init()
    if status != 3:  # 3=전체 접근 (4=쓰기 전용은 읽을 수 없음)
        done = threading.Event()
        result: dict = {}

        def completion(granted, error):
            result["granted"] = bool(granted)
            done.set()

        print("캘린더 접근을 요청합니다. 화면에 허용 창이 뜨면 '허용'을 눌러 주세요...")
        if hasattr(store, "requestFullAccessToEventsWithCompletion_"):
            store.requestFullAccessToEventsWithCompletion_(completion)
        else:
            store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeEvent, completion)
        if not done.wait(120) or not result.get("granted"):
            print("✗ 캘린더 접근이 허용되지 않았습니다.")
            print("  시스템 설정 → 개인정보 보호 및 보안 → 캘린더에서 이 터미널 앱을 '전체 접근'으로 켜 주세요.")
            return False
        store = EventKit.EKEventStore.alloc().init()

    print("✓ 캘린더 접근 가능")
    for cal in store.calendarsForEntityType_(EventKit.EKEntityTypeEvent):
        print(f"    캘린더: {cal.title()}  [계정: {cal.source().title()}]")
    start = dt.datetime.combine(dt.date.today(), dt.time())
    end = start + dt.timedelta(days=7)
    pred = store.predicateForEventsWithStartDate_endDate_calendars_(
        NSDate.dateWithTimeIntervalSince1970_(start.timestamp()),
        NSDate.dateWithTimeIntervalSince1970_(end.timestamp()),
        None,
    )
    events = sorted(store.eventsMatchingPredicate_(pred) or [], key=lambda e: e.startDate().timeIntervalSince1970())
    print(f"    앞으로 7일 일정 {len(events)}개")
    for e in events[:3]:
        when = dt.datetime.fromtimestamp(e.startDate().timeIntervalSince1970())
        print(f"      {when:%m/%d %H:%M} {e.title()}")
    return True


MAIL_SCRIPT = """
tell application "Mail"
    set out to ""
    repeat with acc in every account
        set out to out & (name of acc) & linefeed
    end repeat
    set recent to count of (messages of inbox whose read status is false and date received > ((current date) - 2 * days))
    return out & "최근 48시간 안 읽은 메일 " & recent & "통 (모든 계정의 받은 편지함)"
end tell
"""


def check_mail() -> bool:
    print("\n메일 앱 접근을 확인합니다. 'Mail을 제어하도록 허용' 창이 뜨면 '허용'을 눌러 주세요...")
    try:
        out = subprocess.run(["osascript", "-e", MAIL_SCRIPT], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("✗ 메일 앱이 120초 안에 응답하지 않았습니다.")
        return False
    if out.returncode != 0:
        print(f"✗ 메일 앱에 접근하지 못했습니다: {out.stderr.strip()}")
        print("  시스템 설정 → 개인정보 보호 및 보안 → 자동화에서 이 터미널 앱의 'Mail'을 켜 주세요.")
        return False
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        print("✗ 메일 앱에 계정이 없습니다. 인터넷 계정에서 학교 계정의 '메일'이 켜져 있는지 확인해 주세요.")
        return False
    print("✓ 메일 앱 접근 가능")
    for line in lines:
        print(f"    {line}")
    return True


def main() -> int:
    cal = check_calendar()
    mail = check_mail()
    print("\n결과:", "캘린더 ○" if cal else "캘린더 ×", "/", "메일 ○" if mail else "메일 ×")
    return 0 if cal and mail else 1


if __name__ == "__main__":
    sys.exit(main())
