"""YouTube Music 계정 연결 도우미.

    uv run python -m speaker.ytmusic_setup              # 붙여 넣기
    uv run python -m speaker.ytmusic_setup --clipboard  # 복사해 둔 클립보드에서 바로 읽기

브라우저에서 복사한 요청 헤더(로그인 정보)를 붙여 넣으면 ytmusic_auth.json 으로 저장하고,
실제로 내 보관함을 읽을 수 있는지 확인한다. 저장된 파일은 계정 비밀번호처럼 다뤄야 한다.
"""

from __future__ import annotations

import os
import sys

from .config import load_config

GUIDE = """\
YouTube Music 계정 연결

1. Chrome 에서 https://music.youtube.com 을 열고, 연결할 계정으로 로그인되어 있는지 확인하세요.
2. Cmd + Option + I 로 개발자 도구를 열고 'Network'(네트워크) 탭을 고르세요.
3. 필터 칸에 browse 라고 입력한 뒤, 왼쪽 메뉴에서 '보관함'을 눌러 요청이 생기게 하세요.
4. 목록에 생긴 browse?... 요청을 누르고 'Headers' 탭의 'Request Headers' 부분으로 가세요.
5. 'accept: */*' 같은 첫 줄부터 그 부분 끝까지 마우스로 모두 선택해 복사하세요.
   (Firefox 라면 요청을 오른쪽 클릭 → Copy Value → Copy Request Headers)
6. 아래에 붙여 넣고, 새 줄에서 Ctrl + D 를 누르세요.
   (붙여 넣기가 잘 안 되면: 복사만 해 두고 Ctrl + C 로 끈 뒤 --clipboard 를 붙여 다시 실행)

붙여 넣은 내용은 화면에 그대로 보이지만 이 맥북의 파일로만 저장됩니다.
"""


def main() -> int:
    cfg = load_config()
    path = cfg.ytmusic_auth_path
    print(GUIDE)
    if path.exists():
        print(f"이미 연결 정보가 있습니다: {path}")
        print("새로 붙여 넣으면 덮어씁니다. 그만두려면 Ctrl + C.\n")

    if "--clipboard" in sys.argv:
        # 붙여 넣기가 잘 안 될 때: 브라우저에서 복사해 둔 클립보드를 바로 읽는다
        import subprocess

        raw = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout.strip()
        print("클립보드에서 읽었습니다." if raw else "클립보드가 비어 있습니다.")
    else:
        lines: list[str] = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        except KeyboardInterrupt:
            print("\n취소했습니다.")
            return 1
        raw = "\n".join(lines).strip()
    if not raw:
        print("붙여 넣은 내용이 없어 취소했습니다.")
        return 1

    from ytmusicapi import YTMusic
    from ytmusicapi.auth.browser import setup_browser
    from ytmusicapi.exceptions import YTMusicError

    tmp = path.with_suffix(".tmp")
    try:
        # 파일을 만들기 전에 권한부터 좁혀 둔다 (나만 읽을 수 있게)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.close(fd)
        setup_browser(str(tmp), headers_raw=raw)
    except YTMusicError as exc:
        tmp.unlink(missing_ok=True)
        print(f"\n헤더를 읽지 못했습니다: {exc}")
        print("로그인된 상태에서 browse 요청의 Request Headers 전체를 복사했는지 확인해 주세요.")
        return 1

    print("\n연결을 확인하는 중...")
    try:
        yt = YTMusic(auth=str(tmp), language="ko", location="KR")
        playlists = yt.get_library_playlists(limit=50)
        liked = yt.get_liked_songs(limit=5)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"계정 정보를 확인하지 못했습니다: {type(exc).__name__}: {exc}")
        print("브라우저에서 로그아웃했거나 다른 요청의 헤더를 복사했을 수 있습니다. 다시 시도해 주세요.")
        return 1

    os.replace(tmp, path)
    os.chmod(path, 0o600)
    names = [p.get("title") for p in playlists if p.get("title")]
    print(f"연결됐습니다. 저장 위치: {path}")
    print(f"보관함 재생목록 {len(names)}개: {', '.join(names[:10])}{' …' if len(names) > 10 else ''}")
    print(f"좋아요한 곡: {liked.get('trackCount') or len(liked.get('tracks') or [])}곡")
    print("\n스피커를 다시 켜면 '좋아요한 노래 틀어줘', '내 ○○ 재생목록 틀어줘'를 쓸 수 있습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
