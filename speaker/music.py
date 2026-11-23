"""유튜브 뮤직 재생: ytmusicapi 로 검색하고, 오래 사는 mpv 프로세스 하나를 JSON IPC 로 조종한다.

대기열은 music.youtube.com/watch?v=<id> URL 목록으로 mpv 에 넣고, 각 곡의 실제 스트림은
mpv 의 ytdl_hook(yt-dlp)이 그 곡 차례가 올 때 해석한다 (스트림 URL 은 만료되므로 미리 풀지 않는다).
"""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .config import Config

log = logging.getLogger("speaker.music")

WATCH_URL = "https://music.youtube.com/watch?v={}"
START_TIMEOUT = 20.0  # 곡을 불러오기 시작해서 소리가 날 때까지 기다리는 최대 시간 (초)
IPC_TIMEOUT = 5.0
QUEUE_LIMIT = 50
PODCAST_QUEUE = 11  # 최신 에피소드 + 이전 에피소드 10개 ("다음 에피소드" 용)


class MusicError(Exception):
    """사용자에게 그대로 읽어줄 수 있는 한국어 메시지를 담는다."""


# ---------------------------------------------------------------- 텍스트 도우미


def _norm(text: str) -> str:
    return "".join(re.findall(r"\w+", (text or "").lower()))


def _relevant(query: str, texts: list[str]) -> bool:
    """검색어 토큰이 결과 메타데이터에 (혹은 그 반대로) 들어 있는지 대략 확인."""
    squashed = _norm(query)
    tokens = {t for t in (_norm(w) for w in query.split()) if len(t) >= 2}
    if len(squashed) >= 2:
        tokens.add(squashed)
    for text in texts:
        n = _norm(text)
        if not n:
            continue
        if any(t in n for t in tokens) or (len(n) >= 2 and n in squashed):
            return True
    return False


# 음원 대신 공연·무대·뮤직비디오 영상이 섞이지 않게 거르는 단어
_PERFORMANCE = re.compile(
    r"\blive\b|라이브|\bstage\b|무대|performance|퍼포먼스|직캠|fancam|\bm/?v\b|music video|뮤직\s*비디오|"
    r"교차\s*편집|\bcover\b|커버|dance practice|안무|concert|콘서트|공연|\btour\b|full ver|mirrored|reaction",
    re.I,
)


def _is_audio_track(item: dict) -> bool:
    """유튜브 뮤직의 '음원' 항목인지 (뮤직비디오·사용자 영상이 아닌)."""
    return item.get("videoType") == "MUSIC_VIDEO_TYPE_ATV"


def _is_performance(item: dict) -> bool:
    return bool(_PERFORMANCE.search(item.get("title") or ""))


def _prefer_audio(items: list[dict], min_audio: int = 5) -> list[dict]:
    """음원을 우선한다: 공연·무대·커버 영상은 빼고, 음원이 충분하면(min_audio 곡 이상) 뮤직비디오도 뺀다."""
    items = [t for t in items if _is_audio_track(t) or not _is_performance(t)]
    audio = [t for t in items if _is_audio_track(t)]
    return audio if len(audio) >= min_audio else items


def _best_playlist(name: str, playlists: list[dict]) -> dict | None:
    """말로 들은 이름에 가장 가까운 보관함 재생목록. (음성 인식 오차를 감안해 느슨하게 맞춘다)"""
    want = _norm(name)
    if not want:
        return None
    best, best_score = None, 0.0
    for p in playlists:
        title = _norm(p.get("title") or "")
        if not title:
            continue
        if want == title:
            return p
        if want in title or title in want:
            score = 0.9
        else:
            score = SequenceMatcher(None, want, title).ratio()
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 0.6 else None


def _josa(word: str, with_final: str, without_final: str) -> str:
    """마지막 글자 받침에 맞는 조사. 한글이 아니면 '을(를)' 식으로."""
    for ch in reversed(word.strip(" '\")]")):
        if "가" <= ch <= "힣":
            return with_final if (ord(ch) - 0xAC00) % 28 else without_final
        if ch.isalnum():
            break
    return f"{with_final}({without_final})"


def _artists(item: dict) -> str:
    # ko 파서가 업로드 날짜("2023. 5. 17.")를 아티스트 자리에 넣는 경우가 있어 걸러낸다
    names = [
        a["name"] for a in item.get("artists") or []
        if a.get("name") and not re.fullmatch(r"[\d.\s]+", a["name"])
    ]
    return ", ".join(names)


_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_AGO_UNITS = {"s": 0, "m": 0, "h": 0, "d": 1, "w": 7, "mo": 30, "y": 365}


def _upload_date(text: str | None, today: datetime.date | None = None) -> datetime.date | None:
    """en 검색 결과의 날짜 문구("8h ago", "2d ago", "Nov 5", "Nov 1, 2025")를 날짜로."""
    if not text:
        return None
    today = today or datetime.date.today()
    t = text.lower().replace("\xa0", " ")
    m = re.search(r"(\d+)\s*(mo|[smhdwy])[a-z]*\s+ago", t)
    if m:
        unit = m.group(2)
        days = int(m.group(1)) * _AGO_UNITS[unit]
        if unit == "h":
            return (datetime.datetime.now() - datetime.timedelta(hours=int(m.group(1)))).date()
        return today - datetime.timedelta(days=days)
    if "yesterday" in t:
        return today - datetime.timedelta(days=1)
    if "today" in t or "just now" in t:
        return today
    m = re.search(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2})(?:,\s*(\d{4}))?", t)
    if m and m.group(1) in _MONTHS:
        try:
            d = datetime.date(int(m.group(3) or today.year), _MONTHS[m.group(1)], int(m.group(2)))
        except ValueError:
            return None
        if not m.group(3) and d > today + datetime.timedelta(days=1):  # 연도 없는 날짜는 올해(미래면 작년)
            d = d.replace(year=d.year - 1)
        return d
    return None


def _title_date(title: str) -> datetime.date | None:
    """에피소드 제목에 들어 있는 날짜 ("2026년 11월 13일", "2026.11.13", "20261113")."""
    for pat in (r"(20\d{2})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})", r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"):
        for m in re.finditer(pat, title or ""):
            try:
                return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _episode_date(title: str, upload: str | None) -> datetime.date | None:
    """방송 날짜: 제목에 날짜가 있으면 그것(업로드 날짜와 크게 어긋나지 않을 때), 없으면 업로드 날짜."""
    up = _upload_date(upload)
    td = _title_date(title)
    if td and (up is None or abs((td - up).days) <= 3):
        return td
    return up


def _korean_date(d: datetime.date) -> str:
    if d.year == datetime.date.today().year:
        return f"{d.month}월 {d.day}일"
    return f"{d.year}년 {d.month}월 {d.day}일"


def _clean_show(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip()


def _short_episode(title: str, show: str) -> str:
    """읽어 주기용 에피소드 제목: 쇼 이름 머리말, [괄호] 꼬리표, ' / 방송사' 등을 떼고 짧게."""
    t = _clean_show(title)
    if show and t.lower().startswith(show.lower()) and len(t) > len(show) + 2:
        t = t[len(show):].lstrip(" -:|·")
    t = "".join(" " if unicodedata.category(ch) == "So" else ch for ch in t)  # 이모지 등은 읽지 않는다
    t = re.sub(r"\s+", " ", re.sub(r"\(\s*\)", " ", t)).strip()
    stripped = re.sub(r"\s*[\[【〔][^\]】〕]*[\]】〕]\s*", " ", t).strip()
    t = stripped or t
    t = re.split(r"\s+[/|｜]\s+|\s*｜\s*", t)[0].strip(" -–:|")
    # 제목 속 날짜("– 2026년 11월 15일(일)")는 떼고 설명에서 따로 읽는다 (날짜뿐인 제목이면 그대로)
    no_date = re.sub(r"\s*[-–]?\s*20\d{2}\s*[년.\-/]\s*\d{1,2}\s*[월.\-/]\s*\d{1,2}일?\.?\s*(\([^)]*\))?", " ", t).strip(" -–:|,")
    if len(no_date) >= 8:
        t = no_date
    if len(t) > 55:
        cut = t[:55]
        t = cut[: cut.rfind(" ")] if " " in cut[15:] else cut
    if t.count("(") > t.count(")"):  # 잘려서 짝이 안 맞는 괄호 뒷부분은 버린다
        t = t[: t.rfind("(")]
    t = t.strip(" -–:|,")
    return t or _clean_show(title)[:55]


def _playable_episode(e: dict) -> bool:
    # 예약(최초 공개 예정) 에피소드는 아직 재생할 수 없다. ko 에서는 date 가 "예정일: ..."
    return bool(e.get("videoId")) and not re.search(r"예정|scheduled|premiere", e.get("date") or "", re.I)


def _video_id(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return parse_qs(urlparse(url).query).get("v", [None])[0]
    except ValueError:
        return None


# ---------------------------------------------------------------- mpv IPC


class _MpvIPC:
    """mpv JSON IPC 연결. 응답은 request_id 로 짝짓고, 이벤트는 콜백으로 넘긴다."""

    def __init__(self, path: str, on_event):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(path)
        self._on_event = on_event
        self._write_lock = threading.Lock()
        self._pending: dict[int, list] = {}  # id -> [Event, response]
        self._next_id = 0
        self.closed = False
        threading.Thread(target=self._reader, name="mpv-ipc", daemon=True).start()

    def _reader(self) -> None:
        buf = b""
        try:
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if "request_id" in msg and "event" not in msg:
                        slot = self._pending.get(msg["request_id"])
                        if slot:
                            slot[1] = msg
                            slot[0].set()
                    elif "event" in msg:
                        try:
                            self._on_event(msg)
                        except Exception:  # 이벤트 처리 실패가 리더를 죽이지 않게
                            log.exception("mpv 이벤트 처리 실패")
        except OSError:
            pass
        finally:
            self.closed = True
            for slot in list(self._pending.values()):
                slot[0].set()

    def command(self, *args: Any, timeout: float = IPC_TIMEOUT) -> Any:
        if self.closed:
            raise OSError("mpv IPC closed")
        with self._write_lock:
            self._next_id += 1
            rid = self._next_id
            slot = [threading.Event(), None]
            self._pending[rid] = slot
            data = json.dumps({"command": list(args), "request_id": rid}).encode() + b"\n"
            self._sock.sendall(data)
        try:
            if not slot[0].wait(timeout) or slot[1] is None:
                raise OSError(f"mpv IPC no response: {args[0]}")
            resp = slot[1]
        finally:
            self._pending.pop(rid, None)
        if resp.get("error") != "success":
            raise RuntimeError(f"mpv {args[0]}: {resp.get('error')}")
        return resp.get("data")

    def close(self) -> None:
        self.closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


# ---------------------------------------------------------------- 플레이어


class MusicPlayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.RLock()  # mpv 프로세스/IPC 보호
        self._search_lock = threading.Lock()  # ytmusicapi 세션 보호
        self._proc: subprocess.Popen | None = None
        self._ipc: _MpvIPC | None = None
        self._tmpdir: str | None = None
        self._meta: dict[str, tuple[str, str]] = {}  # videoId -> (곡명, 아티스트)
        self._volume = max(0, min(100, cfg.music_volume))
        self._pre_duck: int | None = None
        self._fade_gen = 0
        self._duck_level = cfg.duck_volume
        self._yt_en = None
        self._yt_ko = None
        self._yt_auth = None  # 계정 연결된 클라이언트 (내 보관함용)
        self._yt_auth_mtime = 0.0
        # 곡을 불러오는 중인지 (play() 는 검색이 끝나면 바로 돌아오고, 소리는 몇 초 뒤에 난다)
        self._play_gen = 0
        self._starting = False
        # 불러오기에 실패했을 때 부를 함수 (한국어 안내 문구를 받는다). main 에서 음성 안내를 연결한다.
        self.on_error: Callable[[str], None] | None = None
        atexit.register(self.shutdown)

    # ------------------------------------------------------------ 검색

    def _clients(self):
        if self._yt_en is None:
            from ytmusicapi import YTMusic

            # language="ko" 는 필터 검색 파서(영문 카테고리명 비교)를 깨뜨리므로 검색은 en+KR,
            # 곡 목록(재생목록/라디오)은 한국어 제목을 받으려고 ko+KR 로 가져온다.
            self._yt_en = YTMusic(language="en", location="KR")
            self._yt_ko = YTMusic(language="ko", location="KR")
        return self._yt_en, self._yt_ko

    @staticmethod
    def _tracks(items: list[dict], limit: int = QUEUE_LIMIT) -> list[tuple[str, str, str]]:
        seen: set[str] = set()
        out = []
        for it in items:
            vid = it.get("videoId")
            if not vid or vid in seen or it.get("isAvailable") is False:
                continue
            seen.add(vid)
            out.append((vid, it.get("title") or "", _artists(it)))
        return out[:limit]

    def _top_result(self, en, query: str) -> list[str]:
        """필터 없는 검색의 '최상위 결과' 텍스트 (관련성 판단용)."""
        try:
            res = en.search(query, limit=5)
        except Exception:
            return []
        if not res:
            return []
        top = res[0]
        return [t for t in (top.get("title"), top.get("artist"), _artists(top)) if t]

    def _search(self, query: str, kind: str) -> tuple[list[tuple[str, str, str]], str]:
        en, ko = self._clients()
        with self._search_lock, ThreadPoolExecutor(max_workers=2) as pool:
            top_future = pool.submit(self._top_result, en, query)
            if kind == "artist":
                return self._search_artist(en, ko, query, top_future)
            if kind == "playlist":
                return self._search_playlist(en, ko, query, top_future)
            return self._search_song(en, ko, query, top_future)

    def _search_song(self, en, ko, query, top_future):
        hits = en.search(query, filter="songs", limit=10) or en.search(query, filter="videos", limit=10)
        hits = [h for h in hits if h.get("videoId")]
        if not hits:
            raise MusicError(f"'{query}'에 해당하는 곡을 찾지 못했습니다.")
        hit = hits[0]
        if not _PERFORMANCE.search(query):
            # 사용자가 라이브·무대를 찾은 게 아니면, 위쪽 결과 중 공연 버전이 아닌 음원을 고른다
            hit = next((h for h in hits[:5] if _is_audio_track(h) and not _is_performance(h)), hit)
        radio = _prefer_audio(ko.get_watch_playlist(videoId=hit["videoId"], limit=QUEUE_LIMIT).get("tracks") or [])
        tracks = self._tracks([t for t in radio if t.get("videoId") != hit["videoId"]])
        first = next((t for t in radio if t.get("videoId") == hit["videoId"]), None) or hit
        seed = (hit["videoId"], first.get("title") or hit.get("title", ""), _artists(first) or _artists(hit))
        texts = [hit.get("title", ""), _artists(hit), (hit.get("album") or {}).get("name", ""), seed[1], seed[2]]
        top = top_future.result()
        agree = any(_norm(t) and (_norm(t) in _norm(hit.get("title", "")) or _norm(hit.get("title", "")) in _norm(t))
                    for t in top[:1]) or any(_norm(t) == _norm(_artists(hit)) for t in top)
        if not (_relevant(query, texts) or agree):
            raise MusicError(f"'{query}'에 해당하는 곡을 찾지 못했습니다.")
        tracks = [seed] + tracks
        title, artist = seed[1], seed[2]
        desc = f"{artist}의 '{title}'{_josa(title, '을', '를')} 재생합니다." if artist else f"'{title}'{_josa(title, '을', '를')} 재생합니다."
        if len(tracks) > 1:
            desc += f" 이어서 비슷한 곡 {len(tracks) - 1}곡이 대기 중입니다."
        return tracks, desc

    def _search_artist(self, en, ko, query, top_future):
        hits = [h for h in en.search(query, filter="artists", limit=5) if h.get("browseId")]
        if not hits:
            raise MusicError(f"'{query}' 아티스트를 찾지 못했습니다.")
        hit = hits[0]
        try:
            artist = ko.get_artist(hit["browseId"])
        except Exception:
            # ytmusicapi 가 일부 아티스트 페이지 구조를 못 읽는다 (제목 없는 묶음이 있을 때 KeyError).
            # 그럴 땐 곡 검색에서 이 아티스트의 곡만 골라 쓴다.
            log.warning("아티스트 페이지를 읽지 못해 곡 검색으로 대신합니다: %s", hit.get("artist"), exc_info=True)
            artist = {"name": hit.get("artist"), "songs": {"results": self._artist_songs(en, ko, hit)}}
        name = artist.get("name") or hit.get("artist") or query
        top = top_future.result()
        if not (_relevant(query, [hit.get("artist", ""), name]) or any(_norm(t) == _norm(hit.get("artist", "")) for t in top)):
            raise MusicError(f"'{query}' 아티스트를 찾지 못했습니다.")
        songs = artist.get("songs") or {}
        items = songs.get("results") or []
        if songs.get("browseId"):
            try:
                items = ko.get_playlist(songs["browseId"], limit=QUEUE_LIMIT).get("tracks") or items
            except Exception:
                log.warning("아티스트 인기곡 목록을 가져오지 못해 요약 목록을 씁니다", exc_info=True)
        tracks = self._tracks(_prefer_audio(items))
        if not tracks:
            raise MusicError(f"{name}의 곡을 찾지 못했습니다.")
        desc = f"{name}의 인기곡 {len(tracks)}곡을 재생합니다. 첫 곡은 '{tracks[0][1]}'입니다."
        return tracks, desc

    @staticmethod
    def _artist_songs(en, ko, hit: dict) -> list[dict]:
        """곡 검색 결과 중 이 아티스트(browseId 일치)의 곡만, 중복 없이."""
        name, aid = hit.get("artist") or "", hit["browseId"]
        out: dict[str, dict] = {}
        for client in (ko, en):
            try:
                results = client.search(name, filter="songs", limit=QUEUE_LIMIT)
            except Exception:
                log.warning("아티스트 곡 검색 실패", exc_info=True)
                continue
            for r in results:
                if r.get("videoId") and any(a.get("id") == aid for a in r.get("artists") or []):
                    out.setdefault(r["videoId"], r)
        return list(out.values())

    def _search_playlist(self, en, ko, query, top_future):
        found: dict[str, list[dict]] = {}
        for flt in ("community_playlists", "featured_playlists"):
            try:
                found[flt] = [p for p in en.search(query, filter=flt, limit=5) if p.get("browseId")]
            except Exception:
                found[flt] = []
                log.warning("재생목록 검색 실패 (%s)", flt, exc_info=True)
        comm, feat = found["community_playlists"], found["featured_playlists"]
        if not comm and not feat:
            raise MusicError(f"'{query}' 재생목록을 찾지 못했습니다.")
        # 한국어 검색어엔 공식(featured) 목록이 엉뚱한 경우가 많아, 제목이 검색어와 겹치는 목록을 우선.
        def hit(p: dict) -> bool:
            return _relevant(query, [p.get("title", "")])

        matched = [p for p in comm + feat if hit(p)][:3]
        if not matched and not _relevant(query, top_future.result()):
            raise MusicError(f"'{query}' 재생목록을 찾지 못했습니다.")
        fallback = [p for p in feat + comm if not hit(p)][:1]
        cands = matched + fallback

        def usable(t: dict) -> bool:
            # 몇 시간짜리 믹스 영상은 '다음 곡'이 의미 없으니 곡 단위 항목만 센다
            d = t.get("duration_seconds") or 0
            return t.get("videoType") in ("MUSIC_VIDEO_TYPE_ATV", "MUSIC_VIDEO_TYPE_OMV") or 60 <= d <= 1200

        def fetch(p: dict):
            try:
                pl = ko.get_playlist(p["browseId"], limit=QUEUE_LIMIT)
            except Exception:
                log.warning("재생목록 불러오기 실패: %s", p.get("title"), exc_info=True)
                return [], [], ""
            items = _prefer_audio(pl.get("tracks") or [])
            return self._tracks([t for t in items if usable(t)]), self._tracks(items), pl.get("title") or p.get("title") or query

        with ThreadPoolExecutor(max_workers=len(cands)) as pool:
            results = list(pool.map(fetch, cands))
        # 우선순위: 제목이 맞고 곡이 10개 이상 → 제목이 맞고 곡이 가장 많은 것(3개 이상)
        #          → 제목이 맞는 첫 목록(긴 믹스여도 분위기가 맞는 쪽) → 대체 목록
        m_res, f_res = results[: len(matched)], results[len(matched):]
        good = sorted([r for r in m_res if len(r[0]) >= 10], key=lambda r: -len(r[0]))
        some = [r for r in m_res if len(r[0]) >= 3]
        anyr = [r for r in m_res if r[1]]
        if good:
            chosen = good[0]
        elif some:
            chosen = max(some, key=lambda r: len(r[0]))
        elif anyr:
            chosen = anyr[0]
        else:
            chosen = max(f_res or results, key=lambda r: (len(r[0]), len(r[1])))
        tracks = chosen[0] if len(chosen[0]) >= 3 else chosen[1]
        title = chosen[2]
        if not tracks:
            raise MusicError(f"'{query}' 재생목록을 불러오지 못했습니다.")
        desc = f"'{title}' 재생목록의 {len(tracks)}곡을 재생합니다."
        if len(tracks[0][1]) <= 40:  # 긴 믹스 영상 제목은 읽지 않는다
            desc += f" 첫 곡은 '{tracks[0][1]}'입니다."
        return tracks, desc

    # ------------------------------------------------------------ mpv 프로세스

    def _alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and self._ipc is not None
            and not self._ipc.closed
        )

    def _ensure_mpv(self) -> _MpvIPC:
        with self._lock:
            if self._alive():
                return self._ipc  # type: ignore[return-value]
            self._kill_mpv()
            mpv = self.cfg.mpv_path
            if not mpv or not (os.path.exists(mpv) or shutil.which(mpv)):
                raise MusicError("음악 재생기(mpv)가 설치되어 있지 않습니다.")
            base = tempfile.gettempdir()
            if len(base) > 60:  # 유닉스 소켓 경로 길이 제한(104바이트) 회피
                base = "/tmp"
            self._tmpdir = tempfile.mkdtemp(prefix="speaker-mpv-", dir=base)
            sock = os.path.join(self._tmpdir, "mpv.sock")
            args = [
                mpv,
                "--no-config",
                "--idle=yes",
                "--no-video",
                "--no-terminal",
                "--force-window=no",
                "--audio-display=no",
                f"--input-ipc-server={sock}",
                "--ytdl=yes",
                # HLS 오디오(itag 234, ~128k AAC)는 즉시 받을 수 있지만, https 오디오는 YouTube 가
                # available_at 으로 ~5초 대기를 요구한다 → HLS 우선, 없으면 일반 bestaudio.
                "--ytdl-format=bestaudio[protocol^=m3u8]/bestaudio/best",
                "--prefetch-playlist=yes",
                # HLS 조각 하나를 받다 실패하면 곡 전체를 건너뛰던 문제: 조각마다 몇 번 다시 받게 한다
                "--demuxer-lavf-o-append=seg_max_retry=5",
                "--volume-max=100",
                f"--volume={self._current_target_volume()}",
            ]
            if self.cfg.ytdlp_path:
                args.append(f"--script-opts=ytdl_hook-ytdl_path={self.cfg.ytdlp_path}")
            node = shutil.which("node") or ("/opt/homebrew/bin/node" if os.path.exists("/opt/homebrew/bin/node") else None)
            if node:
                # yt-dlp 는 기본으로 deno 만 JS 런타임으로 쓴다. node 를 명시적으로 켠다.
                args.append(f"--ytdl-raw-options-append=js-runtimes=node:{node}")
            if self.cfg.mpv_ao:
                args.append(f"--ao={self.cfg.mpv_ao}")
            log.info("mpv 시작: %s", " ".join(args))
            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # 터미널 Ctrl+C 가 mpv 에 직접 가지 않게
            )
            deadline = time.monotonic() + 5
            while True:
                if self._proc.poll() is not None:
                    raise MusicError("음악 재생기를 시작하지 못했습니다.")
                try:
                    self._ipc = _MpvIPC(sock, self._on_event)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        self._kill_mpv()
                        raise MusicError("음악 재생기에 연결하지 못했습니다.")
                    time.sleep(0.05)
            try:
                self._ipc.command("request_log_messages", "warn")
            except Exception:
                pass
            return self._ipc

    def _current_target_volume(self) -> int:
        if self._pre_duck is not None:
            return min(self._duck_level, self._pre_duck)
        return self._volume

    def _on_event(self, msg: dict) -> None:
        ev = msg.get("event")
        if ev == "log-message":
            # 스트림 주소가 수백 자라 로그를 뒤덮는다: 주소는 줄이고, 재시도 과정의 경고는 자세히 볼 때만 남긴다
            text = re.sub(r"https?://\S+", "<스트림 주소>", (msg.get("text") or "").strip())
            log.debug("mpv[%s] %s", msg.get("prefix"), text)
        elif ev == "end-file" and msg.get("reason") == "error":
            log.warning("곡 하나를 불러오지 못해 다음 곡으로 넘어갑니다 (%s)", msg.get("file_error"))

    def _kill_mpv(self) -> None:
        if self._ipc is not None:
            self._ipc.close()
            self._ipc = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(2)
            self._proc = None
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def _cmd(self, *args: Any) -> Any:
        """mpv 가 살아 있을 때만 명령. 죽어 있으면 None."""
        with self._lock:
            if not self._alive():
                return None
            try:
                return self._ipc.command(*args)  # type: ignore[union-attr]
            except (OSError, RuntimeError) as e:
                log.debug("mpv 명령 실패 %s: %s", args, e)
                return None

    def _get(self, prop: str) -> Any:
        return self._cmd("get_property", prop)

    def _set(self, prop: str, value: Any) -> None:
        self._cmd("set_property", prop, value)

    # ------------------------------------------------------------ 공개 API

    def play(self, query: str, kind: str = "song", wait: bool = False) -> str:
        """검색해서 재생 목록을 넣고 곧바로 돌아온다. 실제 소리는 몇 초 뒤(스트림 주소를 받은 뒤)에 난다.

        wait=True 면 소리가 날 때까지 기다리고, 실패하면 MusicError 를 낸다.
        wait=False 면 불러오기 실패를 on_error 로 알린다.
        """
        query = (query or "").strip()
        if not query:
            raise MusicError("무엇을 틀지 알려 주세요.")
        if kind not in ("song", "artist", "playlist"):
            kind = "song"
        t0 = time.monotonic()
        try:
            tracks, desc = self._search(query, kind)
        except MusicError:
            raise
        except Exception as e:
            log.exception("유튜브 뮤직 검색 실패")
            raise MusicError("유튜브 뮤직 검색에 실패했습니다.") from e
        log.info("검색 완료 %.1fs: %s (%d곡)", time.monotonic() - t0, desc, len(tracks))
        return self._start_queue(tracks, desc, t0, wait)

    def _start_queue(self, tracks: list[tuple[str, str, str]], desc: str, t0: float, wait: bool) -> str:
        """곡 목록을 mpv 에 넣고 재생을 시작한다. (play / play_library 공용)"""
        with self._lock:
            ipc = self._ensure_mpv()
            try:
                self._meta = {vid: (title, artist) for vid, title, artist in tracks}
                ipc.command("loadfile", WATCH_URL.format(tracks[0][0]), "replace")
                for vid, _, _ in tracks[1:]:
                    ipc.command("loadfile", WATCH_URL.format(vid), "append")
                ipc.command("set_property", "pause", False)
                self._fade_gen += 1
                ipc.command("set_property", "volume", self._current_target_volume())
            except (OSError, RuntimeError) as e:
                self._kill_mpv()
                raise MusicError("재생을 시작하지 못했습니다.") from e
            self._play_gen += 1
            gen = self._play_gen
            self._starting = True

        if wait:
            error = self._wait_loaded(gen, t0)
            if error:
                raise MusicError(error)
            return desc
        threading.Thread(target=self._watch_start, args=(gen, t0), name="music-start", daemon=True).start()
        return desc

    # ------------------------------------------------------------ 내 보관함 (계정 연결 시)

    def has_account(self) -> bool:
        return self.cfg.ytmusic_auth_path.exists()

    def _account(self):
        """로그인된 ytmusicapi 클라이언트. 연결 정보 파일이 바뀌면 다시 만든다."""
        path = self.cfg.ytmusic_auth_path
        if not path.exists():
            raise MusicError("유튜브 뮤직 계정이 연결되어 있지 않습니다. 터미널에서 계정 연결 도우미를 먼저 실행해야 합니다.")
        mtime = path.stat().st_mtime
        if self._yt_auth is None or self._yt_auth_mtime != mtime:
            from ytmusicapi import YTMusic

            self._yt_auth = YTMusic(auth=str(path), language="ko", location="KR")
            self._yt_auth_mtime = mtime
        return self._yt_auth

    def library_playlists(self) -> list[dict]:
        """보관함의 재생목록 [{title, playlistId, count}] (좋아요 목록 제외)."""
        with self._search_lock:
            items = self._account().get_library_playlists(limit=None)
        return [p for p in items if p.get("playlistId") and p.get("playlistId") != "LM"]

    def play_library(self, source: str, name: str | None = None, shuffle: bool = False, wait: bool = False) -> str:
        """내 계정의 음악을 재생한다.

        source: liked(좋아요한 곡) | playlist(보관함 재생목록, name 으로 찾음) | mix(홈 화면 맞춤 추천) | recent(최근 들은 곡)
        """
        t0 = time.monotonic()
        try:
            tracks, desc = self._library_tracks(source, (name or "").strip())
        except MusicError:
            raise
        except Exception as e:
            log.exception("내 보관함 불러오기 실패 (%s)", source)
            if "auth" in str(e).lower() or "401" in str(e) or "403" in str(e):
                raise MusicError("유튜브 뮤직 로그인이 만료된 것 같습니다. 계정 연결을 다시 해야 합니다.") from e
            raise MusicError("유튜브 뮤직 보관함을 불러오지 못했습니다.") from e
        if not tracks:
            raise MusicError("재생할 곡이 없습니다.")
        if shuffle:
            random.shuffle(tracks)
            desc += " 섞어서 재생합니다."
        log.info("보관함 불러오기 %.1fs: %s (%d곡)", time.monotonic() - t0, desc, len(tracks))
        return self._start_queue(tracks, desc, t0, wait)

    def _library_tracks(self, source: str, name: str) -> tuple[list[tuple[str, str, str]], str]:
        yt = self._account()
        with self._search_lock:
            if source == "liked":
                data = yt.get_liked_songs(limit=QUEUE_LIMIT * 4)
                tracks = self._tracks(data.get("tracks") or [], limit=QUEUE_LIMIT * 4)
                return tracks, f"좋아요한 곡 {len(tracks)}곡을 재생합니다."
            if source == "recent":
                tracks = self._tracks(_prefer_audio(yt.get_history()))
                return tracks, f"최근 들은 곡 {len(tracks)}곡을 재생합니다."
            if source == "mix":
                return self._home_mix(yt)
        if source != "playlist":
            raise MusicError(f"알 수 없는 종류입니다: {source}")
        playlists = self.library_playlists()
        if not playlists:
            raise MusicError("보관함에 재생목록이 없습니다.")
        target = _best_playlist(name, playlists)
        if target is None:
            names = ", ".join(p.get("title", "") for p in playlists[:15])
            raise MusicError(f"'{name}'에 맞는 재생목록을 찾지 못했습니다. 보관함 재생목록: {names}")
        with self._search_lock:
            data = yt.get_playlist(target["playlistId"], limit=QUEUE_LIMIT * 4)
        tracks = self._tracks(data.get("tracks") or [], limit=QUEUE_LIMIT * 4)
        return tracks, f"보관함의 '{target.get('title')}' 재생목록 {len(tracks)}곡을 재생합니다."

    def _home_mix(self, yt) -> tuple[list[tuple[str, str, str]], str]:
        """홈 화면 맞춤 추천: 곡이 여러 개 있는 첫 줄(보통 '빠른 선곡'), 없으면 첫 믹스 재생목록."""
        rows = yt.get_home(limit=6)
        for row in rows:
            songs = [c for c in row.get("contents") or [] if c.get("videoId")]
            if len(songs) >= 5:
                tracks = self._tracks(songs)
                # 추천 곡 뒤에 첫 곡과 비슷한 곡을 이어 붙인다
                radio = yt.get_watch_playlist(videoId=tracks[0][0], limit=QUEUE_LIMIT)
                radio["tracks"] = _prefer_audio(radio.get("tracks") or [])
                seen = {t[0] for t in tracks}
                tracks += [t for t in self._tracks(radio.get("tracks") or []) if t[0] not in seen]
                return tracks[:QUEUE_LIMIT], f"'{row.get('title')}' 추천 곡을 재생합니다."
        for row in rows:
            for c in row.get("contents") or []:
                pid = c.get("playlistId")
                if pid:
                    data = yt.get_watch_playlist(playlistId=pid, limit=QUEUE_LIMIT)
                    tracks = self._tracks(data.get("tracks") or [])
                    if tracks:
                        return tracks, f"맞춤 추천 '{c.get('title')}'을(를) 재생합니다."
        raise MusicError("맞춤 추천을 찾지 못했습니다.")

    # ------------------------------------------------------------ 팟캐스트

    def play_podcast(self, query: str, episode: str = "latest", wait: bool = False) -> str:
        """팟캐스트를 재생한다.

        query: 쇼 이름이나 주제 (예: "김어준의 겸손은힘들다", "KBS 뉴스", "경제 팟캐스트", "tech news")
        episode: "latest" → 가장 잘 맞는 쇼의 최신 에피소드, 뒤로 이전 에피소드들을 대기열에.
                 "search" → 검색어를 에피소드 검색으로 보고 가장 잘 맞는 에피소드를 재생.
        재생 목록의 곡 정보는 (에피소드 제목, 쇼 이름) 이라 now_playing() 은 "에피소드 - 쇼".
        """
        query = (query or "").strip()
        # "KBS 뉴스 팟캐스트 최신 회차"처럼 붙은 일반적인 말은 검색을 흐리므로 뗀다
        cleaned = re.sub(r"(팟\s*캐스트|podcasts?|최신|최근|회차|에피소드|episodes?|latest)", " ", query, flags=re.I)
        cleaned = " ".join(cleaned.split())
        query = cleaned or query
        if not query:
            raise MusicError("어떤 팟캐스트를 틀지 알려 주세요.")
        t0 = time.monotonic()
        try:
            if episode == "search":
                tracks, desc = self._podcast_episode_search(query)
            else:
                tracks, desc = self._podcast_latest(query)
        except MusicError:
            raise
        except Exception as e:
            log.exception("팟캐스트 검색 실패")
            raise MusicError("팟캐스트 검색에 실패했습니다.") from e
        log.info("팟캐스트 검색 완료 %.1fs: %s (%d개)", time.monotonic() - t0, desc, len(tracks))
        return self._start_queue(tracks, desc, t0, wait)

    @staticmethod
    def _search_safe(client, query: str, flt: str, limit: int = 20) -> list[dict]:
        try:
            return client.search(query, filter=flt, limit=limit) or []
        except Exception:
            log.warning("팟캐스트 검색 실패 (%s: %s)", flt, query, exc_info=True)
            return []

    @staticmethod
    def _show_score(query: str, title: str) -> float:
        """쇼 제목이 검색어와 얼마나 겹치는지. 0 이면 무관."""
        squashed = _norm(query)
        tokens = {t for t in (_norm(w) for w in query.split()) if len(t) >= 2}
        n = _norm(title)
        if not n:
            return 0.0
        score = 0.0
        if len(squashed) >= 2 and squashed in n:
            score += 2
        elif len(n) >= 2 and n in squashed:
            score += 1.5
        if tokens:
            score += sum(t in n for t in tokens) / len(tokens)
        return score

    def _podcast_latest(self, query: str) -> tuple[list[tuple[str, str, str]], str]:
        en, ko = self._clients()
        with self._search_lock, ThreadPoolExecutor(max_workers=2) as pool:
            # 필터 검색은 en 에서만 동작한다 (ko 는 빈 결과). 에피소드 검색은 날짜/쇼 후보를 얻으려고 같이.
            shows_f = pool.submit(self._search_safe, en, query, "podcasts", 10)
            eps_f = pool.submit(self._search_safe, en, query, "episodes", 20)
            shows = [s for s in shows_f.result() if s.get("browseId")]
            eps = [e for e in eps_f.result() if e.get("videoId")]

            # 후보: 유튜브 순서(인기/활동 반영)를 따르되 검색어와 겹치는 쇼만. 단, 상위 3개 안에 검색어 전체가
            # 제목에 들어간 쇼가 있으면 그것을 맨 앞으로. (점수만으로 정렬하면 오래전에 끝난 쇼가 뽑힌다)
            scores = [self._show_score(query, s.get("title", "")) for s in shows]
            order = [i for i in range(len(shows)) if scores[i] > 0]
            strong = [i for i in order if i < 3 and scores[i] >= 2]
            order = strong[:1] + [i for i in order if i not in strong[:1]]
            cands: list[tuple[str, str, bool]] = [(shows[i]["browseId"], shows[i].get("title", ""), True) for i in order]
            seen = {c[0] for c in cands}
            for e in eps:
                pod = e.get("podcast") or {}
                if pod.get("id") and pod["id"] not in seen and _relevant(query, [pod.get("name", ""), e.get("title", "")]):
                    cands.append((pod["id"], pod.get("name", ""), True))
                    seen.add(pod["id"])
            # 아무 제목도 안 맞으면 상위 쇼 두 개를 열어 진행자/설명까지 보고 판단한다
            cands += [(s["browseId"], s.get("title", ""), False) for s in shows[:2] if s["browseId"] not in seen]
            if not cands:
                raise MusicError(f"'{query}' 팟캐스트를 찾지 못했습니다.")

            pod = episodes = None
            pid = ""
            for pid, title, verified in cands[:4]:
                try:
                    data = ko.get_podcast(pid, limit=PODCAST_QUEUE + 5)
                except Exception:
                    log.warning("팟캐스트 불러오기 실패: %s", title, exc_info=True)
                    continue
                if not verified:
                    author = (data.get("author") or {}).get("name") or ""
                    if not _relevant(query, [data.get("title") or title, author, data.get("description") or ""]):
                        continue
                playable = [e for e in data.get("episodes") or [] if _playable_episode(e)]
                if playable:
                    pod, episodes = data, playable
                    break
            if not pod or not episodes:
                raise MusicError(f"'{query}' 팟캐스트를 찾지 못했습니다.")

            show = _clean_show(pod.get("title") or title)
            episodes = self._original_titles(en, pid, pod, episodes)
            # 쇼 페이지의 date 칸에는 지금 조회수가 들어와서, 업로드 날짜는 en 에피소드 검색 결과에서 얻는다
            dates = {e["videoId"]: e.get("date") for e in eps}
            latest = episodes[0]
            if latest["videoId"] not in dates and not _title_date(latest.get("title", "")):
                more = self._search_safe(en, latest.get("title", "")[:100], "episodes", 10)
                dates.update({e["videoId"]: e.get("date") for e in more if e.get("videoId")})

        tracks = [(e["videoId"], _clean_show(e.get("title", "")), show) for e in episodes[:PODCAST_QUEUE]]
        d = _episode_date(latest.get("title", ""), dates.get(latest["videoId"]))
        desc = self._episode_desc(f"'{show}' 팟캐스트의 최신 에피소드", latest.get("title", ""), show, d)
        if len(tracks) > 1:
            desc += f" 이전 에피소드 {len(tracks) - 1}개가 이어서 대기 중입니다."
        return tracks, desc

    @staticmethod
    def _original_titles(en, pid: str, pod: dict, episodes: list[dict]) -> list[dict]:
        """유튜브는 제목을 화면 언어로 자동 번역한다 (ko 는 영어 제목을 한국어로, en 은 한국어 제목을 영어로).
        쇼 정보에 한글이 없으면 영어 쇼로 보고 en 으로 다시 받아 원래 제목을 쓴다."""
        meta = " ".join([pod.get("title") or "", (pod.get("author") or {}).get("name") or "", pod.get("description") or ""])
        if re.search(r"[가-힣]", meta):
            return episodes
        try:
            data = en.get_podcast(pid, limit=len(episodes) + 5)
        except Exception:
            log.warning("영어 제목 불러오기 실패", exc_info=True)
            return episodes
        orig = {e.get("videoId"): e.get("title") for e in data.get("episodes") or []}
        return [{**e, "title": orig.get(e["videoId"]) or e.get("title")} for e in episodes]

    @staticmethod
    def _episode_desc(prefix: str, title: str, show: str, d: datetime.date | None) -> str:
        short = _short_episode(title, show)
        date_txt = ""
        if d and not _title_date(short) and not re.search(rf"(?<!\d){d.month}월\s*{d.day}일", short):
            date_txt = f"({_korean_date(d)})"
        return f"{prefix} '{short}'{date_txt}{_josa(short, '을', '를')} 재생합니다."

    def _podcast_episode_search(self, query: str) -> tuple[list[tuple[str, str, str]], str]:
        en, ko = self._clients()
        with self._search_lock, ThreadPoolExecutor(max_workers=3) as pool:
            hits = [h for h in self._search_safe(en, query, "episodes", 20)
                    if h.get("videoId") and h.get("isAvailable") is not False and _playable_episode(h)]
            if not hits:
                raise MusicError(f"'{query}'에 맞는 팟캐스트 에피소드를 찾지 못했습니다.")

            def ko_title(h: dict) -> str:
                try:
                    return ko.get_episode(h["videoId"]).get("title") or ""
                except Exception:
                    return ""

            # en 클라이언트는 한국어 제목을 영어로 번역해 주기도 해서, 안 맞으면 한국어 제목으로 다시 본다
            # 상위 5개 중 검색어와 가장 많이 겹치는 것 (같으면 유튜브 순서). "ai" 같은 짧은 토큰은 아무 데나 걸려서
            def score(h: dict) -> float:
                if not _relevant(query, [h.get("title", ""), (h.get("podcast") or {}).get("name", "")]):
                    return 0.0
                return max(self._show_score(query, h.get("title", "")), self._show_score(query, (h.get("podcast") or {}).get("name", "")), 0.1)

            scored = [(score(h), -i, h) for i, h in enumerate(hits[:5])]
            best = max(scored, key=lambda x: x[:2])
            chosen = best[2] if best[0] > 0 else None
            titles: dict[str, str] = {}
            if chosen is None:
                top = hits[:3]
                titles = dict(zip((h["videoId"] for h in top), pool.map(ko_title, top)))
                chosen = next((h for h in top if _relevant(query, [titles[h["videoId"]]])), None)
            if chosen is None:
                raise MusicError(f"'{query}'에 맞는 팟캐스트 에피소드를 찾지 못했습니다.")

            vid = chosen["videoId"]
            pod_ref = chosen.get("podcast") or {}
            show = _clean_show(pod_ref.get("name") or "")
            episodes: list[dict] = []
            if pod_ref.get("id"):
                try:
                    data = ko.get_podcast(pod_ref["id"], limit=60)
                    show = _clean_show(data.get("title") or show)
                    episodes = [e for e in data.get("episodes") or [] if _playable_episode(e)]
                    if any(e["videoId"] == vid for e in episodes):
                        episodes = self._original_titles(en, pod_ref["id"], data, episodes)
                except Exception:
                    log.warning("에피소드의 팟캐스트 불러오기 실패: %s", show, exc_info=True)
            idx = next((i for i, e in enumerate(episodes) if e["videoId"] == vid), None)
            if idx is not None:
                # 고른 에피소드부터, 이어서 그보다 오래된 에피소드들
                queue = episodes[idx: idx + PODCAST_QUEUE]
                tracks = [(e["videoId"], _clean_show(e.get("title", "")), show) for e in queue]
            else:
                # 한국어 쇼면 ko 제목, 아니면 en 검색 결과의 원래 제목
                title = (titles.get(vid) or ko_title(chosen)) if re.search(r"[가-힣]", show) else ""
                title = title or chosen.get("title", "")
                tracks = [(vid, _clean_show(title), show)]
                seen = {vid}
                for h in hits:  # 쇼 목록에 없으면 검색 결과 중 관련 있는 다른 에피소드를 뒤에
                    if len(tracks) >= PODCAST_QUEUE:
                        break
                    if h["videoId"] not in seen and _relevant(query, [h.get("title", ""), (h.get("podcast") or {}).get("name", "")]):
                        seen.add(h["videoId"])
                        tracks.append((h["videoId"], _clean_show(h.get("title", "")), _clean_show((h.get("podcast") or {}).get("name", ""))))

        title = tracks[0][1]
        d = _episode_date(title, chosen.get("date"))
        prefix = f"'{show}'의 에피소드" if show else "팟캐스트 에피소드"
        return tracks, self._episode_desc(prefix, title, show, d)

    def _loaded(self) -> bool:
        """새 재생 목록의 곡을 받아 재생 위치가 생겼는지. (일시정지 중이어도 True)

        곡을 바꾼 직후에는 이전 곡의 재생 위치가 잠깐 남아 있으므로, 지금 곡이 새 목록에 있는지도 본다.
        """
        with self._lock:
            pos = self._get("time-pos")
            vid = _video_id(self._get("path"))
            paused = self._get("pause") is True
        if not isinstance(pos, (int, float)) or vid not in self._meta:
            return False
        return pos > 0 or paused  # 재생 중이면 실제로 소리가 나기 시작했을 때(위치 > 0)를 기준으로

    def _wait_loaded(self, gen: int, t0: float) -> str | None:
        """곡이 실제로 불러와질 때까지 기다린다. 성공(또는 그사이 다른 곡으로 바뀜)이면 None, 실패하면 안내 문구.

        실패하면 재생을 멈춘다. 실패 안내는 부른 쪽에서 한다.
        """
        loaded_at = time.monotonic()
        deadline = t0 + START_TIMEOUT
        error = None
        try:
            while True:
                if gen != self._play_gen:
                    return None  # 그사이 다른 곡을 틀었거나 껐다
                with self._lock:
                    if not self._alive():
                        error = "음악 재생기가 종료되었어요."
                        break
                if self._loaded():
                    log.info("재생 시작까지 %.1fs", time.monotonic() - t0)
                    return None
                # mpv 는 목록의 곡을 모두 불러오지 못했을 때만 대기(idle) 상태로 돌아간다.
                if time.monotonic() - loaded_at > 1.0 and self._get("idle-active") is True:
                    error = "곡을 불러오지 못했어요. 잠시 후 다시 시도해 주세요."
                    break
                if time.monotonic() > deadline:
                    error = "곡을 불러오는 데 너무 오래 걸려서 멈췄어요. 잠시 후 다시 시도해 주세요."
                    break
                time.sleep(0.2)
            log.warning("재생 실패 (%.1f초): %s", time.monotonic() - t0, error)
            with self._lock:
                if gen == self._play_gen:
                    self._cmd("stop")
                    self._meta = {}
            return error
        finally:
            if gen == self._play_gen:
                self._starting = False

    def _watch_start(self, gen: int, t0: float) -> None:
        error = self._wait_loaded(gen, t0)
        if error and gen == self._play_gen and self.on_error:
            try:
                self.on_error(error)
            except Exception:
                log.exception("재생 실패 안내 중 오류")

    def is_starting(self) -> bool:
        """곡을 틀라고 했지만 아직 소리가 나기 전(스트림을 불러오는 중)인지."""
        return self._starting

    # ------------------------------------------------------------ 음량 페이드 (목소리와 교차할 때 부드럽게)

    def _fade(self, target: float, seconds: float, then=None, wait: bool = False) -> None:
        """mpv 음량을 seconds 동안 target 까지 부드럽게 바꾼다. 새 페이드나 set_volume 이 오면 그만둔다."""
        with self._lock:
            self._fade_gen += 1
            gen = self._fade_gen
            start = self._get("volume")
        if not isinstance(start, (int, float)):
            if then:
                then()
            return

        def run() -> None:
            steps = max(1, int(seconds / 0.04))
            for i in range(1, steps + 1):
                if gen != self._fade_gen:
                    return
                # 귀에 자연스럽도록 처음과 끝을 완만하게 (smoothstep)
                x = i / steps
                x = x * x * (3 - 2 * x)
                self._set("volume", round(start + (target - start) * x, 1))
                time.sleep(seconds / steps)
            if then and gen == self._fade_gen:
                then()

        if wait:
            run()
        else:
            threading.Thread(target=run, name="music-fade", daemon=True).start()

    def _restore_volume(self) -> None:
        """진행 중인 페이드를 멈추고 지금 있어야 할 음량으로 바로 맞춘다."""
        with self._lock:
            self._fade_gen += 1
            self._set("volume", self._current_target_volume())

    def pause(self, fade: bool = True) -> None:
        """일시정지. 기본은 0.35초 동안 소리를 줄인 뒤 멈춘다 (끝날 때까지 기다린다)."""
        if fade and self.is_playing():
            self._fade(0, 0.35, wait=True)
        self._set("pause", True)

    def resume(self) -> None:
        """다시 재생. 작은 소리에서 시작해 1초 동안 원래 음량으로 올린다."""
        was_paused = self._get("pause") is True
        if was_paused:
            self._set("volume", 0)
        self._set("pause", False)
        self._fade(self._current_target_volume(), 1.0 if was_paused else 0.3)

    def stop(self) -> None:
        self._play_gen += 1  # 불러오는 중이던 곡의 감시도 끝낸다
        self._starting = False
        self._cmd("stop")
        self._meta = {}

    def _skip(self, command: str) -> str | None:
        with self._lock:
            if not self._alive():
                return None
            try:
                self._ipc.command(command)  # type: ignore[union-attr]
            except (OSError, RuntimeError):  # 목록 끝/처음이면 실패
                return None
            self._set("pause", False)
            self._restore_volume()  # 일시정지 페이드로 소리가 0 이었을 수 있다
            pos = self._get("playlist-pos")
            if not isinstance(pos, int) or pos < 0:
                return None
            vid = _video_id(self._get(f"playlist/{pos}/filename"))
        if vid and vid in self._meta:
            title, artist = self._meta[vid]
            return f"{title} - {artist}" if artist else title
        return None

    def next(self) -> str | None:
        return self._skip("playlist-next")

    def previous(self) -> str | None:
        return self._skip("playlist-prev")

    def now_playing(self) -> str | None:
        with self._lock:
            if not self._alive() or self._get("idle-active") is not False:
                return None
            path = self._get("path")
            vid = _video_id(path)
            if vid and vid in self._meta:
                title, artist = self._meta[vid]
                return f"{title} - {artist}" if artist else title
            title = self._get("media-title")
            return title or None

    def now_playing_art(self) -> str | None:
        """지금 곡의 앨범·영상 이미지 주소 (유튜브 썸네일)."""
        with self._lock:
            if not self._alive() or self._get("idle-active") is not False:
                return None
            vid = _video_id(self._get("path"))
        return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else None

    def is_playing(self) -> bool:
        with self._lock:
            if not self._alive():
                return False
            if self._get("idle-active") is not False or self._get("pause") is not False:
                return False
            if self._get("core-idle") is not False:
                return False
            pos = self._get("time-pos")
            return isinstance(pos, (int, float)) and pos > 0

    def set_volume(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        with self._lock:
            self._fade_gen += 1  # 진행 중인 페이드보다 우선
            if self._pre_duck is not None:
                self._pre_duck = percent  # 덕킹 중이면 해제 시 적용
                self._volume = percent
                self._set("volume", min(self._duck_level, percent))
            else:
                self._volume = percent
                self._set("volume", percent)

    def get_volume(self) -> int:
        """설정된 음악 음량 (덕킹이나 페이드 중의 일시적인 값이 아니라 돌아갈 값)."""
        with self._lock:
            return self._pre_duck if self._pre_duck is not None else self._volume

    def duck(self, volume: int | None = None) -> None:
        """음악을 0.6초에 걸쳐 작게 (기본: 스피커가 말할 때의 크기 DUCK_VOLUME).

        이미 줄여 둔 상태에서 다시 부르면 그 크기로 옮겨 간다 (듣는 동안 더 작게 → 말하는 동안 조금 크게).
        """
        level = self.cfg.duck_volume if volume is None else volume
        with self._lock:
            if self._pre_duck is None:
                self._pre_duck = self._volume
            target = min(level, self._pre_duck)
            self._duck_level = target
        self._fade(target, 0.6)

    def unduck(self) -> None:
        """말이 끝나면 1.2초에 걸쳐 원래 음량으로."""
        with self._lock:
            if self._pre_duck is None:
                return
            self._volume, self._pre_duck = self._pre_duck, None
        self._fade(self._volume, 1.2)

    def shutdown(self) -> None:
        with self._lock:
            if self._alive():
                try:
                    self._ipc.command("quit", timeout=1)  # type: ignore[union-attr]
                except (OSError, RuntimeError):
                    pass
                try:
                    self._proc.wait(2)  # type: ignore[union-attr]
                except subprocess.TimeoutExpired:
                    pass
            self._kill_mpv()


# ---------------------------------------------------------------- 자가 테스트


def _self_test() -> None:  # pragma: no cover
    import sys

    from .config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    if cfg.mpv_ao != "null":
        print("안전을 위해 MPV_AO=null 환경에서만 실행합니다: MPV_AO=null uv run python -m speaker.music")
        sys.exit(1)
    p = MusicPlayer(cfg)
    ok = True

    def check(name: str, cond: bool, extra: Any = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK' if cond else 'FAIL'}] {name} {extra}")

    def wait_playing(timeout: float = 20) -> float | None:
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if p.is_playing():
                return time.monotonic() - t
            time.sleep(0.1)
        return None

    try:
        t = time.monotonic()
        desc = p.play("아이유 밤편지", wait=True)
        print(f"play() 반환까지 {time.monotonic() - t:.1f}s: {desc}")
        check("is_playing after play", p.is_playing())
        tp1 = p._get("time-pos")
        time.sleep(2)
        tp2 = p._get("time-pos")
        check("time-pos advances", tp1 is not None and tp2 is not None and tp2 > tp1, f"{tp1} -> {tp2}")
        check("now_playing", bool(p.now_playing()), p.now_playing())

        t = time.monotonic()
        nxt = p.next()
        dt = wait_playing()
        tp = p._get("time-pos")
        check("next() starts another track", dt is not None, f"next={nxt!r} 시작까지 {dt and round(time.monotonic() - t, 1)}s now={p.now_playing()!r} time-pos={tp}")
        prev = p.previous()
        dt = wait_playing()
        check("previous()", dt is not None, f"prev={prev!r} now={p.now_playing()!r}")

        p.pause()
        time.sleep(0.3)
        check("pause -> not playing", not p.is_playing())
        p.resume()
        check("resume -> playing", wait_playing(5) is not None)

        p.set_volume(60)
        p.duck()
        time.sleep(1.0)  # 0.6초 페이드
        ducked = p._get("volume")
        check("duck", ducked == min(cfg.duck_volume, 60), f"mpv volume={ducked} get_volume={p.get_volume()}")
        p.duck()  # 중복 호출 무해
        p.unduck()
        p.unduck()
        time.sleep(1.6)  # 1.2초 페이드
        check("unduck restores", p._get("volume") == 60 and p.get_volume() == 60, p._get("volume"))

        for q, kind in (("재즈", "playlist"), ("뉴진스", "artist")):
            t = time.monotonic()
            desc = p.play(q, kind=kind, wait=True)
            check(f"play({q!r}, {kind})", p.is_playing(), f"{time.monotonic() - t:.1f}s: {desc} | now={p.now_playing()!r}")

        # mpv 가 죽어도 다음 play() 에서 다시 띄우는지
        p._proc.kill()
        p._proc.wait()
        check("dead mpv -> not playing", not p.is_playing() and p.now_playing() is None)
        t = time.monotonic()
        desc = p.play("뉴진스 하입보이", wait=True)
        check("respawn after mpv death", p.is_playing(), f"{time.monotonic() - t:.1f}s: {desc}")

        p.stop()
        time.sleep(0.3)
        check("stop -> not playing", not p.is_playing() and p.now_playing() is None)
        check("next() after stop -> None", p.next() is None)

        try:
            p.play("qzxwvjkplmnbqzxwv zzqqxx", wait=True)
            check("nonsense raises MusicError", False)
        except MusicError as e:
            check("nonsense raises MusicError", True, str(e))
    finally:
        pid = p._proc.pid if p._proc else None
        p.shutdown()
        time.sleep(0.3)
        # 이 테스트가 띄운 mpv 만 본다 (따로 켜 둔 스피커의 mpv 는 제외)
        alive = subprocess.run(["pgrep", "-P", str(os.getpid()), "-x", "mpv"], capture_output=True, text=True).stdout.strip()
        check("shutdown leaves no mpv", not alive, f"pid={pid} pgrep={alive!r}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


def _podcast_self_test() -> None:  # pragma: no cover
    """MPV_AO=null uv run python -m speaker.music podcast"""
    import sys

    from .config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    if cfg.mpv_ao != "null":
        print("안전을 위해 MPV_AO=null 환경에서만 실행합니다: MPV_AO=null uv run python -m speaker.music podcast")
        sys.exit(1)
    p = MusicPlayer(cfg)
    ok = True

    def check(name: str, cond: bool, extra: Any = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK' if cond else 'FAIL'}] {name} {extra}")

    def wait_playing(timeout: float = 20) -> float | None:
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if p.is_playing():
                return time.monotonic() - t
            time.sleep(0.1)
        return None

    # 단위: 날짜 해석
    today = datetime.date(2026, 11, 15)
    check("_upload_date 2d ago", _upload_date("2d ago", today) == datetime.date(2026, 11, 13))
    check("_upload_date Nov 5", _upload_date("Nov 5", today) == datetime.date(2026, 11, 5))
    check("_upload_date Dec 20 -> last year", _upload_date("Dec 20", today) == datetime.date(2025, 12, 20))
    check("_upload_date Nov 1, 2025", _upload_date("Nov 1, 2025", today) == datetime.date(2025, 11, 1))
    check("_title_date", _title_date("뉴스공장 2026년 11월 13일 금요일") == datetime.date(2026, 11, 13)
          and _title_date("30분 만에 읽기 | 20261113") == datetime.date(2026, 11, 13))

    cases = [
        ("김어준의 겸손은힘들다", "latest"),
        ("KBS 뉴스", "latest"),
        ("뉴스 브리핑", "latest"),
        ("경제 팟캐스트", "latest"),
        ("The Daily", "latest"),
        ("tech news", "latest"),
        ("엔비디아 젠슨 황", "search"),
        ("AI agents", "search"),
    ]
    try:
        for q, ep in cases:
            t = time.monotonic()
            try:
                desc = p.play_podcast(q, episode=ep, wait=True)
            except MusicError as e:
                check(f"play_podcast({q!r}, {ep})", False, f"MusicError: {e}")
                continue
            dt = time.monotonic() - t
            playing = wait_playing(5) is not None
            count = p._get("playlist-count")
            now = p.now_playing()
            check(f"play_podcast({q!r}, {ep})", playing and bool(now),
                  f"\n    {dt:.1f}s 반환 | is_playing={playing} | 대기열={count}\n    desc: {desc}\n    now:  {now}")
            if isinstance(count, int) and count > 1:
                t = time.monotonic()
                nxt = p.next()
                started = wait_playing()
                pos = p._get("playlist-pos")
                check("  next() -> 이전 에피소드", started is not None and pos == 1 and nxt != now,
                      f"{time.monotonic() - t:.1f}s pos={pos} now={p.now_playing()!r}")

        for q, ep in (("qzxwvjkplmnbqzxwv zzqqxx", "latest"), ("qzxwvjkplmnbqzxwv zzqqxx", "search")):
            try:
                p.play_podcast(q, episode=ep, wait=True)
                check(f"nonsense ({ep}) raises MusicError", False)
            except MusicError as e:
                check(f"nonsense ({ep}) raises MusicError", True, str(e))
        p.stop()
    finally:
        p.shutdown()
        time.sleep(0.3)
        alive = subprocess.run(["pgrep", "-P", str(os.getpid()), "-x", "mpv"], capture_output=True, text=True).stdout.strip()
        check("shutdown leaves no mpv", not alive, f"pgrep={alive!r}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys as _sys

    if _sys.argv[1:2] == ["podcast"]:
        _podcast_self_test()
    else:
        _self_test()
