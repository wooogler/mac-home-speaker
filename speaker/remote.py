"""아이폰 브라우저용 리모컨: 같은 와이파이의 휴대폰에서 글로 명령하거나, 버튼을 누르고 말한다.

휴대폰이 원격 마이크 역할을 한다 (음악이 나올 때 맥 마이크는 잘 못 듣는다).
요청은 스피커에 말로 한 요청과 똑같이 처리되고 (handler 가 담당), 답은 휴대폰에 글로 보이며
원하면 스피커로도 말한다.

보안: 모든 요청에 비밀 토큰이 있어야 한다 (첫 접속은 ?t=..., 이후 쿠키 또는 X-Remote-Token 헤더).
아이폰 Safari 는 보안 연결(HTTPS)에서만 마이크를 허락하므로, 처음 실행할 때 openssl 로
이 맥 전용 인증 기관(CA)과 서버 인증서를 만든다. 휴대폰에 CA 를 한 번 설치·신뢰하면 된다 (/cert).

환경 변수 (나중에 config.py 로 옮길 것):
  REMOTE_PORT       기본 8443
  REMOTE_TOKEN      비우면 logs/remote_token 에 무작위 토큰을 만들어 저장 (권한 0600)
  REMOTE_CERT_DIR   기본 <프로젝트>/certs
  FFMPEG_PATH       기본: PATH 의 ffmpeg, 없으면 /opt/homebrew/bin/ffmpeg

자체 시험: uv run python -m speaker.remote --selftest
"""

from __future__ import annotations

import hmac
import html
import json
import logging
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

import numpy as np

log = logging.getLogger("speaker.remote")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OPENSSL = "/usr/bin/openssl" if Path("/usr/bin/openssl").exists() else (shutil.which("openssl") or "openssl")

SAMPLE_RATE = 16000
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_AUDIO_SECONDS = 30.5
MIN_AUDIO_SECONDS = 0.4
MAX_TEXT_BODY = 16 * 1024
MAX_TEXT_CHARS = 1000
COOKIE_NAME = "speaker_remote"

# 잘못된 토큰 시도 제한
FAIL_LIMIT = 5  # 이 횟수만큼 틀리면
FAIL_WINDOW = 600.0  # (이 시간 안에)
BLOCK_SECONDS = 900.0  # 이 시간 동안 차단
LEAF_DAYS = 825  # 애플 규칙: TLS 서버 인증서는 825일 이하
CA_DAYS = 3650


class RemoteHandler(Protocol):
    def remote_text(self, text: str, speak: bool) -> str: ...

    def remote_audio(self, pcm16k: np.ndarray, speak: bool) -> tuple[str, str]: ...

    def remote_status(self) -> dict: ...


class UserError(Exception):
    """휴대폰에 그대로 보여 줄 한국어 오류."""


# ---------------------------------------------------------------- 도우미


def lan_ip() -> str:
    """기본 경로의 LAN IP (패킷은 실제로 보내지 않는다)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def local_hostname() -> str:
    try:
        name = subprocess.run(
            ["scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    if not name:
        name = socket.gethostname().split(".")[0]
    return f"{name}.local"


def ffmpeg_path() -> str:
    env = os.environ.get("FFMPEG_PATH", "").strip()
    if env:
        return env
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def load_or_create_token(path: Path) -> str:
    try:
        token = path.read_text().strip()
        if len(token) >= 16:
            return token
    except OSError:
        pass
    token = secrets.token_urlsafe(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    os.chmod(path, 0o600)
    return token


def _openssl(*args: str) -> None:
    r = subprocess.run([OPENSSL, *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"openssl {args[0]} 실패: {r.stderr.strip()[-500:]}")


def ensure_certs(cert_dir: Path, ip: str, hostname: str) -> tuple[Path, Path, Path]:
    """CA(한 번만, 휴대폰에 설치) + 서버 인증서(IP/이름이 바뀌면 다시 만듦).

    반환: (서버 인증서+체인, 서버 키, CA 인증서 DER)
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cert_dir, 0o700)
    ca_key, ca_crt, ca_der = cert_dir / "ca.key", cert_dir / "ca.crt", cert_dir / "ca.der"
    key, crt, chain, san_file = (
        cert_dir / "server.key",
        cert_dir / "server.crt",
        cert_dir / "server-chain.pem",
        cert_dir / "server.san",
    )
    old_umask = os.umask(0o077)
    try:
        if not (ca_key.exists() and ca_crt.exists()):
            log.info("리모컨용 인증 기관(CA) 인증서를 만듭니다: %s", ca_crt)
            _openssl(
                "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes",
                "-keyout", str(ca_key), "-out", str(ca_crt), "-days", str(CA_DAYS),
                "-subj", f"/CN=Mac Speaker Local CA ({hostname})",
                "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            )
            for p in (crt, chain, san_file):
                p.unlink(missing_ok=True)
        if not ca_der.exists():
            _openssl("x509", "-in", str(ca_crt), "-outform", "der", "-out", str(ca_der))

        san = f"IP:{ip},IP:127.0.0.1,DNS:{hostname},DNS:localhost"
        expiring = False
        if crt.exists():
            r = subprocess.run(
                [OPENSSL, "x509", "-in", str(crt), "-noout", "-checkend", str(30 * 86400)],
                capture_output=True,
            )
            expiring = r.returncode != 0
        current_san = san_file.read_text().strip() if san_file.exists() else ""
        if expiring or current_san != san or not (key.exists() and chain.exists()):
            log.info("리모컨 서버 인증서를 만듭니다 (%s)", san)
            with tempfile.TemporaryDirectory() as td:
                csr, ext = Path(td) / "server.csr", Path(td) / "ext.cnf"
                ext.write_text(
                    f"subjectAltName={san}\n"
                    "extendedKeyUsage=serverAuth\n"
                    "keyUsage=critical,digitalSignature,keyEncipherment\n"
                    "basicConstraints=critical,CA:FALSE\n"
                )
                _openssl(
                    "req", "-new", "-newkey", "rsa:2048", "-sha256", "-nodes",
                    "-keyout", str(key), "-out", str(csr), "-subj", f"/CN={hostname}",
                )
                _openssl(
                    "x509", "-req", "-in", str(csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
                    "-set_serial", str(secrets.randbits(63)), "-sha256",
                    "-days", str(LEAF_DAYS), "-extfile", str(ext), "-out", str(crt),
                )
            chain.write_text(crt.read_text() + ca_crt.read_text())
            san_file.write_text(san + "\n")
        for p in (ca_key, key):
            os.chmod(p, 0o600)
    finally:
        os.umask(old_umask)
    return chain, key, ca_der


def decode_audio(data: bytes) -> np.ndarray:
    """MediaRecorder 결과(mp4/AAC, webm/opus, wav ...)를 16 kHz 모노 float32 로.

    mp4 는 moov 가 끝에 올 수 있어 파이프로는 못 읽으므로 임시 파일을 거친다.
    """
    with tempfile.NamedTemporaryFile(prefix="remote-", suffix=".bin") as f:
        f.write(data)
        f.flush()
        try:
            r = subprocess.run(
                [
                    ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", f.name, "-t", str(MAX_AUDIO_SECONDS + 1),
                    "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
                ],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("ffmpeg 실행 실패: %s", e)
            raise UserError("녹음을 변환하지 못했어요 (ffmpeg 문제).") from e
    if r.returncode != 0:
        log.warning("ffmpeg 변환 실패: %s", r.stderr.decode(errors="replace")[-300:])
        raise UserError("녹음 파일을 읽지 못했어요. 다시 해 볼래요?")
    pcm = np.frombuffer(r.stdout, dtype=np.float32).copy()
    seconds = len(pcm) / SAMPLE_RATE
    if seconds < MIN_AUDIO_SECONDS:
        raise UserError("녹음이 너무 짧아요. 버튼을 누른 채로 말해 주세요.")
    if seconds > MAX_AUDIO_SECONDS:
        raise UserError("녹음이 너무 길어요. 30초 안으로 말해 주세요.")
    if float(np.sqrt(np.mean(pcm**2))) < 1e-4:
        raise UserError("소리가 거의 안 들려요. 마이크를 확인해 주세요.")
    return pcm


class _FailTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, list[float]] = {}
        self._blocked: dict[str, float] = {}

    def blocked(self, ip: str) -> bool:
        with self._lock:
            until = self._blocked.get(ip)
            if until is None:
                return False
            if time.monotonic() >= until:
                del self._blocked[ip]
                return False
            return True

    def fail(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            times = [t for t in self._fails.get(ip, []) if now - t < FAIL_WINDOW]
            times.append(now)
            self._fails[ip] = times
            if len(times) >= FAIL_LIMIT:
                self._blocked[ip] = now + BLOCK_SECONDS
                self._fails.pop(ip, None)
                log.warning("리모컨: %s 에서 잘못된 토큰이 반복되어 %d분 차단합니다", ip, BLOCK_SECONDS // 60)

    def success(self, ip: str) -> None:
        with self._lock:
            self._fails.pop(ip, None)


# ---------------------------------------------------------------- HTTP


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    remote: RemoteServer

    def server_bind(self):
        # HTTPServer.server_bind 는 socket.getfqdn() 을 불러 이 맥에서 35초나 멈춘다 → 건너뛴다
        import socketserver

        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port

    def finish_request(self, request, client_address):  # TLS 악수는 요청 스레드에서
        request.settimeout(15)
        try:
            tls = self.remote._ssl_ctx.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError) as e:
            log.debug("리모컨 TLS 연결 실패 %s: %s", client_address[0], e)
            return
        try:
            self.RequestHandlerClass(tls, client_address, self)
        finally:
            try:
                tls.close()
            except OSError:
                pass

    def handle_error(self, request, client_address):
        log.debug("리모컨 요청 처리 중 연결 오류 (%s)", client_address[0], exc_info=True)


class _RequestHandler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"
    timeout = 60  # 소켓 읽기/쓰기 제한 (handler 처리 시간과는 무관)
    server_version = "MacSpeaker"
    sys_version = ""

    def log_message(self, format, *args):  # 토큰이 든 URL 을 로그에 남기지 않는다
        pass

    @property
    def remote(self) -> RemoteServer:
        return self.server.remote

    # -- 응답
    def _send(
        self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None, cache: str = "no-store"
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: dict, extra: dict[str, str] | None = None) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def _forbidden(self) -> None:
        self.close_connection = True
        # 작은 본문은 읽어 버린다: 안 읽고 끊으면 TCP RST 로 403 응답까지 사라질 수 있다
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if 0 < length <= MAX_TEXT_BODY:
            try:
                self.rfile.read(length)
            except OSError:
                pass
        self._send(403, "접근할 수 없어요.".encode(), "text/plain; charset=utf-8", {"Connection": "close"})

    # -- 인증
    def _auth(self) -> tuple[bool, dict[str, str] | None]:
        """(통과 여부, 응답에 붙일 쿠키 헤더).

        토큰을 아예 내지 않은 요청은 403 만 주고, 틀린 토큰을 낸 경우만 실패로 센다.
        """
        ip = self.client_address[0]
        tracker = self.remote._fails
        if tracker.blocked(ip):
            return False, None
        token = self.remote.token
        q = parse_qs(urlsplit(self.path).query).get("t", [""])[0]
        hdr = self.headers.get("X-Remote-Token", "")
        cookie = ""
        if self.headers.get("Cookie"):
            try:
                c = SimpleCookie(self.headers["Cookie"])
                if COOKIE_NAME in c:
                    cookie = c[COOKIE_NAME].value
            except Exception:
                cookie = ""
        presented = [v for v in (q, hdr, cookie) if v]
        if presented and all(hmac.compare_digest(v.encode(), token.encode()) for v in presented):
            tracker.success(ip)
            extra = None
            if q:
                extra = {
                    "Set-Cookie": f"{COOKIE_NAME}={token}; Max-Age=31536000; Path=/; Secure; HttpOnly; SameSite=Strict"
                }
            return True, extra
        if presented:
            tracker.fail(ip)
        time.sleep(self.remote.fail_delay)
        return False, None

    def _read_body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise UserError("요청 형식이 잘못됐어요.") from None
        if length < 0:
            raise UserError("요청 형식이 잘못됐어요.")
        if length > limit:
            self.close_connection = True
            if length <= 4 * limit:  # 조금 큰 정도면 읽어 버리고 친절한 413 을 보낸다
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            raise _TooLarge()
        return self.rfile.read(length)

    def _read_json(self) -> dict:
        body = self._read_body(MAX_TEXT_BODY)
        try:
            obj = json.loads(body or b"{}")
        except ValueError:
            raise UserError("요청 형식이 잘못됐어요.") from None
        if not isinstance(obj, dict):
            raise UserError("요청 형식이 잘못됐어요.")
        return obj

    def _direct(self, method: str, *args, wrap: str | None = None) -> None:
        """Claude 를 거치지 않는 직접 조작. 잠금 없이 바로 handler 를 부른다."""
        fn = getattr(self.remote.handler, method, None)
        if fn is None:
            return self._json(501, {"ok": False, "error": "이 기능은 아직 준비 중이에요."})
        result = fn(*args)
        if wrap:
            out = {"ok": True, wrap: result}
        elif isinstance(result, dict):
            out = dict(result)
            out["ok"] = bool(out.get("ok", True))
            if not out["ok"]:
                out.setdefault("error", out.get("message") or "실패했어요.")
        else:
            out = {"ok": True, "result": result}
        body = json.dumps(out, ensure_ascii=False, default=str).encode()
        self._send(200, body, "application/json; charset=utf-8")

    def _public(self, path: str) -> bool:
        """토큰 없이 주는 것: 아이콘·매니페스트 (개인 정보 없음). 홈 화면 아이콘은 쿠키 없이 받아 가기도 한다."""
        if path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png", "/icon-180.png", "/favicon.ico"):
            self._send(200, icon_png(180), "image/png", cache="public, max-age=86400")
        elif path == "/icon-512.png":
            self._send(200, icon_png(512), "image/png", cache="public, max-age=86400")
        elif path == "/manifest.json":
            self._send(200, MANIFEST_JSON.encode(), "application/manifest+json; charset=utf-8", cache="public, max-age=3600")
        else:
            return False
        return True

    # -- 라우팅
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urlsplit(self.path).path
        if self._public(path):
            return
        ok, cookie = self._auth()
        if not ok:
            return self._forbidden()
        try:
            if path == "/":
                self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8", cookie)
            elif path == "/api/status":
                status = self.remote.handler.remote_status()
                body = json.dumps({"ok": True, "status": status}, ensure_ascii=False, default=str).encode()
                self._send(200, body, "application/json; charset=utf-8", cookie)
            elif path == "/api/schedule":
                self._direct("remote_schedule", wrap="items")
            elif path == "/api/settings":
                self._direct("remote_settings", wrap="settings")
            elif path == "/cert":
                data = self.remote.ca_der.read_bytes()
                self._send(
                    200, data, "application/x-x509-ca-cert",
                    {"Content-Disposition": 'attachment; filename="mac-home-speaker-ca.crt"'},
                )
            else:
                self._json(404, {"ok": False, "error": "없는 주소예요."})
        except Exception:
            log.exception("리모컨 요청 처리 실패 (%s)", path)
            self._json(500, {"ok": False, "error": "처리하다가 문제가 생겼어요."})

    def do_DELETE(self):
        ok, _ = self._auth()
        if not ok:
            return self._forbidden()
        path = urlsplit(self.path).path
        try:
            if path.startswith("/api/schedule/") and len(path) > len("/api/schedule/"):
                from urllib.parse import unquote

                self._direct("remote_schedule_delete", unquote(path[len("/api/schedule/"):]))
            else:
                self._json(404, {"ok": False, "error": "없는 주소예요."})
        except Exception:
            log.exception("리모컨 삭제 처리 실패")
            self._json(500, {"ok": False, "error": "처리하다가 문제가 생겼어요."})

    def do_POST(self):
        ok, _ = self._auth()
        if not ok:
            # 본문을 읽지 않고 끊는다
            return self._forbidden()
        path = urlsplit(self.path).path
        speak = parse_qs(urlsplit(self.path).query).get("speak", ["1"])[0] not in ("0", "false", "")
        try:
            if path == "/api/text":
                obj = self._read_json()
                text = str(obj.get("text", "")).strip()
                speak = bool(obj.get("speak", speak))
                if not text:
                    raise UserError("할 말을 입력해 주세요.")
                if len(text) > MAX_TEXT_CHARS:
                    raise UserError("글이 너무 길어요.")
                log.info("리모컨 글 요청: %s", text)
                with self.remote._busy:
                    reply = self.remote.handler.remote_text(text, speak)
                self._json(200, {"ok": True, "heard": text, "reply": reply or ""})
            elif path == "/api/audio":
                data = self._read_body(MAX_UPLOAD_BYTES)
                if len(data) < 200:
                    raise UserError("녹음이 비어 있어요. 버튼을 누른 채로 말해 주세요.")
                pcm = decode_audio(data)
                log.info("리모컨 음성 요청: %.1f초", len(pcm) / SAMPLE_RATE)
                with self.remote._busy:
                    heard, reply = self.remote.handler.remote_audio(pcm, speak)
                self._json(200, {"ok": True, "heard": heard or "", "reply": reply or ""})
            elif path == "/api/schedule":
                obj = self._read_json()
                if not obj.get("kind"):
                    raise UserError("종류를 골라 주세요.")
                self._direct("remote_schedule_add", obj)
            elif path == "/api/schedule/update":
                obj = self._read_json()
                event_id = str(obj.pop("id", "") or "").strip()
                if not event_id:
                    raise UserError("고칠 항목을 골라 주세요.")
                if not obj:
                    raise UserError("바꿀 내용이 없어요.")
                self._direct("remote_schedule_update", event_id, obj)
            elif path == "/api/schedule/delete":
                event_id = str(self._read_json().get("id", "")).strip()
                if not event_id:
                    raise UserError("지울 항목을 골라 주세요.")
                self._direct("remote_schedule_delete", event_id)
            elif path == "/api/settings":
                obj = self._read_json()
                if not obj:
                    raise UserError("바꿀 설정이 없어요.")
                self._direct("remote_settings_set", obj)
            elif path == "/api/music":
                obj = self._read_json()
                action = str(obj.get("action", "")).strip()
                if not action:
                    raise UserError("무엇을 할지 골라 주세요.")
                query = obj.get("query")
                query = str(query).strip() or None if query is not None else None
                kind = str(obj.get("kind") or "song")
                if action == "play" and not query:
                    raise UserError("틀 곡이나 가수를 입력해 주세요.")
                self._direct("remote_music", action, query, kind)
            else:
                self._json(404, {"ok": False, "error": "없는 주소예요."})
        except _TooLarge:
            self._json(413, {"ok": False, "error": "보낸 내용이 너무 커요 (최대 2MB, 30초)."}, {"Connection": "close"})
        except UserError as e:
            self._json(400, {"ok": False, "error": str(e)})
        except Exception:
            log.exception("리모컨 요청 처리 실패 (%s)", path)
            self._json(500, {"ok": False, "error": "처리하다가 문제가 생겼어요."})

class _TooLarge(Exception):
    pass


class RemoteServer:
    def __init__(
        self,
        handler: RemoteHandler,
        port: int = 8443,
        token: str | None = None,
        cert_dir: Path | None = None,
    ) -> None:
        self.handler = handler
        self.port = port
        self.token = (
            token
            or os.environ.get("REMOTE_TOKEN", "").strip()
            or load_or_create_token(PROJECT_DIR / "logs" / "remote_token")
        )
        env_dir = os.environ.get("REMOTE_CERT_DIR", "").strip()
        self.cert_dir = cert_dir or (Path(env_dir).expanduser() if env_dir else PROJECT_DIR / "certs")
        self.ip = lan_ip()
        self.hostname = local_hostname()
        self.fail_delay = 1.0
        self._fails = _FailTracker()
        self._busy = threading.Lock()  # Claude 경로(글·음성)만 한 번에 하나씩. 직접 조작은 잠그지 않는다
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self._ssl_ctx: ssl.SSLContext | None = None
        self.ca_der: Path = self.cert_dir / "ca.der"

    @property
    def url(self) -> str:
        return f"https://{self.ip}:{self.port}/?t={self.token}"

    @property
    def cert_url(self) -> str:
        return f"https://{self.ip}:{self.port}/cert?t={self.token}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        chain, key, self.ca_der = ensure_certs(self.cert_dir, self.ip, self.hostname)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(str(chain), str(key))
        self._ssl_ctx = ctx
        httpd = _Server(("0.0.0.0", self.port), _RequestHandler)
        httpd.remote = self
        self.port = httpd.server_address[1]  # port=0 이면 실제 포트
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, name="remote-http", daemon=True)
        self._thread.start()
        # 토큰이 든 주소는 시작할 때 이 한 줄만 남긴다
        log.info("휴대폰 리모컨: %s  (처음이면 인증서 설치: %s)", self.url, self.cert_url)

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("휴대폰 리모컨을 껐습니다")


# ---------------------------------------------------------------- 홈 화면 아이콘


def _png_bytes(rgb: np.ndarray) -> bytes:
    """(h, w, 3) uint8 → PNG (표준 라이브러리만)."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


_ICON_CACHE: dict[int, bytes] = {}
_ICON_LOCK = threading.Lock()


def icon_png(size: int) -> bytes:
    """남색 바탕에 흰 스피커와 소리 물결. iOS 가 모서리를 둥글게 깎으므로 바탕은 꽉 채운다."""
    with _ICON_LOCK:
        if size in _ICON_CACHE:
            return _ICON_CACHE[size]
        ss = 4  # 4배로 그린 뒤 줄여서 부드러운 가장자리
        n = size * ss
        c = (np.arange(n) + 0.5) / n
        x, y = np.meshgrid(c, c)
        dy = y - 0.5
        body = (x >= 0.19) & (x <= 0.33) & (np.abs(dy) <= 0.095)
        cone = (x >= 0.31) & (x <= 0.49) & (np.abs(dy) <= 0.095 + (x - 0.31) / 0.18 * 0.13)
        mask = body | cone
        r = np.hypot(x - 0.47, dy)
        ang = np.arctan2(dy, x - 0.47)
        lim = np.radians(42)
        half = 0.024
        for rad in (0.15, 0.235, 0.32):
            mask |= (np.abs(r - rad) <= half) & (np.abs(ang) <= lim)
            for s in (-1, 1):  # 둥근 끝
                ex, ey = 0.47 + rad * np.cos(lim), 0.5 + s * rad * np.sin(lim)
                mask |= np.hypot(x - ex, y - ey) <= half
        alpha = mask.reshape(size, ss, size, ss).mean(axis=(1, 3))[..., None]
        t = ((np.arange(size) + 0.5) / size)[:, None, None]
        top, bottom = np.array([32, 52, 104.0]), np.array([12, 22, 52.0])
        bg = top * (1 - t) + bottom * t
        img = bg * (1 - alpha) + 255.0 * alpha
        img = np.broadcast_to(img, (size, size, 3))
        data = _png_bytes(np.clip(img + 0.5, 0, 255).astype(np.uint8))
        _ICON_CACHE[size] = data
        return data


# start_url 은 일부러 뺀다: 없으면 매니페스트를 연결한 문서 주소(?t= 포함)가 시작 주소가 되고,
# 매니페스트 자체는 공개라 토큰을 넣을 수 없다.
MANIFEST_JSON = json.dumps(
    {
        "name": "스피커 리모컨",
        "short_name": "스피커",
        "display": "standalone",
        "background_color": "#111418",
        "theme_color": "#14213d",
        "icons": [
            {"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    },
    ensure_ascii=False,
)


# ---------------------------------------------------------------- 페이지

PAGE_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="스피커">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="color-scheme" content="dark light">
<meta name="theme-color" content="#0d1015" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#f2f3f7" media="(prefers-color-scheme: light)">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" href="/icon-180.png">
<link rel="manifest" href="/manifest.json">
<title>스피커</title>
<style>
:root{
  --bg:#0d1015;--surface:#161a21;--surface-2:#1e232c;--surface-3:#272d38;
  --line:rgba(255,255,255,.07);--fg:#f1f3f6;--fg-2:#c3c9d2;--muted:#8a93a1;
  --accent:#6b8cff;--accent-fg:#fff;--accent-soft:rgba(107,140,255,.16);
  --danger:#ff6b6b;--danger-soft:rgba(255,107,107,.14);--ok:#3ddc84;--warn:#ffb547;
  --bubble-me:#3d5bd9;--bubble-me-fg:#fff;--bubble-bot:#1e232c;
  --shadow:0 1px 0 rgba(255,255,255,.04) inset,0 8px 24px rgba(0,0,0,.28);
  --bar:rgba(16,19,25,.82);
  --r-lg:20px;--r-md:14px;--r-sm:10px;
  --s1:4px;--s2:8px;--s3:12px;--s4:16px;--s5:24px;--s6:32px;
  --tabbar:58px;--maxw:560px;
  --sat:env(safe-area-inset-top);--sab:env(safe-area-inset-bottom);
}
@media (prefers-color-scheme:light){:root{
  --bg:#f2f3f7;--surface:#fff;--surface-2:#f1f2f6;--surface-3:#e6e8ee;
  --line:rgba(17,20,24,.08);--fg:#111418;--fg-2:#3b424d;--muted:#6b7380;
  --accent:#3a63f0;--accent-soft:rgba(58,99,240,.1);--danger:#e5484d;--danger-soft:rgba(229,72,77,.1);--ok:#1fa45a;--warn:#c27c0e;
  --bubble-me:#3a63f0;--bubble-bot:#fff;
  --shadow:0 1px 2px rgba(17,20,24,.05),0 6px 20px rgba(17,20,24,.06);
  --bar:rgba(248,249,251,.86);
}}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
[hidden]{display:none!important}
html,body{overflow:hidden;overscroll-behavior:none;touch-action:manipulation}
body{position:fixed;top:0;right:0;bottom:0;left:0}
/* 홈 화면 앱 보정 (html.fixh, JS 가 붙임): iOS standalone + black-translucent 에서는 fixed 기준 상자가
   화면보다 상단 안전 영역만큼 짧아 bottom:0 이 위로 뜬다. body 를 실제 화면 높이(--app-h)로 만들고
   겹쳐 놓는 요소들을 body 기준 absolute 로 바꿔 진짜 바닥에 붙인다. 일반 Safari 에는 적용하지 않는다. */
html.fixh,html.fixh body{height:var(--app-h)}
html.fixh body{bottom:auto}
html.fixh .app,html.fixh nav.tabbar,html.fixh .dock,html.fixh .toast,html.fixh .banner,html.fixh .scrim,html.fixh .sheet{position:absolute}
#saProbe{position:absolute;visibility:hidden;pointer-events:none;top:0;left:0;width:0;height:0;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
#diag{margin:0;font:12px/1.6 ui-monospace,Menlo,monospace;white-space:pre-wrap;color:var(--fg-2)}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text","Apple SD Gothic Neo","Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;-webkit-text-size-adjust:100%;overscroll-behavior-y:none}
button,input,select{font:inherit;color:inherit}
button{cursor:pointer;touch-action:manipulation}
button:disabled{opacity:.45;cursor:default}
svg.i{width:24px;height:24px;flex:none;fill:none;stroke:currentColor;stroke-width:1.9;stroke-linecap:round;stroke-linejoin:round}
svg.f{fill:currentColor;stroke:none}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}

/* ---- 틀 ---- */
.app{position:fixed;top:0;right:0;bottom:0;left:0;overflow-y:auto;overflow-x:hidden;-webkit-overflow-scrolling:touch;overscroll-behavior:contain;--gut:max(var(--s4),calc((100% - var(--maxw)) / 2 + var(--s4)));padding:calc(var(--sat) + 8px) var(--gut) calc(var(--tabbar) + var(--sab) + 32px)}
.sheet-open .app{overflow:hidden}
.app.home{padding-bottom:calc(var(--tabbar) + var(--sab) + 150px)}
header.top{display:flex;align-items:center;justify-content:space-between;min-height:52px;margin-bottom:var(--s3)}
header.top h1{font-size:30px;line-height:1.1;font-weight:750;letter-spacing:-.02em;margin:0}
.conn{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--muted);padding:6px 10px;border-radius:99px;background:var(--surface)}
.conn i{width:8px;height:8px;border-radius:50%;background:var(--ok)}
.conn.offline i{background:var(--warn)}.conn.denied i{background:var(--danger)}
section.tab{display:flex;flex-direction:column;gap:var(--s3)}
section.tab[hidden]{display:none}
.card{background:var(--surface);border-radius:var(--r-lg);padding:var(--s4);box-shadow:var(--shadow)}
.eyebrow{font-size:12px;font-weight:600;letter-spacing:.04em;color:var(--muted)}
.muted{color:var(--muted)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap;flex:none;min-height:44px;padding:0 16px;border-radius:12px;border:0;background:var(--surface-2);font-weight:600;font-size:16px}
.btn:active{filter:brightness(1.12)}
.btn.primary{background:var(--accent);color:var(--accent-fg)}
.btn.ghost{background:transparent;color:var(--accent);padding:0 8px}
.btn.block{width:100%;min-height:50px;font-size:17px;border-radius:14px}
.btn.busy{position:relative;color:transparent!important}
.btn.busy::after{content:"";position:absolute;width:18px;height:18px;border-radius:50%;border:2px solid var(--accent-fg);border-right-color:transparent;animation:spin .7s linear infinite}
.btn:not(.primary).busy::after{border-color:var(--fg);border-right-color:transparent}
@keyframes spin{to{transform:rotate(360deg)}}
.icon-btn{width:44px;height:44px;padding:0;line-height:0;display:inline-flex;align-items:center;justify-content:center;border:0;border-radius:12px;background:transparent;color:var(--fg-2)}
.icon-btn:active{background:var(--surface-2)}
.icon-btn.danger{color:var(--danger)}
.icon-btn.round{border-radius:50%;background:var(--accent-soft);color:var(--accent)}
input[type=text],input[type=number],input[type=time],input[type=date],select{font-size:17px;min-height:46px;padding:0 14px;border-radius:12px;border:1px solid var(--line);background:var(--surface-2);width:100%;min-width:0;-webkit-appearance:none;appearance:none}
input[type=time],input[type=date]{display:flex;align-items:center}
select{padding-right:38px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238a93a1' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m7 10 5 5 5-5'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;background-size:18px}
input:focus,select:focus{outline:2px solid var(--accent);outline-offset:-1px}
input::placeholder{color:var(--muted)}
.chips{display:flex;gap:var(--s2);overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch;padding:2px 0}
.chips::-webkit-scrollbar{display:none}
.chip{flex:none;min-height:40px;padding:0 14px;border-radius:99px;border:1px solid var(--line);background:var(--surface);color:var(--fg-2);font-size:15px;font-weight:500;white-space:nowrap}
.chip:active,.chip.on{background:var(--accent-soft);color:var(--accent);border-color:transparent}
.seg{display:flex;padding:3px;border-radius:12px;background:var(--surface-2);gap:2px}
.seg button{flex:1;min-height:38px;border:0;border-radius:9px;background:transparent;font-size:15px;font-weight:600;color:var(--fg-2)}
.seg button.on{background:var(--surface-3);color:var(--fg);box-shadow:0 1px 3px rgba(0,0,0,.18)}
@media (prefers-color-scheme:light){.seg button.on{background:#fff}}

/* ---- 지금 재생 ---- */
.np{display:flex;align-items:center;gap:var(--s3)}
.cover{flex:none;width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;color:#fff;background:linear-gradient(135deg,#6b8cff,#9b6bff 55%,#ff7ab6)}
.cover.idle{background:var(--surface-2);color:var(--muted)}
.cover{position:relative;overflow:hidden}
.cover img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scale(1.34);opacity:0;transition:opacity .3s}
.cover.art{background:#000}.cover.art img{opacity:1}.cover.art svg{visibility:hidden}
.np .txt{flex:1;min-width:0}
.np .title{font-size:17px;font-weight:650;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere}
.np .artist{font-size:14px;color:var(--fg-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:1px}
.np .state{font-size:13px;color:var(--muted);margin-top:2px}
.transport{display:grid;grid-template-columns:repeat(5,1fr);justify-items:center;align-items:center;margin:var(--s3) auto 0;width:100%;max-width:340px}
.tbtn.side{width:44px;height:44px;color:var(--fg-2)}.tbtn.side svg{width:21px;height:21px}
.tbtn.side.on{color:var(--accent);background:var(--accent-soft)}
.volrow{display:flex;align-items:center;gap:var(--s3);margin:var(--s2) auto 0;width:100%;max-width:340px;padding:0 var(--s2)}
.volrow svg{width:20px;height:20px;color:var(--muted)}
.volrow output{min-width:28px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;font-size:15px}
.tbtn{width:52px;height:52px;border:0;border-radius:50%;background:transparent;display:flex;align-items:center;justify-content:center;color:var(--fg);padding:0;margin:0;line-height:0;-webkit-appearance:none;appearance:none}
.tbtn svg,.mic svg,.icon-btn svg,.send svg,.spk svg{display:block}
.tbtn:active{background:var(--surface-2)}
.tbtn.main{width:60px;height:60px;background:var(--fg);color:var(--bg)}
.tbtn.main:active{transform:scale(.96)}
.tbtn svg{width:26px;height:26px}.tbtn.main svg{width:28px;height:28px}

/* ---- 다음 알람 ---- */
.next{display:flex;align-items:center;gap:var(--s3);width:100%;border:0;text-align:left;font:inherit;color:inherit}
.next .ic{width:44px;height:44px;border-radius:12px;background:var(--accent-soft);color:var(--accent);display:flex;align-items:center;justify-content:center;flex:none}
.next .txt{flex:1;min-width:0}
.next .big{font-size:22px;font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums;line-height:1.2}
.next .big small{font-size:14px;font-weight:600;color:var(--fg-2);margin-right:4px}
.next .sub{font-size:13px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chev{color:var(--muted)}

/* ---- 대화 ---- */
.chat{display:flex;flex-direction:column;gap:6px;padding:var(--s2) 0}
.chat .empty{text-align:center;padding:var(--s5) var(--s4);color:var(--muted);font-size:15px}
.chat .empty b{display:block;color:var(--fg-2);font-size:16px;margin-bottom:4px}
.bubble{max-width:82%;padding:9px 14px;border-radius:20px;font-size:16px;line-height:1.4;white-space:pre-wrap;word-break:break-word;animation:pop .18s ease-out}
.bubble.me{align-self:flex-end;background:var(--bubble-me);color:var(--bubble-me-fg);border-bottom-right-radius:6px}
.bubble.bot{align-self:flex-start;background:var(--bubble-bot);border-bottom-left-radius:6px;box-shadow:var(--shadow)}
.bubble.err{align-self:center;background:var(--danger-soft);color:var(--danger);font-size:14px;border-radius:12px;max-width:92%;text-align:center}
.bubble.wait{color:var(--muted)}
.bubble.me.wait{color:rgba(255,255,255,.75)}
@keyframes pop{from{opacity:0;transform:translateY(6px)}}
.dots{display:inline-flex;gap:4px;vertical-align:middle;padding:4px 0}
.dots i{width:7px;height:7px;border-radius:50%;background:currentColor;opacity:.4;animation:blink 1.2s infinite}
.dots i:nth-child(2){animation-delay:.2s}.dots i:nth-child(3){animation-delay:.4s}
@keyframes blink{30%{opacity:1;transform:translateY(-2px)}}
.chat-head{display:flex;align-items:center;justify-content:space-between;margin-top:var(--s2)}
.chat-head h2{font-size:20px;margin:0;font-weight:700}

/* ---- 입력 도크 (홈) ---- */
.dock{position:fixed;left:0;right:0;bottom:calc(var(--tabbar) + var(--sab));z-index:15;pointer-events:none}
.dock-in{max-width:var(--maxw);margin:0 auto;padding:var(--s2) var(--s4) var(--s3);pointer-events:auto;background:linear-gradient(to top,var(--bg) 70%,transparent)}
.dock[hidden]{display:none}
.dock .chips{margin-bottom:var(--s2)}
.dock-row{display:flex;align-items:center;gap:var(--s2)}
.field-wrap{flex:1;min-width:0;position:relative;height:52px}
.field-wrap input{height:52px;border-radius:26px;padding:0 50px 0 18px;background:var(--surface);border-color:var(--line);box-shadow:var(--shadow)}
.field-wrap .send{position:absolute;right:4px;top:4px;width:44px;height:44px;border-radius:50%;border:0;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center}
.field-wrap .send:disabled{background:var(--surface-3);color:var(--muted);opacity:1}
.recbar{position:absolute;inset:0;border-radius:26px;background:var(--danger-soft);color:var(--danger);display:none;align-items:center;gap:10px;padding:0 18px;font-weight:600;font-variant-numeric:tabular-nums}
.recbar .dot{width:10px;height:10px;border-radius:50%;background:var(--danger);animation:blink 1s infinite}
.recbar .hint{font-weight:500;font-size:14px;opacity:.85;margin-left:auto;white-space:nowrap}
.recording .recbar{display:flex}
.spk{flex:none;height:52px;min-width:52px;border-radius:26px;border:0;background:var(--surface);color:var(--muted);display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:10px;font-weight:600;gap:1px;box-shadow:var(--shadow)}
.spk svg{width:22px;height:22px}
.spk.on{color:var(--accent)}
.mic{flex:none;width:64px;height:64px;border-radius:50%;border:0;padding:0;line-height:0;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 20px rgba(80,110,255,.45);user-select:none;-webkit-user-select:none;-webkit-touch-callout:none;touch-action:none;transition:transform .12s,background .2s}
.mic svg{width:30px;height:30px;pointer-events:none}
.mic:active{transform:scale(.95)}
.recording .mic{background:var(--danger);box-shadow:0 0 0 0 rgba(255,107,107,.6);animation:ring 1.4s infinite}
.mic.busy{opacity:.7}
@keyframes ring{70%{box-shadow:0 0 0 16px rgba(255,107,107,0)}100%{box-shadow:0 0 0 0 rgba(255,107,107,0)}}
.mic .stop{display:none}.recording .mic .stop{display:block}.recording .mic .mi{display:none}

/* ---- 알람 ---- */
.group h3{font-size:13px;font-weight:600;color:var(--muted);margin:var(--s3) var(--s2) var(--s2);letter-spacing:.02em}
.list{list-style:none;margin:0;padding:0;background:var(--surface);border-radius:var(--r-lg);box-shadow:var(--shadow);overflow:hidden}
.list>li{display:flex;align-items:center;gap:var(--s3);padding:var(--s3) var(--s2) var(--s3) var(--s4);min-height:64px;position:relative}
.list>li.edit{cursor:pointer}.list>li.edit:active{background:var(--surface-2)}
.list>li .when,.list>li .meta{transition:opacity .2s}.list>li.off .when,.list>li.off .meta{opacity:.4}
.list>li .sw{margin-right:var(--s2)}
.list>li+li::before{content:"";position:absolute;top:0;left:var(--s4);right:0;height:1px;background:var(--line)}
.when{flex:none;min-width:96px;font-variant-numeric:tabular-nums}
.when .hm{font-size:30px;font-weight:300;letter-spacing:-.02em;line-height:1}
.when .ampm{font-size:14px;color:var(--fg-2);margin-right:3px;font-weight:500}
.meta{flex:1;min-width:0}
.meta .t{font-size:16px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.meta .s{font-size:13px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.empty-state{text-align:center;padding:var(--s6) var(--s4);color:var(--muted)}
.empty-state .ic{width:56px;height:56px;border-radius:16px;background:var(--surface);margin:0 auto var(--s3);display:flex;align-items:center;justify-content:center;color:var(--muted)}
.empty-state b{display:block;color:var(--fg);font-size:17px;margin-bottom:4px}

/* ---- 시트 ---- */
.scrim{position:fixed;inset:0;background:rgba(0,0,0,.45);opacity:0;pointer-events:none;transition:opacity .25s;z-index:30}
.sheet{position:fixed;left:0;right:0;bottom:0;z-index:31;max-width:var(--maxw);margin:0 auto;background:var(--surface);border-radius:22px 22px 0 0;transform:translateY(105%);transition:transform .3s cubic-bezier(.2,.8,.2,1);max-height:calc(100% - var(--sat) - 24px);display:flex;flex-direction:column;box-shadow:0 -10px 40px rgba(0,0,0,.35);overscroll-behavior:contain}
.sheet-open .scrim{opacity:1;pointer-events:auto}
.sheet-open .sheet{transform:none}
.sheet .grab{width:38px;height:5px;border-radius:3px;background:var(--surface-3);margin:8px auto 0}
.sheet-head{display:flex;align-items:center;justify-content:space-between;padding:var(--s2) var(--s2) var(--s2) var(--s4)}
.sheet-head h2{font-size:19px;margin:0}
.sheet-body{overflow-y:auto;padding:0 var(--s4) var(--s4);-webkit-overflow-scrolling:touch;overscroll-behavior:contain;min-height:0;flex:1 1 auto}
.sheet-foot{display:flex;gap:var(--s2);padding:var(--s3) var(--s4) calc(var(--s3) + var(--sab));border-top:1px solid var(--line)}
.sheet-foot .block{flex:1}
.btn.danger-soft{background:var(--danger-soft);color:var(--danger);min-height:50px;border-radius:14px;padding:0 20px}
.fld{display:flex;flex-direction:column;gap:6px;margin-top:var(--s4)}
.fld>span{font-size:13px;font-weight:600;color:var(--muted)}
.f2{display:grid;grid-template-columns:1fr 1fr;gap:var(--s3)}
.check{display:flex;align-items:center;justify-content:space-between;gap:var(--s3);min-height:48px;margin-top:var(--s3)}
.sw{position:relative;width:51px;height:31px;flex:none}
.sw input{position:absolute;opacity:0;inset:0;margin:0;z-index:1;cursor:pointer}
.sw span{position:absolute;inset:0;border-radius:99px;background:var(--surface-3);transition:background .2s}
.sw span::after{content:"";position:absolute;top:2px;left:2px;width:27px;height:27px;border-radius:50%;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,.25);transition:transform .2s}
.sw input:checked+span{background:var(--ok)}
.sw input:checked+span::after{transform:translateX(20px)}

/* ---- 음악 ---- */
.hero{display:flex;flex-direction:column;align-items:center;text-align:center;padding:var(--s5) var(--s4)}
.hero .cover{width:min(46vw,190px);height:min(46vw,190px);border-radius:28px;box-shadow:0 18px 40px rgba(90,100,255,.3)}
.hero .cover svg{width:72px;height:72px}
.hero .cover.idle{box-shadow:none}
.hero .title{font-size:21px;font-weight:700;margin-top:var(--s4);line-height:1.3;min-height:28px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;max-width:100%;overflow-wrap:anywhere}
.hero .artist{font-size:16px;color:var(--fg-2);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
.hero .state{color:var(--muted);font-size:14px;margin-top:2px}
.hero .transport{margin-top:var(--s4)}
.hero .tbtn.main{width:72px;height:72px}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:var(--s2)}
.tile{border:0;border-radius:var(--r-md);background:var(--surface);box-shadow:var(--shadow);min-height:88px;display:flex;flex-direction:column;align-items:flex-start;justify-content:space-between;padding:var(--s3);font-size:15px;font-weight:600;text-align:left}
.tile .ic{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center}
.tile:active{transform:scale(.98)}
.tile.busy,.tbtn.busy{opacity:.5}
.t-like .ic{background:rgba(255,95,135,.16);color:#ff5f87}
.t-mix .ic{background:rgba(155,107,255,.18);color:#a98bff}
.t-recent .ic{background:rgba(61,220,132,.15);color:var(--ok)}
.sec-title{font-size:20px;font-weight:700;margin:var(--s3) 0 0}
.search{display:flex;gap:var(--s2);margin-top:var(--s3)}

/* ---- 설정 ---- */
.rows{background:var(--surface);border-radius:var(--r-lg);box-shadow:var(--shadow);overflow:hidden}
.row{display:flex;align-items:center;gap:var(--s3);padding:0 var(--s2) 0 var(--s4);min-height:52px;position:relative}
.row+.row::before,.slider+.slider::before{content:"";position:absolute;top:0;left:var(--s4);right:0;height:1px;background:var(--line)}
.row .lbl{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}
.row .val{color:var(--muted);font-variant-numeric:tabular-nums;padding-right:var(--s2)}
.slider{padding:var(--s3) var(--s4) var(--s2);position:relative}
.slider .top{display:flex;justify-content:space-between;align-items:baseline}
.slider .top output{color:var(--muted);font-variant-numeric:tabular-nums;font-size:15px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:36px;background:transparent;margin:0}
input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:linear-gradient(to right,var(--accent) var(--p,50%),var(--surface-3) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:28px;height:28px;margin-top:-11px;border-radius:50%;background:#fff;box-shadow:0 2px 6px rgba(0,0,0,.3)}
input[type=range]::-moz-range-track{height:6px;border-radius:3px;background:var(--surface-3)}
input[type=range]::-moz-range-progress{height:6px;border-radius:3px;background:var(--accent)}
input[type=range]::-moz-range-thumb{width:26px;height:26px;border-radius:50%;background:#fff;border:0}
input[type=range]:disabled{opacity:.4}
.badge{font-size:12px;color:var(--muted);background:var(--surface-2);padding:3px 8px;border-radius:99px;margin-right:var(--s2);white-space:nowrap}
.addrow{display:flex;gap:var(--s2);padding:var(--s2) var(--s2) var(--s2) var(--s4)}
.groupt{font-size:13px;font-weight:600;color:var(--muted);margin:var(--s4) var(--s4) var(--s2);letter-spacing:.02em}
.foot{font-size:13px;color:var(--muted);margin:var(--s2) var(--s4) 0;line-height:1.5}
details.rows summary{list-style:none;display:flex;align-items:center;min-height:52px;padding:0 var(--s4);cursor:pointer}
details.rows summary::-webkit-details-marker{display:none}
details.rows summary .chev{margin-left:auto;transition:transform .2s}
details[open].rows summary .chev{transform:rotate(90deg)}
details.rows .inner{padding:0 var(--s4) var(--s4);font-size:15px;color:var(--fg-2);line-height:1.6}
details.rows ol{margin:0;padding-left:20px}
dl.kv{margin:0;display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:14px}
dl.kv dt{color:var(--muted)}dl.kv dd{margin:0;word-break:break-word}
dl.kv ul{margin:0;padding-left:18px}
a{color:var(--accent)}

/* ---- 탭 막대 ---- */
nav.tabbar{position:fixed;left:0;right:0;bottom:0;top:auto;z-index:20;background:var(--bar);-webkit-backdrop-filter:saturate(180%) blur(20px);backdrop-filter:saturate(180%) blur(20px);border-top:1px solid var(--line);padding-bottom:var(--sab)}
nav.tabbar .in{max-width:var(--maxw);margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);height:var(--tabbar)}
nav.tabbar button{border:0;background:transparent;color:var(--muted);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;font-size:11px;font-weight:600}
nav.tabbar button svg{width:26px;height:26px}
nav.tabbar button.on{color:var(--accent)}

/* ---- 알림 ---- */
.toast{position:fixed;left:50%;bottom:calc(var(--tabbar) + var(--sab) + 16px);transform:translate(-50%,12px);max-width:calc(100% - 32px);width:max-content;background:rgba(30,35,44,.96);color:#fff;padding:12px 18px;border-radius:14px;font-size:15px;font-weight:500;box-shadow:0 10px 30px rgba(0,0,0,.35);opacity:0;pointer-events:none;transition:opacity .2s,transform .2s;z-index:40;text-align:center}
.home-active .toast{bottom:calc(var(--tabbar) + var(--sab) + 150px)}
.toast.show{opacity:1;transform:translate(-50%,0)}
.toast.bad{background:rgba(200,45,50,.96)}
.banner{position:fixed;top:calc(var(--sat) + 8px);left:50%;transform:translate(-50%,-140%);z-index:35;max-width:calc(100% - 32px);width:max-content;padding:10px 16px;border-radius:99px;font-size:14px;font-weight:600;background:var(--warn);color:#1a1300;transition:transform .25s;box-shadow:0 6px 20px rgba(0,0,0,.3)}
.banner.denied{background:var(--danger);color:#fff}
.banner:not(.show){visibility:hidden;box-shadow:none}
.banner.show{transform:translate(-50%,0)}
@media (min-width:700px){.app{padding-top:32px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="app home" id="app">
  <header class="top">
    <h1 id="title">홈</h1>
    <div id="hdr-home" class="conn" role="status"><i></i><span id="connText">연결 중</span></div>
    <button id="hdr-add" class="icon-btn round" type="button" aria-label="알람 추가" hidden><svg class="i" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></button>
  </header>

  <!-- 홈 -->
  <section class="tab" id="tab-home">
    <div class="card">
      <div class="np">
        <div class="cover idle" id="hCover"><svg class="i" viewBox="0 0 24 24"><path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/></svg></div>
        <div class="txt"><div class="eyebrow">지금 재생</div><div class="title" id="hTitle">-</div><div class="artist" id="hArtist" hidden></div></div>
      </div>
      <div class="transport">
        <button class="tbtn side" type="button" data-act="stop" aria-label="끄기"><svg class="i f" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg></button>
        <button class="tbtn" type="button" data-act="previous" aria-label="이전 곡"><svg class="i f" viewBox="0 0 24 24"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg></button>
        <button class="tbtn main" type="button" data-act="toggle" aria-label="재생/일시정지"><svg class="i f pp" viewBox="0 0 24 24"><path d="M7.5 4v16l12-8z"/></svg></button>
        <button class="tbtn" type="button" data-act="next" aria-label="다음 곡"><svg class="i f" viewBox="0 0 24 24"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg></button>
        <button class="tbtn side" type="button" data-vol="h" aria-label="음악 음량" aria-expanded="false"><svg class="i" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16.5 9a4 4 0 0 1 0 6M19 6.5a7.5 7.5 0 0 1 0 11"/></svg></button>
      </div>
      <div class="volrow" id="volrow-h" hidden><svg class="i" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/></svg><input type="range" class="volr" min="0" max="100" step="1" aria-label="음악 음량"><output>-</output></div>
    </div>
    <div class="card" style="padding:12px 12px 12px 16px">
      <button class="next" id="nextCard" type="button" style="background:none;padding:0">
        <div class="ic"><svg class="i" viewBox="0 0 24 24"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M5 3 2 6M19 3l3 3"/></svg></div>
        <div class="txt"><div class="eyebrow" id="nextKind">다음 알람</div><div class="big" id="nextBig">-</div><div class="sub" id="nextSub">&nbsp;</div></div>
        <svg class="i chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>
      </button>
    </div>
    <div class="chat-head"><h2>대화</h2><button class="btn ghost" type="button" id="clearChat">지우기</button></div>
    <div class="chat" id="chat"></div>
  </section>

  <!-- 알람 -->
  <section class="tab" id="tab-alarm" hidden>
    <div id="alarmGroups"></div>
  </section>

  <!-- 음악 -->
  <section class="tab" id="tab-music" hidden>
    <div class="card hero">
      <div class="cover idle" id="mCover"><svg class="i" viewBox="0 0 24 24"><path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/></svg></div>
      <div class="title" id="mTitle">-</div>
      <div class="artist" id="mArtist" hidden></div>
      <div class="state" id="mState">&nbsp;</div>
      <div class="transport">
        <button class="tbtn side" type="button" data-act="stop" aria-label="끄기"><svg class="i f" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg></button>
        <button class="tbtn" type="button" data-act="previous" aria-label="이전 곡"><svg class="i f" viewBox="0 0 24 24"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg></button>
        <button class="tbtn main" type="button" data-act="toggle" aria-label="재생/일시정지"><svg class="i f pp" viewBox="0 0 24 24"><path d="M7.5 4v16l12-8z"/></svg></button>
        <button class="tbtn" type="button" data-act="next" aria-label="다음 곡"><svg class="i f" viewBox="0 0 24 24"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg></button>
        <button class="tbtn side" type="button" data-vol="m" aria-label="음악 음량" aria-expanded="false"><svg class="i" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16.5 9a4 4 0 0 1 0 6M19 6.5a7.5 7.5 0 0 1 0 11"/></svg></button>
      </div>
      <div class="volrow" id="volrow-m" hidden><svg class="i" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/></svg><input type="range" class="volr" min="0" max="100" step="1" aria-label="음악 음량"><output>-</output></div>
    </div>
    <h2 class="sec-title">내 음악</h2>
    <div class="tiles">
      <button class="tile t-like" type="button" data-act="liked"><span class="ic"><svg class="i" viewBox="0 0 24 24"><path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.5-7 10-7 10z"/></svg></span>좋아요한 곡</button>
      <button class="tile t-mix" type="button" data-act="mix"><span class="ic"><svg class="i" viewBox="0 0 24 24"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/></svg></span>맞춤 추천</button>
      <button class="tile t-recent" type="button" data-act="recent"><span class="ic"><svg class="i" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M12 7v5l3 2"/></svg></span>최근 들은 곡</button>
    </div>
    <h2 class="sec-title">찾아서 틀기</h2>
    <div class="card">
      <div class="seg" id="mKind"><button type="button" data-v="song" class="on">곡</button><button type="button" data-v="artist">가수</button><button type="button" data-v="playlist">분위기</button></div>
      <form class="search" id="musicForm" autocomplete="off">
        <input type="text" id="mQuery" placeholder="예: 아이유 밤편지" enterkeyhint="go" aria-label="검색어">
        <button class="btn primary" type="submit" id="mPlay">틀기</button>
      </form>
    </div>
  </section>

  <!-- 설정 -->
  <section class="tab" id="tab-set" hidden>
    <div class="groupt" style="margin-top:0">소리</div>
    <div class="rows">
      <div class="slider"><div class="top"><span>전체 음량</span><output id="o_system_volume">-</output></div><input type="range" id="s_system_volume" min="0" max="100" step="1" aria-label="전체 음량"></div>
      <div class="slider"><div class="top"><span>음악 음량</span><output id="o_music_volume">-</output></div><input type="range" id="s_music_volume" min="0" max="100" step="1" aria-label="음악 음량"></div>
      <div class="slider"><div class="top"><span>목소리 음량</span><output id="o_voice_volume">-</output></div><input type="range" id="s_voice_volume" min="0" max="100" step="1" aria-label="목소리 음량"></div>
      <div class="slider"><div class="top"><span>말 속도</span><output id="o_voice_speed">-</output></div><input type="range" id="s_voice_speed" min="0.7" max="1.6" step="0.05" aria-label="말 속도"></div>
    </div>
    <div class="groupt">호출어</div>
    <div class="rows" id="wakeRows"><div class="row"><span class="lbl muted">불러오는 중…</span></div></div>
    <form class="rows addrow" id="wakeForm" autocomplete="off" style="margin-top:8px">
      <input type="text" id="wakeAdd" placeholder="새 호출어 (예: 스피커야)" aria-label="새 호출어">
      <button class="btn primary" type="submit" id="wakeBtn">추가</button>
    </form>
    <div class="foot">부르는 말로 스피커를 깨워요. 영어 호출어는 고정이라 여기서 바꿀 수 없어요.</div>
    <div class="groupt">뉴스</div>
    <div class="rows">
      <div class="slider"><div class="top"><span>뉴스 개수</span><output id="o_news_count">-</output></div><input type="range" id="s_news_count" min="1" max="10" step="1" aria-label="뉴스 개수"></div>
    </div>
    <div class="groupt">뉴스 관심 분야</div>
    <div class="rows" id="newsRows"><div class="row"><span class="lbl muted">불러오는 중…</span></div></div>
    <form class="rows addrow" id="newsForm" autocomplete="off" style="margin-top:8px">
      <input type="text" id="newsAdd" placeholder="새 분야 (예: 인공지능)" aria-label="새 뉴스 관심 분야">
      <button class="btn primary" type="submit" id="newsBtn">추가</button>
    </form>
    <div class="foot">뉴스 브리핑과 아침 알람 브리핑에서 이 분야 소식을 먼저 챙겨요. 비워 두면 한국과 세계의 주요 뉴스를 전해요.</div>
    <div class="groupt">맥</div>
    <div class="rows"><button class="btn block" type="button" id="displayOff">맥 화면 끄기</button></div>
    <div class="foot">스피커는 계속 동작해요. 맥의 키보드나 트랙패드를 건드리면 다시 켜져요.</div>
    <div class="groupt">정보</div>
    <div class="rows">
      <div class="row"><span class="lbl">연결 상태</span><span class="val" id="infoConn">-</span></div>
      <div class="row"><span class="lbl">이어 말하기</span><span class="val" id="infoFollow">-</span></div>
    </div>
    <details class="rows" style="margin-top:8px"><summary>마이크·인증서 도움말<svg class="i chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg></summary>
      <div class="inner">아이폰은 믿을 수 있는 보안 연결(HTTPS)에서만 마이크를 허락해요.
        <ol><li><a id="certlink" href="/cert">이 맥의 인증서 받기</a> → 허용</li>
        <li>설정 › 일반 › VPN 및 기기 관리 › 받은 프로파일 설치</li>
        <li>설정 › 일반 › 정보 › 인증서 신뢰 설정 › 'Mac Speaker Local CA' 켜기</li>
        <li>앱을 닫았다가 다시 열기. 설정 › Safari › 마이크 가 '거부'면 '묻기'로</li></ol></div>
    </details>
    <details class="rows" style="margin-top:8px" id="diagDetails"><summary>화면 정보<svg class="i chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg></summary>
      <div class="inner"><pre id="diag">-</pre></div></details>
    <details class="rows" style="margin-top:8px" id="statusDetails"><summary>스피커 상태 자세히<svg class="i chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg></summary>
      <div class="inner" id="statusBox">-</div></details>
  </section>
</div>

<!-- 홈 입력 도크 -->
<div class="dock" id="dock">
  <div class="dock-in" id="dockIn">
    <div class="chips" id="chips">
      <button class="chip" type="button">음악 꺼줘</button>
      <button class="chip" type="button">다음 곡</button>
      <button class="chip" type="button">소리 줄여줘</button>
      <button class="chip" type="button">오늘 일정 알려줘</button>
      <button class="chip" type="button">내일 날씨 어때?</button>
    </div>
    <div class="dock-row">
      <button class="spk on" id="spk" type="button" aria-pressed="true" aria-label="스피커로도 말하기">
        <svg class="i" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path class="w" d="M16.5 9a4 4 0 0 1 0 6M19 6.5a7.5 7.5 0 0 1 0 11"/><path class="x" d="M17 9l5 6M22 9l-5 6" style="display:none"/></svg><span id="spkTxt">스피커</span>
      </button>
      <form class="field-wrap" id="f" autocomplete="off">
        <input id="text" type="text" placeholder="말 대신 입력하기" enterkeyhint="send" aria-label="명령 입력">
        <button class="send" type="submit" id="sendBtn" aria-label="보내기" disabled><svg class="i" viewBox="0 0 24 24" style="width:20px;height:20px"><path d="M12 19V5M6 11l6-6 6 6"/></svg></button>
        <div class="recbar" aria-live="polite"><span class="dot"></span><span id="recTime">0.0초</span><span class="hint" id="recHint">손을 떼면 보내요</span></div>
      </form>
      <button class="mic" id="talk" type="button" aria-label="누르고 말하기">
        <svg class="i mi" viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>
        <svg class="i f stop" viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>
      </button>
    </div>
  </div>
</div>

<!-- 추가 시트 -->
<div class="scrim" id="scrim"></div>
<div class="sheet" id="sheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle">
  <div class="grab"></div>
  <div class="sheet-head"><h2 id="sheetTitle">새로 추가</h2><button class="icon-btn" type="button" id="sheetClose" aria-label="닫기"><svg class="i" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg></button></div>
  <div class="sheet-body">
    <div class="seg" id="kindSeg">
      <button type="button" data-v="alarm">알람</button><button type="button" data-v="reminder">리마인더</button><button type="button" data-v="timer">타이머</button><button type="button" data-v="sleep">수면</button>
    </div>
    <form id="schedForm" autocomplete="off">
      <div data-for="alarm">
        <label class="fld"><span>시각</span><input type="time" id="aTime"></label>
        <div class="chips" style="margin-top:8px" data-preset="aTime"><button type="button" class="chip" data-v="06:30">오전 6:30</button><button type="button" class="chip" data-v="07:00">오전 7:00</button><button type="button" class="chip" data-v="07:30">오전 7:30</button><button type="button" class="chip" data-v="08:00">오전 8:00</button></div>
        <div class="f2">
          <label class="fld"><span>반복</span><select id="aRepeat"><option value="none">한 번</option><option value="daily">매일</option><option value="weekdays">평일</option><option value="weekends">주말</option></select></label>
          <label class="fld"><span>날짜 (한 번일 때)</span><input type="date" id="aDate"></label>
        </div>
        <label class="fld"><span>이름</span><input type="text" id="aLabel" placeholder="예: 출근 (선택)"></label>
        <label class="fld"><span>깨우는 방법</span><select id="aMusic"><option value="">알림음</option><option value="liked">좋아요한 곡</option><option value="mix">맞춤 추천</option><option value="custom">직접 입력…</option></select></label>
        <label class="fld" id="aMusicTextWrap" hidden><span>틀 곡·가수·분위기</span><input type="text" id="aMusicText" placeholder="예: 잔잔한 피아노"></label>
        <label class="check"><span>울린 뒤 날씨·뉴스 브리핑</span><span class="sw"><input type="checkbox" id="aBrief"><span></span></span></label>
      </div>
      <div data-for="reminder" hidden>
        <label class="fld"><span>알려 줄 내용</span><input type="text" id="rMsg" placeholder="예: 약 먹기"></label>
        <div class="fld"><span>언제</span><div class="seg" id="rModeSeg"><button type="button" data-v="in">몇 분 뒤</button><button type="button" data-v="at">시각 정하기</button></div></div>
        <div id="rIn">
          <div class="chips" style="margin-top:12px" data-preset="rMin"><button type="button" class="chip" data-v="10">10분 뒤</button><button type="button" class="chip" data-v="30">30분 뒤</button><button type="button" class="chip" data-v="60">1시간 뒤</button><button type="button" class="chip" data-v="120">2시간 뒤</button></div>
          <label class="fld"><span>몇 분 뒤</span><input type="number" id="rMin" min="1" max="1440" inputmode="numeric" placeholder="30"></label>
        </div>
        <div id="rAt" hidden>
          <div class="f2">
            <label class="fld"><span>시각</span><input type="time" id="rTime"></label>
            <label class="fld"><span>날짜</span><input type="date" id="rDate"></label>
          </div>
          <label class="fld"><span>반복</span><select id="rRepeat"><option value="none">한 번</option><option value="daily">매일</option><option value="weekdays">평일</option><option value="weekends">주말</option></select></label>
        </div>
      </div>
      <div data-for="timer" hidden>
        <div class="chips" style="margin-top:16px" data-preset="tMin"><button type="button" class="chip" data-v="1">1분</button><button type="button" class="chip" data-v="3">3분</button><button type="button" class="chip" data-v="5">5분</button><button type="button" class="chip" data-v="10">10분</button><button type="button" class="chip" data-v="15">15분</button><button type="button" class="chip" data-v="30">30분</button></div>
        <label class="fld"><span>몇 분</span><input type="number" id="tMin" min="0.1" max="1440" step="any" inputmode="decimal" placeholder="5"></label>
        <label class="fld"><span>이름</span><input type="text" id="tLabel" placeholder="예: 라면 (선택)"></label>
      </div>
      <div data-for="sleep" hidden>
        <p class="muted" style="margin:16px 0 0;font-size:15px">정한 시간이 지나면 음악을 꺼요.</p>
        <div class="chips" style="margin-top:12px" data-preset="sMin"><button type="button" class="chip" data-v="15">15분</button><button type="button" class="chip" data-v="30">30분</button><button type="button" class="chip" data-v="45">45분</button><button type="button" class="chip" data-v="60">1시간</button><button type="button" class="chip" data-v="0">수면 타이머 끄기</button></div>
        <label class="fld"><span>몇 분 뒤 끄기 (0 = 끄기)</span><input type="number" id="sMin" min="0" max="600" inputmode="numeric" placeholder="30"></label>
      </div>
    </form>
  </div>
  <div class="sheet-foot"><button class="btn danger-soft" type="button" id="schedDelete" hidden>삭제</button><button class="btn primary block" type="submit" form="schedForm" id="schedSubmit">추가</button></div>
</div>

<nav class="tabbar" aria-label="탭"><div class="in">
  <button type="button" data-tab="home"><svg class="i" viewBox="0 0 24 24"><path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/></svg>홈</button>
  <button type="button" data-tab="alarm"><svg class="i" viewBox="0 0 24 24"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M5 3 2 6M19 3l3 3"/></svg>알람</button>
  <button type="button" data-tab="music"><svg class="i" viewBox="0 0 24 24"><path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/></svg>음악</button>
  <button type="button" data-tab="set"><svg class="i" viewBox="0 0 24 24"><path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/></svg>설정</button>
</div></nav>
<div class="banner" id="banner" role="alert"></div>
<div class="toast" id="toast" role="status" aria-live="polite"></div>
<div id="saProbe" aria-hidden="true"></div>

<script>
(function(){
"use strict";
const $=id=>document.getElementById(id);
/* 확대 막기: iOS Safari 는 user-scalable=no 를 무시하므로 제스처를 직접 막는다 */
["gesturestart","gesturechange","gestureend"].forEach(ev=>document.addEventListener(ev,e=>e.preventDefault(),{passive:false}));
document.addEventListener("touchmove",e=>{if(e.touches&&e.touches.length>1)e.preventDefault()},{passive:false});
document.addEventListener("dblclick",e=>e.preventDefault(),{passive:false});
let lastTouchEnd=0;
document.addEventListener("touchend",e=>{
  const now=Date.now(),t=e.target;
  const interactive=t&&t.closest&&t.closest("button,input,select,textarea,label,a,[role=button]");
  if(!interactive&&now-lastTouchEnd<350&&e.cancelable)e.preventDefault();  /* 빈 곳 두 번 톡 → 확대 방지 */
  lastTouchEnd=now;
},{passive:false});
/* 시트가 열린 동안 뒤 배경은 스크롤되지 않게 */
document.addEventListener("touchmove",e=>{
  if(!document.body.classList.contains("sheet-open"))return;
  const body=document.querySelector(".sheet-body");
  if(!(body&&body.contains(e.target))&&e.cancelable)e.preventDefault();
},{passive:false});
const $$=(sel,root)=>Array.prototype.slice.call((root||document).querySelectorAll(sel));
function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
let TOKEN=null;try{TOKEN=new URLSearchParams(location.search).get("t")}catch(e){}
if(TOKEN)$("certlink").href="/cert?t="+encodeURIComponent(TOKEN);
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}

/* ---------- 알림·연결 ---------- */
let toastTimer=null;
function toast(msg,bad){
  const t=$("toast");t.textContent=msg;t.className="toast show"+(bad?" bad":"");
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>{t.className="toast"+(bad?" bad":"")},bad?4200:2600);
}
let conn="connecting";
function setConn(c){
  if(c===conn)return;conn=c;
  const txt={ok:"연결됨",offline:"연결 끊김",denied:"접근 거부",connecting:"연결 중"}[c];
  $("connText").textContent=txt;$("hdr-home").className="conn "+c;$("infoConn").textContent=txt;
  const b=$("banner");
  if(c==="offline"){b.textContent="스피커에 연결할 수 없어요 · 같은 와이파이인지 확인해 주세요";b.className="banner show"}
  else if(c==="denied"){b.textContent="접근할 수 없어요 · 처음 받은 주소로 다시 열어 주세요";b.className="banner denied show"}
  else b.className="banner";
}
function hdrs(h){h=h||{};if(TOKEN)h["X-Remote-Token"]=TOKEN;return h}
const NET_ERR="연결이 끊겼어요. 같은 와이파이인지, 스피커가 켜져 있는지 확인해 주세요.";
/* 항상 {ok,...} 로 끝난다 */
function api(method,url,body,timeoutMs,raw){
  const ctl=window.AbortController?new AbortController():null;
  const timer=setTimeout(()=>{if(ctl)ctl.abort()},timeoutMs||20000);
  const opts={method:method,credentials:"same-origin",cache:"no-store",headers:hdrs()};
  if(raw){opts.body=raw.body;opts.headers["Content-Type"]=raw.type}
  else if(body!==undefined){opts.body=JSON.stringify(body);opts.headers["Content-Type"]="application/json"}
  if(ctl)opts.signal=ctl.signal;
  return fetch(url,opts).then(r=>{
    setConn(r.status===403?"denied":"ok");
    return r.json().catch(()=>({ok:false,error:r.status===403?"접근할 수 없어요. 처음 받은 주소(토큰 포함)로 다시 열어 주세요.":"서버 응답이 이상해요 ("+r.status+")."}));
  }).catch(e=>{
    if(e&&e.name==="AbortError")return {ok:false,error:"답이 너무 오래 걸려요. 잠시 후 다시 해 주세요."};
    setConn("offline");return {ok:false,error:NET_ERR};
  }).then(j=>{clearTimeout(timer);return j||{ok:false,error:"실패했어요."}});
}
function report(j){toast(j.message||j.error||(j.ok?"됐어요.":"실패했어요."),!j.ok);return j}
function busy(btn,on){if(!btn)return;btn.disabled=on;btn.classList.toggle("busy",on)}

/* ---------- 탭 ---------- */
const TABS={home:"홈",alarm:"알람",music:"음악",set:"설정"};
let cur=null;
function showTab(name){
  if(!TABS[name])name="home";
  cur=name;store("tab",name);
  Object.keys(TABS).forEach(t=>{$("tab-"+t).hidden=t!==name});
  $$("nav.tabbar button").forEach(b=>b.classList.toggle("on",b.dataset.tab===name));
  $("title").textContent=TABS[name];
  $("hdr-home").hidden=name==="alarm";$("hdr-add").hidden=name!=="alarm";
  $("dock").hidden=name!=="home";
  $("app").classList.toggle("home",name==="home");
  document.body.classList.toggle("home-active",name==="home");
  const sc=$("app");sc.scrollTop=name==="home"?sc.scrollHeight:0;
  if(name==="alarm"||name==="home")loadSchedule();
  if(name==="set")loadSettings();
  if(name!=="alarm")refresh();
}
$$("nav.tabbar button").forEach(b=>b.addEventListener("click",()=>showTab(b.dataset.tab)));

/* ---------- 상태·지금 재생 ---------- */
let nowPlaying=null, paused=false, art=null;
function setArt(){
  const url=(typeof art==="string"&&art.indexOf("https://")===0)?art:null;
  ["hCover","mCover"].forEach(id=>{
    const c=$(id);let img=c.querySelector("img");
    if(!url){if(img)img.remove();c.classList.remove("art");return}
    if(img&&img.dataset.src===url)return;
    if(!img){img=document.createElement("img");img.alt="";img.decoding="async";img.referrerPolicy="no-referrer";c.appendChild(img)}
    img.dataset.src=url;c.classList.remove("art");
    img.onload=()=>{if(img.naturalWidth>1)c.classList.add("art");else img.onerror()};
    img.onerror=()=>{img.remove();c.classList.remove("art")};
    img.src=url;
  });
}
function findKey(o,re,depth,wantObj){
  if(!o||typeof o!=="object"||depth>3)return undefined;
  for(const k of Object.keys(o)){if(re.test(k.toLowerCase())){const v=o[k];if(v===null||typeof v!=="object")return v}}
  for(const k of Object.keys(o)){const r=findKey(o[k],re,depth+1);if(r!==undefined)return r}
  return undefined;
}
const PP_PLAY='<path d="M7.5 4v16l12-8z"/>', PP_PAUSE='<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>';
function renderNP(){
  const has=!!(nowPlaying&&String(nowPlaying).trim());
  const title=has?String(nowPlaying):"재생 중인 음악이 없어요";
  const state=has?(paused?"일시정지됨":"재생 중"):"아래에서 골라 틀어 보세요";
  const i=has?title.lastIndexOf(" - "):-1;
  const name=i>0?title.slice(0,i):title, artist=i>0?title.slice(i+3):"";
  $("hTitle").textContent=name;$("mTitle").textContent=name;$("mState").textContent=state;
  [$("hArtist"),$("mArtist")].forEach(e=>{e.textContent=artist;e.hidden=!artist});
  $("hTitle").title=$("mTitle").title=title;
  $("hCover").classList.toggle("idle",!has);$("mCover").classList.toggle("idle",!has);
  $$("svg.pp").forEach(s=>{s.innerHTML=(has&&!paused)?PP_PAUSE:PP_PLAY});
  $$("[data-act=toggle]").forEach(b=>b.setAttribute("aria-label",(has&&!paused)?"일시정지":"다시 재생"));
}
function render(v){
  if(v===null||v===undefined||v==="")return document.createTextNode("-");
  if(Array.isArray(v)){
    if(!v.length)return document.createTextNode("없음");
    const ul=el("ul");v.forEach(x=>{const li=el("li");li.appendChild(render(x));ul.appendChild(li)});return ul;
  }
  if(typeof v==="object"){
    const dl=el("dl","kv");
    Object.keys(v).forEach(k=>{dl.appendChild(el("dt","",k));const dd=el("dd");dd.appendChild(render(v[k]));dl.appendChild(dd)});
    return dl;
  }
  if(typeof v==="boolean")return document.createTextNode(v?"예":"아니요");
  return document.createTextNode(String(v));
}
function refresh(){
  return api("GET","/api/status").then(j=>{
    if(!j.ok)return;
    const np=findKey(j.status,/now_?playing|playing|재생/,0);
    if(np!==undefined){nowPlaying=(np===false)?null:np}
    const p=findKey(j.status,/paused|일시정지/,0);
    if(typeof p==="boolean")paused=p;
    const a=findKey(j.status,/^(art|artwork|thumbnail)$/,0);
    if(a!==undefined)art=a;
    renderNP();setArt();
    const box=$("statusBox");box.innerHTML="";box.appendChild(render(j.status));
  });
}

/* ---------- 음악 조작 ---------- */
function music(action,query,kind,btn){
  if(action==="toggle")action=(nowPlaying&&!paused)?"pause":"resume";
  busy(btn,true);
  const body={action:action};if(query)body.query=query;if(kind)body.kind=kind;
  return api("POST","/api/music",body,30000).then(j=>{
    busy(btn,false);report(j);
    if(j.ok){
      if(action==="pause")paused=true;else if(["resume","play","liked","mix","recent","next","previous"].indexOf(action)>=0)paused=false;
      if(action==="stop"){nowPlaying=null;art=null;setArt()}
      if("now_playing" in j)nowPlaying=j.now_playing;
      renderNP();setTimeout(refresh,1500);
    }
    return j;
  });
}
$$("[data-act]").forEach(b=>b.addEventListener("click",()=>music(b.dataset.act,null,null,b)));
/* 음악 음량 (지금 재생 카드 안에서 바로) */
let volT=null;
function setVolRows(v){
  if(v===null||v===undefined)return;
  $$(".volr").forEach(r=>{if(r!==document.activeElement&&!volT){r.value=v;paintEl(r);r.nextElementSibling.textContent=Math.round(v)}});
}
$$("[data-vol]").forEach(b=>b.addEventListener("click",()=>{
  const row=$("volrow-"+b.dataset.vol),open=row.hidden;
  row.hidden=!open;b.classList.toggle("on",open);b.setAttribute("aria-expanded",String(open));
  if(open)api("GET","/api/settings").then(j=>{if(j.ok&&j.settings)setVolRows(j.settings.music_volume)});
}));
$$(".volr").forEach(r=>r.addEventListener("input",()=>{
  r.nextElementSibling.textContent=r.value;paintEl(r);
  clearTimeout(volT);
  volT=setTimeout(()=>{volT=null;const v=parseInt(r.value,10);saveSettings({music_volume:v},true).then(j=>{if(j.ok)toast("음악 음량 "+v)})},400);
}));
let mKind="song";
$$("#mKind button").forEach(b=>b.addEventListener("click",()=>{mKind=b.dataset.v;$$("#mKind button").forEach(x=>x.classList.toggle("on",x===b))}));
$("musicForm").addEventListener("submit",e=>{
  e.preventDefault();const q=$("mQuery").value.trim();
  if(!q){toast("무엇을 틀지 입력해 주세요.",true);return}
  $("mQuery").blur();music("play",q,mKind,$("mPlay"));
});

/* ---------- 대화 ---------- */
const chatEl=$("chat");
let chat=[];try{chat=JSON.parse(store("chat")||"[]")}catch(e){chat=[]}
function saveChat(){chat=chat.slice(-40);store("chat",JSON.stringify(chat))}
function scrollChat(){if(cur==="home")requestAnimationFrame(()=>{const sc=$("app");sc.scrollTo({top:sc.scrollHeight,behavior:"smooth"})})}
function emptyChat(){
  if(chat.length||chatEl.querySelector(".bubble"))return;
  chatEl.innerHTML="";const d=el("div","empty");d.appendChild(el("b","","무엇을 도와드릴까요?"));
  d.appendChild(document.createTextNode("마이크 버튼을 누른 채 말하거나, 아래에 입력해 보세요."));chatEl.appendChild(d);
}
function bubble(cls,text,persist){
  const e=chatEl.querySelector(".empty");if(e)e.remove();
  const b=el("div","bubble "+cls,text);chatEl.appendChild(b);
  if(persist){chat.push({c:cls,t:text});saveChat()}
  scrollChat();return b;
}
function waiting(cls,label){
  const b=bubble(cls+" wait","");
  if(label)b.appendChild(document.createTextNode(label+" "));
  const d=el("span","dots");d.innerHTML="<i></i><i></i><i></i>";b.appendChild(d);return b;
}
function settle(b,cls,text){b.className="bubble "+cls;b.textContent=text;chat.push({c:cls,t:text});saveChat();scrollChat()}
chat.forEach(m=>{const b=el("div","bubble "+m.c,m.t);b.style.animation="none";chatEl.appendChild(b)});
emptyChat();
$("clearChat").addEventListener("click",()=>{if(!chat.length)return;if(!confirm("대화 기록을 지울까요?"))return;chat=[];saveChat();chatEl.innerHTML="";emptyChat()});

function handleReply(j,mine,pending){
  if(!j.ok){pending.remove();if(mine&&mine.classList.contains("wait"))mine.remove();bubble("err",j.error||"실패했어요.",true);return}
  if(mine&&mine.classList.contains("wait"))settle(mine,"me",j.heard||"(알아듣지 못했어요)");
  settle(pending,"bot",j.reply||"(답이 없어요)");
  refresh();if(cur==="home")loadSchedule();
}
function sendText(t){
  t=(t||"").trim();if(!t)return;
  bubble("me",t,true);const p=waiting("bot");
  api("POST","/api/text",{text:t,speak:speakOn},120000).then(j=>handleReply(j,null,p));
}
const textEl=$("text"),sendBtn=$("sendBtn");
textEl.addEventListener("input",()=>{sendBtn.disabled=!textEl.value.trim()});
$("f").addEventListener("submit",e=>{e.preventDefault();const t=textEl.value;textEl.value="";sendBtn.disabled=true;textEl.blur();sendText(t)});
$$("#chips .chip").forEach(b=>b.addEventListener("click",()=>sendText(b.textContent)));

/* 스피커로도 말하기 */
let speakOn=store("speak")!=="0";
function renderSpk(){
  const b=$("spk");b.classList.toggle("on",speakOn);b.setAttribute("aria-pressed",String(speakOn));
  b.querySelector(".w").style.display=speakOn?"":"none";b.querySelector(".x").style.display=speakOn?"none":"";
  $("spkTxt").textContent=speakOn?"스피커":"폰만";
}
$("spk").addEventListener("click",()=>{speakOn=!speakOn;store("speak",speakOn?"1":"0");renderSpk();toast(speakOn?"답을 스피커로도 말해요":"답을 휴대폰에만 보여 줘요")});
renderSpk();

/* ---------- 녹음 ---------- */
const talk=$("talk"),dockIn=$("dockIn");
let stream=null,rec=null,chunks=[],t0=0,tick=null,pressT=0,toggled=false,wantStop=false,inflight=0,mime="";
function pickMime(){
  if(!window.MediaRecorder||!MediaRecorder.isTypeSupported)return "";
  for(const c of ["audio/mp4","audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus"])if(MediaRecorder.isTypeSupported(c))return c;
  return "";
}
function micHelp(e){
  const name=e&&e.name||"";
  if(!window.isSecureContext||!navigator.mediaDevices)return "보안 연결이 아니라서 마이크를 쓸 수 없어요. 설정 › 마이크·인증서 도움말을 따라 주세요.";
  if(name==="NotAllowedError"||name==="SecurityError")return "마이크 사용이 거부됐어요. 설정 › Safari › 마이크 를 '묻기'로 바꿔 주세요.";
  if(name==="NotFoundError")return "마이크를 찾지 못했어요.";
  return "마이크를 켜지 못했어요 ("+(name||e)+").";
}
function getStream(){
  if(stream&&stream.getAudioTracks().some(t=>t.readyState==="live"))return Promise.resolve(stream);
  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)return Promise.reject({name:"Insecure"});
  return navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}}).then(s=>{stream=s;return s});
}
function releaseStream(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}}
document.addEventListener("visibilitychange",()=>{if(document.hidden){if(rec&&rec.state==="recording")stopRec();releaseStream()}});
function setIdle(){dockIn.classList.remove("recording");clearInterval(tick);talk.setAttribute("aria-label","누르고 말하기")}
function startRec(){
  if(rec&&rec.state==="recording")return;
  wantStop=false;
  getStream().then(s=>{
    if(wantStop)return;
    mime=pickMime();
    try{rec=mime?new MediaRecorder(s,{mimeType:mime}):new MediaRecorder(s)}catch(e){rec=new MediaRecorder(s)}
    chunks=[];
    rec.ondataavailable=ev=>{if(ev.data&&ev.data.size)chunks.push(ev.data)};
    rec.onstop=onStopped;rec.start();
    t0=Date.now();dockIn.classList.add("recording");talk.setAttribute("aria-label","녹음 멈추고 보내기");
    if(navigator.vibrate)navigator.vibrate(15);
    const upd=()=>{const s=(Date.now()-t0)/1000;$("recTime").textContent=s.toFixed(1)+"초";if(s>=30)stopRec()};
    upd();tick=setInterval(upd,100);
    $("recHint").textContent=toggled?"다시 누르면 보내요":"손을 떼면 보내요";
  }).catch(e=>{toggled=false;setIdle();bubble("err",micHelp(e))});
}
function stopRec(){wantStop=true;toggled=false;if(rec&&rec.state==="recording")rec.stop();else setIdle()}
function onStopped(){
  const dur=(Date.now()-t0)/1000;setIdle();
  const type=(rec&&rec.mimeType)||mime||"audio/mp4";
  const blob=new Blob(chunks,{type:type});chunks=[];
  if(dur<0.5||blob.size<200){toast("너무 짧아요. 버튼을 누른 채로 말해 주세요.",true);return}
  if(blob.size>2*1024*1024){toast("녹음이 너무 커요 (최대 30초).",true);return}
  const mine=waiting("me","듣는 중"),p=waiting("bot");
  inflight++;talk.classList.add("busy");
  api("POST","/api/audio?speak="+(speakOn?"1":"0"),undefined,120000,{body:blob,type:type.split(";")[0]}).then(j=>{
    handleReply(j,mine,p);if(--inflight<=0){inflight=0;talk.classList.remove("busy")}
  });
}
function down(e){
  if(e.cancelable)e.preventDefault();
  pressT=Date.now();
  if(rec&&rec.state==="recording"){pressT=0;stopRec();return}
  startRec();
}
function up(e){
  if(e&&e.cancelable)e.preventDefault();
  if(!pressT)return;
  const held=Date.now()-pressT;pressT=0;
  if(held<350){toggled=true;$("recHint").textContent="다시 누르면 보내요";return}
  stopRec();
}
let touch=false;
talk.addEventListener("touchstart",e=>{touch=true;down(e)},{passive:false});
talk.addEventListener("touchend",up,{passive:false});
talk.addEventListener("touchcancel",up,{passive:false});
talk.addEventListener("mousedown",e=>{if(!touch)down(e)});
window.addEventListener("mouseup",e=>{if(!touch&&pressT)up(e)});
talk.addEventListener("contextmenu",e=>e.preventDefault());

/* ---------- 알람 ---------- */
const KIND_ORDER=["alarm","reminder","timer","sleep"];
const KIND_NAME={alarm:"알람",reminder:"리마인더",timer:"타이머",sleep:"수면 타이머"};
const MUSIC_NAME={liked:"좋아요한 곡",mix:"맞춤 추천"};
let items=[];
function hm(at){
  const s=String(at||""),i=s.indexOf("T");if(i<0)return null;
  let h=parseInt(s.substr(i+1,2),10);const mm=s.substr(i+4,2);if(isNaN(h)||mm.length<2)return null;
  const ap=h<12?"오전":"오후";h=h%12||12;return {ap:ap,hm:h+":"+mm};
}
function sortAt(a,b){return String(a.at||"9").localeCompare(String(b.at||"9"))}
function itemTitle(it){return it.label||it.message||it.kind_name||KIND_NAME[it.kind]||""}
function itemSub(it){
  const parts=[];
  if(it.repeat_name)parts.push(it.repeat_name);
  if(it.when_text)parts.push(it.when_text);
  if(it.music)parts.push("♪ "+(MUSIC_NAME[it.music]||it.music));
  if(it.briefing)parts.push("브리핑");
  return parts.join(" · ")||it.description||"";
}
function loadSchedule(){
  return api("GET","/api/schedule").then(j=>{
    if(!j.ok){if(cur==="alarm")renderAlarmError(j.error);return}
    items=(j.items||[]).slice().sort(sortAt);renderAlarms();renderNext();
  });
}
function renderAlarmError(msg){
  const root=$("alarmGroups");root.innerHTML="";
  const d=el("div","empty-state");d.appendChild(el("b","","불러오지 못했어요"));d.appendChild(document.createTextNode(msg||""));root.appendChild(d);
}
function renderNext(){
  const act=items.filter(it=>it.enabled!==false);
  const n=act.find(it=>it.kind==="alarm"||it.kind==="reminder")||act[0];
  if(!n){$("nextKind").textContent="다음 알람";$("nextBig").textContent="예약 없음";$("nextSub").textContent="눌러서 알람 추가하기";return}
  $("nextKind").textContent="다음 "+(n.kind_name||KIND_NAME[n.kind]||"예약");
  const t=hm(n.at),big=$("nextBig");big.innerHTML="";
  if(t){big.appendChild(el("small","",t.ap));big.appendChild(document.createTextNode(t.hm))}else big.textContent=n.when_text||"-";
  $("nextSub").textContent=[itemTitle(n),n.repeat_name||n.when_text].filter(Boolean).join(" · ");
}
$("nextCard").addEventListener("click",()=>{showTab("alarm");if(!items.length)openSheet()});
function renderAlarms(){
  const root=$("alarmGroups");root.innerHTML="";
  if(!items.length){
    const d=el("div","empty-state");
    const ic=el("div","ic");ic.innerHTML='<svg class="i" viewBox="0 0 24 24"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M5 3 2 6M19 3l3 3"/></svg>';d.appendChild(ic);
    d.appendChild(el("b","","예약된 게 없어요"));d.appendChild(document.createTextNode("알람, 리마인더, 타이머를 추가해 보세요."));
    const btn=el("button","btn primary","추가하기");btn.type="button";btn.style.marginTop="16px";btn.addEventListener("click",()=>openSheet());d.appendChild(el("br"));d.appendChild(btn);
    root.appendChild(d);return;
  }
  KIND_ORDER.concat(["other"]).forEach(k=>{
    const list=items.filter(it=>k==="other"?KIND_ORDER.indexOf(it.kind)<0:it.kind===k);
    if(!list.length)return;
    const g=el("div","group");g.appendChild(el("h3","",KIND_NAME[k]||"기타"));
    const ul=el("ul","list");
    list.forEach(it=>{
      const editable=it.kind==="alarm"||it.kind==="reminder";
      const li=el("li",editable?"edit":"");if(editable&&it.enabled===false)li.classList.add("off");
      const w=el("div","when");const t=hm(it.at);
      if(t){w.appendChild(el("span","ampm",t.ap));w.appendChild(el("span","hm",t.hm))}else w.appendChild(el("span","hm","-"));
      const m=el("div","meta");m.appendChild(el("div","t",itemTitle(it)));m.appendChild(el("div","s",itemSub(it)));
      if(editable){
        li.setAttribute("role","button");li.tabIndex=0;li.setAttribute("aria-label",itemTitle(it)+" 편집");
        li.addEventListener("click",()=>openEdit(it));
        li.addEventListener("keydown",e=>{if(e.key==="Enter")openEdit(it)});
        const sw=el("label","sw"),inp=el("input");inp.type="checkbox";inp.checked=it.enabled!==false;inp.setAttribute("aria-label",itemTitle(it)+" 켜기/끄기");
        sw.appendChild(inp);sw.appendChild(el("span"));
        sw.addEventListener("click",e=>e.stopPropagation());
        inp.addEventListener("change",()=>{
          const on=inp.checked;li.classList.toggle("off",!on);inp.disabled=true;
          api("POST","/api/schedule/update",{id:String(it.id),enabled:on}).then(r=>{
            inp.disabled=false;report(r);
            if(r.ok){it.enabled=on;renderNext()}else{inp.checked=!on;li.classList.toggle("off",on)}
          });
        });
        li.appendChild(w);li.appendChild(m);li.appendChild(sw);ul.appendChild(li);return;
      }
      const del=el("button","icon-btn danger");del.type="button";del.setAttribute("aria-label","삭제");
      del.innerHTML='<svg class="i" viewBox="0 0 24 24" style="width:22px;height:22px"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>';
      del.addEventListener("click",()=>{
        if(!confirm("'"+(it.description||itemTitle(it))+"' 예약을 지울까요?"))return;
        busy(del,true);
        api("POST","/api/schedule/delete",{id:String(it.id)}).then(r=>{busy(del,false);report(r);loadSchedule()});
      });
      li.appendChild(w);li.appendChild(m);li.appendChild(del);ul.appendChild(li);
    });
    g.appendChild(ul);root.appendChild(g);
  });
}

/* ---------- 추가 시트 ---------- */
let kind=store("kind")||"alarm",rMode="in";
function pad(n){return (n<10?"0":"")+n}
function hhmm(d){return pad(d.getHours())+":"+pad(d.getMinutes())}
function ymd(d){return d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate())}
function setSeg(id,v){$$("#"+id+" button").forEach(b=>b.classList.toggle("on",b.dataset.v===v))}
let editing=null;
function setKind(k){
  kind=k;if(!editing)store("kind",k);setSeg("kindSeg",k);
  $$("#schedForm [data-for]").forEach(d=>{d.hidden=d.dataset.for!==k});
  $("schedSubmit").textContent=editing?"저장":{alarm:"알람 추가",reminder:"리마인더 추가",timer:"타이머 시작",sleep:"수면 타이머 설정"}[k];
}
function resetForm(){
  ["aLabel","aMusicText","rMsg","rMin","tLabel","tMin","sMin","aDate","rDate","aTime","rTime"].forEach(id=>{$(id).value=""});
  $("aMusic").value="";$("aMusicTextWrap").hidden=true;$("aBrief").checked=false;
  $("aRepeat").value="none";$("rRepeat").value="none";$("aDate").disabled=false;$("rDate").disabled=false;
}
function setMode(it){
  editing=it||null;
  $("sheetTitle").textContent=it?"편집":"새로 추가";
  $("kindSeg").hidden=!!it;$("schedDelete").hidden=!it;
}
function openEdit(it){
  setMode(it);resetForm();
  const once=(it.repeat||"none")==="none";
  if(it.kind==="alarm"){
    $("aTime").value=it.time||"";$("aRepeat").value=it.repeat||"none";
    $("aDate").value=once?(it.date||""):"";$("aDate").disabled=!once;  /* 한 번짜리는 날짜까지 그대로: 그냥 저장해도 같은 때 울린다 */
    $("aLabel").value=it.label||"";
    const mu=it.music;
    if(mu==="liked"||mu==="mix")$("aMusic").value=mu;
    else if(mu){$("aMusic").value="custom";$("aMusicText").value=mu;$("aMusicTextWrap").hidden=false}
    $("aBrief").checked=!!it.briefing;
  }else{
    $("rMsg").value=it.message||it.label||"";setRMode("at");
    $("rTime").value=it.time||"";$("rRepeat").value=it.repeat||"none";
    $("rDate").value=once?(it.date||""):"";$("rDate").disabled=!once;
  }
  $("aDate").min=$("rDate").min=ymd(new Date());
  setKind(it.kind);syncPresets();
  document.body.classList.add("sheet-open");
}
function setRMode(m){rMode=m;setSeg("rModeSeg",m);$("rAt").hidden=m!=="at";$("rIn").hidden=m!=="in"}
function syncPresets(){$$("[data-preset]").forEach(c=>{const v=$(c.dataset.preset).value;$$(".chip",c).forEach(b=>b.classList.toggle("on",b.dataset.v===v))})}
$$("#kindSeg button").forEach(b=>b.addEventListener("click",()=>setKind(b.dataset.v)));
$$("#rModeSeg button").forEach(b=>b.addEventListener("click",()=>setRMode(b.dataset.v)));
$$("[data-preset]").forEach(c=>{
  $$(".chip",c).forEach(b=>b.addEventListener("click",()=>{$(c.dataset.preset).value=b.dataset.v;syncPresets()}));
  $(c.dataset.preset).addEventListener("input",syncPresets);
});
$("aMusic").addEventListener("change",()=>{$("aMusicTextWrap").hidden=$("aMusic").value!=="custom"});
$("aRepeat").addEventListener("change",()=>{$("aDate").disabled=$("aRepeat").value!=="none"});
$("rRepeat").addEventListener("change",()=>{$("rDate").disabled=$("rRepeat").value!=="none"});
function openSheet(k){
  if(editing){setMode(null);resetForm()}
  const now=new Date();
  const nextHour=new Date(now.getTime()+60*60000);nextHour.setMinutes(0);
  if(now.getHours()>=12||now.getHours()<4){nextHour.setHours(7,0)}  /* 오후·밤이면 다음 날 아침 7시 */
  const half=new Date(now.getTime()+30*60000);half.setMinutes(half.getMinutes()<30?30:60,0,0);
  if(!$("aTime").value)$("aTime").value=hhmm(nextHour);
  if(!$("rTime").value)$("rTime").value=hhmm(half);
  $("aDate").min=$("rDate").min=ymd(now);
  setKind(k||kind);syncPresets();
  document.body.classList.add("sheet-open");
}
function closeSheet(){document.body.classList.remove("sheet-open");document.activeElement&&document.activeElement.blur()}
$("hdr-add").addEventListener("click",()=>openSheet());
$("sheetClose").addEventListener("click",closeSheet);$("scrim").addEventListener("click",closeSheet);
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeSheet()});
const val=id=>($(id).value||"").trim();
$("schedForm").addEventListener("submit",e=>{
  e.preventDefault();
  const d={kind:kind};
  if(kind==="alarm"){
    if(!val("aTime")){toast("알람 시각을 정해 주세요.",true);return}
    d.time=val("aTime");d.repeat=$("aRepeat").value;if(d.repeat==="none"&&val("aDate"))d.date=val("aDate");
    if(val("aLabel"))d.label=val("aLabel");
    const m=$("aMusic").value;
    if(m==="custom"){if(!val("aMusicText")){toast("틀 곡이나 분위기를 입력해 주세요.",true);return}d.music=val("aMusicText")}
    else if(m)d.music=m;
    d.briefing=$("aBrief").checked;
  }else if(kind==="reminder"){
    if(!val("rMsg")){toast("알려 줄 내용을 입력해 주세요.",true);return}
    d.message=val("rMsg");
    if(rMode==="in"){const n=parseInt(val("rMin"),10);if(!(n>0)){toast("몇 분 뒤인지 골라 주세요.",true);return}d.in_minutes=n}
    else{if(!val("rTime")){toast("시각을 정해 주세요.",true);return}d.time=val("rTime");d.repeat=$("rRepeat").value;if(d.repeat==="none"&&val("rDate"))d.date=val("rDate")}
  }else if(kind==="timer"){
    const tm=parseFloat(val("tMin"));if(!(tm>0)){toast("몇 분인지 골라 주세요.",true);return}
    d.minutes=tm;if(val("tLabel"))d.label=val("tLabel");
  }else{
    const sm=parseInt(val("sMin"),10);if(isNaN(sm)||sm<0){toast("몇 분인지 골라 주세요.",true);return}
    d.minutes=sm;
  }
  const btn=$("schedSubmit");busy(btn,true);
  const req=editing?api("POST","/api/schedule/update",Object.assign({id:String(editing.id)},d)):api("POST","/api/schedule",d);
  req.then(j=>{
    busy(btn,false);report(j);
    if(j.ok){["aLabel","aMusicText","rMsg","rMin","tLabel","tMin","sMin"].forEach(id=>{$(id).value=""});syncPresets();closeSheet();loadSchedule()}
  });
});
$("schedDelete").addEventListener("click",()=>{
  const it=editing;if(!it)return;
  if(!confirm("'"+(it.description||itemTitle(it))+"' 예약을 지울까요?"))return;
  const b=$("schedDelete");busy(b,true);
  api("POST","/api/schedule/delete",{id:String(it.id)}).then(r=>{busy(b,false);report(r);if(r.ok){closeSheet();loadSchedule()}});
});
setKind(kind);setRMode("in");

/* ---------- 설정 ---------- */
const SL=["system_volume","music_volume","voice_volume","voice_speed","news_count"],sendT={},drag={};
function fmt(k,v){if(v===null||v===undefined)return "알 수 없음";return k==="voice_speed"?Number(v).toFixed(2)+"배":k==="news_count"?Math.round(v)+"개":String(Math.round(v))}
function paintEl(e){const p=(e.value-e.min)/(e.max-e.min)*100;e.style.setProperty("--p",p+"%")}
function paint(k){paintEl($("s_"+k))}
function applySettings(s){
  if(!s)return;
  SL.forEach(k=>{
    const e=$("s_"+k);if(drag[k]||sendT[k])return;
    const v=s[k];
    if(v===null||v===undefined){e.disabled=true;$("o_"+k).textContent="알 수 없음";return}
    e.disabled=false;e.value=v;$("o_"+k).textContent=fmt(k,v);paint(k);
  });
  const rows=$("wakeRows");rows.innerHTML="";
  (s.wake_phrases||[]).forEach(p=>{
    const r=el("div","row");r.appendChild(el("span","lbl",p));
    const del=el("button","icon-btn danger");del.type="button";del.setAttribute("aria-label",p+" 삭제");
    del.innerHTML='<svg class="i" viewBox="0 0 24 24" style="width:22px;height:22px"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>';
    del.addEventListener("click",()=>{if(!confirm("'"+p+"' 을(를) 호출어에서 뺄까요?"))return;busy(del,true);saveSettings({wake_phrase_remove:p}).then(()=>busy(del,false))});
    r.appendChild(del);rows.appendChild(r);
  });
  if(s.english_wake){const r=el("div","row");r.appendChild(el("span","lbl",s.english_wake));r.appendChild(el("span","badge","고정 · 말로 바꿀 수 없어요"));rows.appendChild(r)}
  if(!rows.children.length){const r=el("div","row");r.appendChild(el("span","lbl muted","호출어가 없어요"));rows.appendChild(r)}
  const nrows=$("newsRows");nrows.innerHTML="";
  (s.news_topics||[]).forEach(t=>{
    const r=el("div","row");r.appendChild(el("span","lbl",t));
    const del=el("button","icon-btn danger");del.type="button";del.setAttribute("aria-label",t+" 삭제");
    del.innerHTML='<svg class="i" viewBox="0 0 24 24" style="width:22px;height:22px"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>';
    del.addEventListener("click",()=>{busy(del,true);saveSettings({news_topic_remove:t}).then(()=>busy(del,false),()=>busy(del,false))});
    r.appendChild(del);nrows.appendChild(r);
  });
  if(!nrows.children.length){const r=el("div","row");r.appendChild(el("span","lbl muted","없음 · 한국과 세계 주요 뉴스"));nrows.appendChild(r)}
  setVolRows(s.music_volume);
  $("infoFollow").textContent=s.follow_up_seconds>0?("답한 뒤 "+s.follow_up_seconds+"초"):"끔";
}
function loadSettings(){return api("GET","/api/settings").then(j=>{if(!j.ok){toast(j.error||"설정을 읽지 못했어요.",true);return}applySettings(j.settings)})}
function saveSettings(d,quiet){return api("POST","/api/settings",d).then(j=>{if(!j.ok||!quiet)report(j);if(j.settings)applySettings(j.settings);return j})}
SL.forEach(k=>{
  const e=$("s_"+k);
  e.addEventListener("input",()=>{
    $("o_"+k).textContent=fmt(k,e.value);paint(k);
    clearTimeout(sendT[k]);
    sendT[k]=setTimeout(()=>{sendT[k]=null;const d={};d[k]=k==="voice_speed"?parseFloat(e.value):parseInt(e.value,10);
      saveSettings(d,true).then(j=>{if(j.ok)toast(e.getAttribute("aria-label")+" "+fmt(k,d[k]))})},400);
  });
  ["touchstart","mousedown"].forEach(ev=>e.addEventListener(ev,()=>{drag[k]=true},{passive:true}));
  ["touchend","touchcancel","mouseup","change"].forEach(ev=>e.addEventListener(ev,()=>{drag[k]=false},{passive:true}));
});
$("wakeForm").addEventListener("submit",e=>{
  e.preventDefault();const p=val("wakeAdd");if(!p){toast("호출어를 입력해 주세요.",true);return}
  const b=$("wakeBtn");busy(b,true);
  saveSettings({wake_phrase_add:p}).then(j=>{busy(b,false);if(j.ok)$("wakeAdd").value=""});
});

$("newsForm").addEventListener("submit",e=>{
  e.preventDefault();const t=val("newsAdd");if(!t){toast("분야를 입력해 주세요.",true);return}
  const b=$("newsBtn");busy(b,true);
  saveSettings({news_topic_add:t}).then(j=>{busy(b,false);if(j.ok)$("newsAdd").value=""},()=>busy(b,false));
});

$("displayOff").addEventListener("click",()=>{
  const b=$("displayOff");busy(b,true);
  saveSettings({display_off:true}).then(()=>busy(b,false),()=>busy(b,false));
});

/* ---------- 화면 높이 보정 (홈 화면 앱) ---------- */
const isStandalone=()=>window.navigator.standalone===true||(window.matchMedia&&matchMedia("(display-mode: standalone)").matches);
function screenH(){
  const land=window.matchMedia&&matchMedia("(orientation: landscape)").matches;
  const a=screen.width||0,b=screen.height||0;
  return land?Math.min(a,b):Math.max(a,b);   /* iOS 의 screen 크기는 돌려도 바뀌지 않는다 */
}
function safeInsets(){
  const cs=getComputedStyle($("saProbe"));
  return {top:parseFloat(cs.paddingTop)||0,right:parseFloat(cs.paddingRight)||0,bottom:parseFloat(cs.paddingBottom)||0,left:parseFloat(cs.paddingLeft)||0};
}
function layoutGuard(){
  const root=document.documentElement;
  if(isStandalone()){
    const h=Math.max(window.innerHeight||0,root.clientHeight||0,screenH());
    root.style.setProperty("--app-h",h+"px");root.classList.add("fixh");
  }else{root.classList.remove("fixh");root.style.removeProperty("--app-h")}
  updateDiag();
}
function updateDiag(){
  const d=$("diag");if(!d)return;
  const vv=window.visualViewport,sa=safeInsets(),root=document.documentElement,bar=document.querySelector("nav.tabbar").getBoundingClientRect();
  d.textContent=[
    "standalone: "+isStandalone()+" (navigator.standalone="+window.navigator.standalone+")",
    "보정: "+(root.classList.contains("fixh")?("켬, --app-h="+root.style.getPropertyValue("--app-h")):"끔"),
    "innerHeight: "+window.innerHeight+" / innerWidth: "+window.innerWidth,
    "clientHeight: "+root.clientHeight,
    "screen: "+screen.width+"×"+screen.height+" (현재 방향 높이 "+screenH()+")",
    "visualViewport: "+(vv?Math.round(vv.width)+"×"+Math.round(vv.height)+" (offsetTop "+Math.round(vv.offsetTop)+")":"없음"),
    "safe-area: 위 "+sa.top+" · 아래 "+sa.bottom+" · 왼 "+sa.left+" · 오른 "+sa.right,
    "탭 막대: 위 "+Math.round(bar.top)+" · 아래 "+Math.round(bar.bottom),
    "devicePixelRatio: "+window.devicePixelRatio
  ].join(String.fromCharCode(10));
}
layoutGuard();
window.addEventListener("resize",layoutGuard);
window.addEventListener("orientationchange",()=>setTimeout(layoutGuard,300));
if(window.visualViewport)window.visualViewport.addEventListener("resize",layoutGuard);
window.addEventListener("pageshow",layoutGuard);
$("diagDetails").addEventListener("toggle",updateDiag);

/* ---------- 시작 ---------- */
renderNP();renderNext();
const hash=(location.hash||"").replace("#","");
const startTab=hash.split("-")[0];
showTab(TABS[startTab]?startTab:(store("tab")||"home"));
if(hash==="alarm-add")openSheet();
if(hash==="music-vol")$$("[data-vol=m]")[0].click();
if(/-end$/.test(hash))setTimeout(()=>{const sc=$("app");sc.scrollTop=sc.scrollHeight},300);
if(hash==="alarm-edit")loadSchedule().then(()=>{const it=items.find(x=>x.kind==="alarm");if(it)openEdit(it)});
setInterval(()=>{if(!document.hidden&&cur!=="alarm"&&cur!=="set")refresh()},10000);
setInterval(()=>{if(!document.hidden&&(cur==="alarm"||cur==="home"))loadSchedule()},30000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden){refresh();if(cur==="alarm"||cur==="home")loadSchedule();if(cur==="set")loadSettings()}});
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- 자체 시험


def _selftest() -> int:
    import urllib.error
    import urllib.request

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scratch = Path(
        os.environ.get("REMOTE_SELFTEST_DIR") or tempfile.gettempdir()
    )
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="remote-selftest-", dir=scratch))
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    class MinimalStub:
        """main.py 가 새 메서드를 아직 안 가진 경우."""

        def remote_text(self, text: str, speak: bool) -> str:
            return ""

        def remote_audio(self, pcm16k: np.ndarray, speak: bool) -> tuple[str, str]:
            return "", ""

        def remote_status(self) -> dict:
            return {}

    class Stub(MinimalStub):
        def __init__(self) -> None:
            self.audio: list[float] = []
            self.texts: list[tuple[str, bool]] = []
            self.events: list[dict] = []
            self.added: list[dict] = []
            self.music_calls: list[tuple] = []
            self.display_off = 0
            self.settings = {
                "system_volume": 50, "music_volume": 40, "voice_volume": 80, "voice_speed": 1.0,
                "wake_phrases": ["스피커야"], "english_wake": "hey_jarvis", "follow_up_seconds": 6.0,
                "news_topics": ["인공지능"], "news_count": 4,
            }

        def remote_text(self, text: str, speak: bool) -> str:
            self.texts.append((text, speak))
            if text == "느리게":
                time.sleep(3)
            return f"알겠어요: {text}"

        def remote_audio(self, pcm16k: np.ndarray, speak: bool) -> tuple[str, str]:
            assert pcm16k.dtype == np.float32 and pcm16k.ndim == 1
            self.audio.append(len(pcm16k) / SAMPLE_RATE)
            return "음악 꺼줘", f"{len(pcm16k) / SAMPLE_RATE:.2f}초 들었어요"

        def remote_status(self) -> dict:
            return {"재생 중": "테스트 곡", "음량": {"음악": 40}, "예정": ["07:00 알람"]}

        def remote_schedule(self) -> list[dict]:
            return list(self.events)

        def remote_schedule_add(self, data: dict) -> dict:
            self.added.append(data)
            if data.get("kind") == "alarm" and not data.get("time"):
                return {"ok": False, "message": "알람 시각이 필요해요."}
            eid = f"e{len(self.added)}"
            self.events.append({
                "id": eid, "kind": data["kind"], "kind_name": "알람", "label": data.get("label", ""),
                "message": "", "when_text": "내일 오전 7시", "at": "2026-11-16T07:00", "repeat": "none",
                "repeat_name": "", "music": data.get("music"), "briefing": bool(data.get("briefing")),
                "description": "내일 오전 7시 알람", "enabled": True, "time": data.get("time"), "date": data.get("date"),
            })
            return {"ok": True, "message": "내일 오전 7시에 알람을 맞췄어요."}

        def remote_schedule_update(self, event_id: str, data: dict) -> dict:
            for i, e in enumerate(self.events):
                if e["id"] == event_id:
                    if set(data) == {"enabled"}:
                        e["enabled"] = bool(data["enabled"])
                        return {"ok": True, "message": "껐어요." if not e["enabled"] else "켰어요."}
                    new = dict(e, **data, enabled=True)
                    self.events[i] = new
                    return {"ok": True, "message": "바꿨어요."}
            return {"ok": False, "message": "그런 예약이 없어요."}

        def remote_schedule_delete(self, event_id: str) -> dict:
            for e in self.events:
                if e["id"] == event_id:
                    self.events.remove(e)
                    return {"ok": True, "message": "지웠어요."}
            return {"ok": False, "message": "그런 예약이 없어요."}

        def remote_settings(self) -> dict:
            return dict(self.settings, wake_phrases=list(self.settings["wake_phrases"]),
                        news_topics=list(self.settings["news_topics"]))

        def remote_settings_set(self, data: dict) -> dict:
            for k in ("system_volume", "music_volume", "voice_volume", "voice_speed", "news_count"):
                if k in data:
                    self.settings[k] = data[k]
            if data.get("wake_phrase_add"):
                self.settings["wake_phrases"].append(data["wake_phrase_add"])
            if data.get("display_off"):
                self.display_off += 1
            if data.get("news_topic_add"):
                self.settings["news_topics"].append(data["news_topic_add"])
            if data.get("news_topic_remove") in self.settings["news_topics"]:
                self.settings["news_topics"].remove(data["news_topic_remove"])
            if data.get("wake_phrase_remove") in self.settings["wake_phrases"]:
                self.settings["wake_phrases"].remove(data["wake_phrase_remove"])
            return {"ok": True, "message": "바꿨어요.", "settings": self.remote_settings()}

        def remote_music(self, action: str, query: str | None = None, kind: str = "song") -> dict:
            self.music_calls.append((action, query, kind))
            if action == "boom":
                raise RuntimeError("test")
            return {"ok": True, "message": f"{action} 했어요.", "now_playing": query}

    stub = Stub()
    token = secrets.token_urlsafe(16)
    srv = RemoteServer(stub, port=0, token=token, cert_dir=work / "certs")
    srv.fail_delay = 0.05
    srv.start()
    base = f"https://127.0.0.1:{srv.port}"
    ctx = ssl.create_default_context(cafile=str(work / "certs" / "ca.crt"))  # 만든 CA 를 믿는다

    def req(path: str, data: bytes | None = None, headers: dict | None = None, method: str | None = None):
        r = urllib.request.Request(base + path, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(r, context=ctx, timeout=60) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    try:
        # 인증서: 만든 CA 로 실제 검증 (hostname 127.0.0.1 SAN 포함)
        st, body, hdrs = req(f"/?t={token}")
        check("page with token (TLS verified by generated CA)", st == 200 and "누르고 말하기" in body.decode(), f"status={st}")
        cookie = hdrs.get("Set-Cookie", "").split(";")[0]
        check("cookie set", cookie.startswith(COOKIE_NAME + "="))
        st, _, _ = req("/", headers={"Cookie": cookie})
        check("page with cookie only", st == 200)
        st, _, _ = req("/api/status", headers={"X-Remote-Token": token})
        check("status with header", st == 200)
        st, body, _ = req("/")
        check("403 without token", st == 403 and b"token" not in body.lower(), f"status={st}")
        st, _, _ = req("/?t=wrong-token-123")
        check("403 with wrong token", st == 403)
        srv._fails.success("127.0.0.1")  # 위 실패 2회는 초기화

        st, body, _ = req(
            "/api/text", json.dumps({"text": "음악 꺼줘", "speak": False}).encode(),
            {"Content-Type": "application/json", "Cookie": cookie},
        )
        j = json.loads(body)
        check("/api/text round trip", st == 200 and j["reply"] == "알겠어요: 음악 꺼줘" and stub.texts[-1] == ("음악 꺼줘", False), str(j))
        st, body, _ = req("/api/status", headers={"Cookie": cookie})
        j = json.loads(body)
        check("/api/status", st == 200 and j["status"]["재생 중"] == "테스트 곡", str(j)[:80])
        st, body, _ = req("/cert", headers={"Cookie": cookie})
        check("/cert serves DER CA", st == 200 and body[:1] == b"\x30", f"{len(body)} bytes")
        st, _, _ = req("/cert")
        check("/cert needs token", st == 403)
        srv._fails.success("127.0.0.1")

        # 음성 파일 만들기
        aiff = work / "x.aiff"
        subprocess.run(["say", "-v", "Yuna", "-o", str(aiff), "음악 꺼줘"], check=True)
        ff = ffmpeg_path()
        ref = decode_audio(aiff.read_bytes())
        ref_s = len(ref) / SAMPLE_RATE
        print(f"reference duration (aiff): {ref_s:.2f}s")
        formats = {
            "m4a (AAC)": (["-c:a", "aac", "-b:a", "64k"], ".m4a", "audio/mp4"),
            "mp4 fragmented (Safari-like)": (["-c:a", "aac", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4"], ".mp4", "audio/mp4"),
            "webm (opus)": (["-c:a", "libopus", "-b:a", "32k"], ".webm", "audio/webm"),
            "wav": (["-c:a", "pcm_s16le", "-ar", "44100"], ".wav", "audio/wav"),
        }
        for name, (args, ext, ctype) in formats.items():
            out = work / f"x{ext}"
            r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(aiff), *args, str(out)], capture_output=True)
            if r.returncode != 0:
                check(f"/api/audio {name}", False, "ffmpeg encode failed: " + r.stderr.decode()[-200:])
                continue
            st, body, _ = req(f"/api/audio?speak=0", out.read_bytes(), {"Content-Type": ctype, "Cookie": cookie})
            j = json.loads(body)
            got = stub.audio[-1] if st == 200 else -1
            check(
                f"/api/audio {name}",
                st == 200 and abs(got - ref_s) < 0.25 and j["heard"] == "음악 꺼줘",
                f"status={st} decoded={got:.2f}s ref={ref_s:.2f}s size={out.stat().st_size}B",
            )

        # 너무 짧은 녹음
        short = work / "short.m4a"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(aiff), "-t", "0.2", "-c:a", "aac", str(short)], check=True)
        st, body, _ = req("/api/audio", short.read_bytes(), {"Content-Type": "audio/mp4", "Cookie": cookie})
        check("too-short audio rejected (Korean msg)", st == 400 and "짧아요" in json.loads(body)["error"], body.decode())
        st, body, _ = req("/api/audio", b"", {"Content-Type": "audio/mp4", "Cookie": cookie})
        check("empty audio rejected", st == 400, body.decode())
        st, body, _ = req("/api/audio", os.urandom(5000), {"Content-Type": "audio/mp4", "Cookie": cookie})
        check("garbage audio rejected", st == 400, body.decode())

        # 너무 큰 업로드
        st, body, _ = req("/api/audio", b"\0" * (MAX_UPLOAD_BYTES + 1), {"Content-Type": "audio/mp4", "Cookie": cookie})
        check("oversized upload (2 MB + 1) rejected with 413", st == 413, f"status={st} {body.decode()}")
        try:
            st, body, _ = req("/api/audio", b"\0" * (10 * MAX_UPLOAD_BYTES), {"Content-Type": "audio/mp4", "Cookie": cookie})
            check("huge upload (20 MB) rejected", st == 413, f"status={st}")
        except (urllib.error.URLError, ConnectionError, ssl.SSLError) as e:
            check("huge upload (20 MB) rejected (connection closed unread)", True, repr(e))
        # 너무 긴 녹음 (40초)
        long = work / "long.m4a"
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=40", "-c:a", "aac", "-b:a", "24k", str(long)], check=True)
        st, body, _ = req("/api/audio", long.read_bytes(), {"Content-Type": "audio/mp4", "Cookie": cookie})
        check("40 s audio rejected", st == 400 and "길어요" in json.loads(body)["error"], f"size={long.stat().st_size}B {body.decode()}")

        # 동시 요청 중에도 상태는 응답 (스레드)
        st, _, _ = req("/api/status", headers={"Cookie": cookie})
        check("status still OK", st == 200)

        # ---- 직접 조작 (Claude 안 거침)
        J = {"Content-Type": "application/json", "Cookie": cookie}

        def jreq(path: str, obj=None, method: str | None = None, headers: dict | None = None):
            data = None if obj is None else (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
            st, body, _ = req(path, data, headers or J, method)
            try:
                return st, json.loads(body)
            except ValueError:
                return st, {"raw": body.decode(errors="replace")}

        for path, method, obj in [
            ("/api/schedule", None, None), ("/api/settings", None, None),
            ("/api/schedule", "POST", {"kind": "timer", "minutes": 1}),
            ("/api/schedule/delete", "POST", {"id": "e1"}), ("/api/schedule/e1", "DELETE", None),
            ("/api/schedule/update", "POST", {"id": "e1", "enabled": False}),
            ("/api/settings", "POST", {"music_volume": 10}), ("/api/music", "POST", {"action": "stop"}),
        ]:
            st, _ = jreq(path, obj, method, headers={"Content-Type": "application/json"})
            check(f"403 without token: {method or 'GET'} {path}", st == 403, f"status={st}")
        check("no-token requests are not counted as failures", not srv._fails.blocked("127.0.0.1"))

        st, j = jreq("/api/schedule")
        check("GET /api/schedule (empty)", st == 200 and j == {"ok": True, "items": []}, str(j))
        alarm = {"kind": "alarm", "time": "07:00", "date": "2026-11-16", "repeat": "weekdays", "label": "출근", "music": "liked", "briefing": True}
        st, j = jreq("/api/schedule", alarm)
        check("POST /api/schedule alarm", st == 200 and j["ok"] and "알람" in j["message"] and stub.added[-1] == alarm, str(j))
        st, j = jreq("/api/schedule", {"kind": "alarm"})
        check("POST /api/schedule handler rejection passes message", st == 200 and not j["ok"] and j["error"] == "알람 시각이 필요해요.", str(j))
        st, j = jreq("/api/schedule", {"time": "07:00"})
        check("POST /api/schedule without kind -> 400", st == 400 and not j["ok"], str(j))
        st, j = jreq("/api/schedule", b"{not json")
        check("POST /api/schedule bad JSON -> 400", st == 400 and j["error"] == "요청 형식이 잘못됐어요.", str(j))
        st, j = jreq("/api/schedule", b"[1,2]")
        check("POST /api/schedule non-object JSON -> 400", st == 400, str(j))
        st, j = jreq("/api/schedule")
        check("GET /api/schedule lists added", st == 200 and len(j["items"]) == 1 and j["items"][0]["id"] == "e1", str(j)[:120])
        st, j = jreq("/api/schedule/delete", {"id": "nope"})
        check("delete unknown id -> ok:false message", st == 200 and not j["ok"] and j["message"] == "그런 예약이 없어요.", str(j))
        st, j = jreq("/api/schedule/delete", {})
        check("delete without id -> 400", st == 400, str(j))
        st, j = jreq("/api/schedule/delete", {"id": "e1"})
        check("POST /api/schedule/delete", st == 200 and j["ok"] and not stub.events, str(j))
        # 편집·켜고 끄기
        jreq("/api/schedule", alarm)  # e3
        st, j = jreq("/api/schedule/update", {"id": "e3", "enabled": False})
        check("update: toggle off", st == 200 and j["ok"] and stub.events[0]["enabled"] is False, str(j))
        st, j = jreq("/api/schedule")
        check("schedule shows enabled=false", j["items"][0]["enabled"] is False, str(j["items"][0])[:80])
        st, j = jreq("/api/schedule/update", {"id": "e3", "kind": "alarm", "time": "06:30", "repeat": "daily", "label": "운동"})
        check("update: replace fields", st == 200 and j["ok"] and stub.events[0]["time"] == "06:30" and stub.events[0]["label"] == "운동", str(j))
        st, j = jreq("/api/schedule/update", {"id": "nope", "enabled": True})
        check("update: unknown id -> ok:false message", st == 200 and not j["ok"] and j["error"] == "그런 예약이 없어요.", str(j))
        st, j = jreq("/api/schedule/update", {"enabled": True})
        check("update: missing id -> 400", st == 400, str(j))
        st, j = jreq("/api/schedule/update", {"id": "e3"})
        check("update: no fields -> 400", st == 400, str(j))
        st, j = jreq("/api/schedule/update", b"{bad")
        check("update: bad JSON -> 400", st == 400, str(j))
        st, j = jreq("/api/schedule/e3", method="DELETE")
        check("DELETE /api/schedule/<id>", st == 200 and j["ok"] and not stub.events, str(j))

        st, j = jreq("/api/settings")
        check("GET /api/settings", st == 200 and j["ok"] and j["settings"]["music_volume"] == 40, str(j)[:120])
        st, j = jreq("/api/settings", {"music_volume": 25, "voice_speed": 1.15, "wake_phrase_add": "자비스"})
        check(
            "POST /api/settings",
            st == 200 and j["ok"] and j["settings"]["music_volume"] == 25 and j["settings"]["voice_speed"] == 1.15
            and "자비스" in j["settings"]["wake_phrases"],
            str(j)[:160],
        )
        st, j = jreq("/api/settings", {"display_off": True})
        check("POST /api/settings display_off", st == 200 and j["ok"] and stub.display_off == 1, str(j)[:120])
        check("page has display-off button", 'id="displayOff"' in PAGE_HTML, "")
        st, j = jreq("/api/settings", {"news_topic_add": "반도체"})
        check("POST /api/settings news_topic_add", st == 200 and j["settings"]["news_topics"] == ["인공지능", "반도체"], str(j)[:160])
        st, j = jreq("/api/settings", {"news_topic_remove": "인공지능"})
        check("POST /api/settings news_topic_remove", st == 200 and j["settings"]["news_topics"] == ["반도체"], str(j)[:160])
        st, j = jreq("/api/settings", {"news_count": 6})
        check("POST /api/settings news_count", st == 200 and j["settings"]["news_count"] == 6, str(j)[:160])
        check("page has news count slider", 'id="s_news_count"' in PAGE_HTML, "")
        check("page has news topics form", 'id="newsForm"' in PAGE_HTML and 'id="newsRows"' in PAGE_HTML, "")
        st, j = jreq("/api/settings", {})
        check("POST /api/settings empty -> 400", st == 400, str(j))
        st, j = jreq("/api/settings", b"nope")
        check("POST /api/settings bad JSON -> 400", st == 400, str(j))

        st, j = jreq("/api/music", {"action": "pause"})
        check("POST /api/music pause", st == 200 and j["ok"] and stub.music_calls[-1] == ("pause", None, "song"), str(j))
        st, j = jreq("/api/music", {"action": "play", "query": "아이유", "kind": "artist"})
        check("POST /api/music play artist", st == 200 and j["now_playing"] == "아이유" and stub.music_calls[-1] == ("play", "아이유", "artist"), str(j))
        st, j = jreq("/api/music", {"action": "play"})
        check("POST /api/music play without query -> 400", st == 400, str(j))
        st, j = jreq("/api/music", {"action": "boom"})
        check("handler exception -> 500 Korean message", st == 500 and not j["ok"], str(j))
        st, j = jreq("/api/music", b"{")
        check("POST /api/music bad JSON -> 400", st == 400, str(j))

        # 긴 Claude 요청이 도는 동안에도 직접 조작은 바로 끝나야 한다
        slow = threading.Thread(target=lambda: jreq("/api/text", {"text": "느리게", "speak": False}))
        slow.start()
        time.sleep(0.3)
        t = time.monotonic()
        st, j = jreq("/api/music", {"action": "next"})
        took = time.monotonic() - t
        st2, _ = jreq("/api/schedule")
        check("direct control not blocked by a running Claude request", st == 200 and st2 == 200 and took < 1.0 and slow.is_alive(), f"took={took:.2f}s")
        slow.join()

        # ---- 홈 화면 아이콘·매니페스트 (토큰 없이)
        import struct

        for path, size in (("/apple-touch-icon.png", 180), ("/apple-touch-icon-precomposed.png", 180), ("/icon-180.png", 180), ("/icon-512.png", 512)):
            st, body, hdrs = req(path)
            ok = st == 200 and body[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", body[16:24]) == (size, size)
            check(f"{path} served without token, PNG {size}x{size}", ok and hdrs.get("Content-Type") == "image/png", f"status={st} {len(body)}B")
            if path == "/icon-512.png":
                (scratch / "icon-512-preview.png").write_bytes(body)
        st, body, _ = req("/manifest.json")
        m = json.loads(body) if st == 200 else {}
        check("/manifest.json without token, no start_url (keeps ?t=)", st == 200 and m.get("short_name") == "스피커" and "start_url" not in m and "t=" not in body.decode(), str(m)[:100])
        check("public paths did not count as failures", not srv._fails.blocked("127.0.0.1"))
        page = req("/", headers={"Cookie": cookie})[1].decode()
        check("page has apple-touch-icon + title meta", 'rel="apple-touch-icon"' in page and 'apple-mobile-web-app-title" content="스피커"' in page)

        # 새 메서드가 없는 handler → 501 친절한 메시지
        srv2 = RemoteServer(MinimalStub(), port=0, token=token, cert_dir=work / "certs")
        srv2.start()
        try:
            base_saved = base
            base = f"https://127.0.0.1:{srv2.port}"
            res = [jreq("/api/schedule"), jreq("/api/settings"), jreq("/api/music", {"action": "stop"}),
                   jreq("/api/schedule", {"kind": "timer", "minutes": 1}), jreq("/api/settings", {"music_volume": 1}),
                   jreq("/api/schedule/delete", {"id": "x"}), jreq("/api/schedule/update", {"id": "x", "enabled": True})]
            check("handler without new methods -> 501 '준비 중'", all(s == 501 and "준비 중" in j["error"] for s, j in res), str([s for s, _ in res]))
        finally:
            base = base_saved
            srv2.stop()

        # 반복된 잘못된 토큰 → 차단 (올바른 토큰도 막힘)
        codes = [req(f"/?t=bad{i}")[0] for i in range(FAIL_LIMIT)]
        st, _, _ = req("/", headers={"Cookie": cookie})
        check("rate limit: blocked after repeated bad tokens", all(c == 403 for c in codes) and st == 403, f"codes={codes} then valid={st}")
        check("IP marked blocked", srv._fails.blocked("127.0.0.1"))
    finally:
        srv.stop()

    # 파일 권한
    key_mode = oct((work / "certs" / "server.key").stat().st_mode & 0o777)
    ca_mode = oct((work / "certs" / "ca.key").stat().st_mode & 0o777)
    check("key files 0600", key_mode == "0o600" and ca_mode == "0o600", f"server={key_mode} ca={ca_mode}")
    tok = work / "tok"
    t1 = load_or_create_token(tok)
    t2 = load_or_create_token(tok)
    check("token persisted 0600", t1 == t2 and oct(tok.stat().st_mode & 0o777) == "0o600")
    # 서버가 정말 꺼졌는지
    try:
        socket.create_connection(("127.0.0.1", srv.port), timeout=1).close()
        check("server stopped", False)
    except OSError:
        check("server stopped", True)

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f"; FAILED: {failed}" if failed else ""))
    shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("사용법: uv run python -m speaker.remote --selftest")
