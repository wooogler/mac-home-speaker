"""사용자에 대한 기억: 직접 말해 준 사실 + 지난 대화 요약. memory.json 한 파일에 저장한다."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("speaker.memory")

MAX_FACTS = 100
MAX_SUMMARIES = 30  # 프롬프트에는 최근 것만 넣는다
SUMMARIES_IN_PROMPT = 10


class Memory:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.facts: list[dict] = []  # {"text", "added"}
        self.summaries: list[dict] = []  # {"when", "text"}
        self._mtime = 0.0
        self._load()

    def _refresh(self) -> None:
        """파일을 밖에서 고쳤으면 (직접 편집, 가져오기) 다시 읽는다. 호출 전 _lock 을 잡고 있어야 한다."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self._load()

    # ------------------------------------------------------------ 저장소

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.error("기억 파일을 읽지 못했습니다 (%s). 빈 기억으로 시작합니다.", exc)
            return
        self.facts = [f for f in data.get("facts", []) if isinstance(f, dict) and f.get("text")]
        self.summaries = [s for s in data.get("summaries", []) if isinstance(s, dict) and s.get("text")]
        self._mtime = self.path.stat().st_mtime

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # 나만 읽을 수 있게
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"facts": self.facts, "summaries": self.summaries}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    # ------------------------------------------------------------ 사실

    def remember(self, text: str) -> str:
        text = " ".join(text.split())
        with self._lock:
            self._refresh()
            for f in self.facts:
                if f["text"] == text:
                    return "이미 기억하고 있는 내용입니다."
            self.facts.append({"text": text, "added": time.strftime("%Y-%m-%d")})
            del self.facts[:-MAX_FACTS]
            self._save()
        log.info("기억 추가: %s", text)
        return f"새로 저장했습니다: {text}"

    def forget(self, about: str | None) -> list[str]:
        """about 이 들어간 기억을 지운다. about 이 비어 있으면 사실과 대화 요약을 모두 지운다."""
        with self._lock:
            self._refresh()
            if not about:
                removed = [f["text"] for f in self.facts] + [s["text"] for s in self.summaries]
                self.facts, self.summaries = [], []
            else:
                key = about.replace(" ", "")
                removed = [f["text"] for f in self.facts if key in f["text"].replace(" ", "")]
                self.facts = [f for f in self.facts if f["text"] not in removed]
                # 지난 대화 요약에 남은 같은 내용도 지운다
                gone = [s for s in self.summaries if key in s["text"].replace(" ", "")]
                self.summaries = [s for s in self.summaries if s not in gone]
                removed += [f"(대화 요약) {s['text']}" for s in gone]
            if removed:
                self._save()
        log.info("기억 삭제 %d건 (%r)", len(removed), about)
        return removed

    # ------------------------------------------------------------ 대화 요약

    def add_summary(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        with self._lock:
            self._refresh()
            self.summaries.append({"when": time.strftime("%Y-%m-%d %H:%M"), "text": text})
            del self.summaries[:-MAX_SUMMARIES]
            self._save()
        log.info("지난 대화 요약 저장: %s", text)

    # ------------------------------------------------------------ 프롬프트

    def prompt_block(self) -> str | None:
        """시스템 프롬프트 뒤에 붙일 기억. 기억이 없으면 None."""
        with self._lock:
            self._refresh()
            facts = list(self.facts)
            summaries = self.summaries[-SUMMARIES_IN_PROMPT:]
        if not facts and not summaries:
            return None
        parts = ["사용자에 대해 기억하고 있는 것 (이전 대화에서 저장한 내용입니다. 필요할 때 자연스럽게 활용하고, 목록을 그대로 읊지 마세요)"]
        if facts:
            parts.append("\n".join(f"- {f['text']}" for f in facts))
        else:
            parts.append("- 아직 없음")
        if summaries:
            parts.append("최근 대화 (오래된 것부터)")
            parts.append("\n".join(f"- {s['when']}: {s['text']}" for s in summaries))
        return "\n".join(parts)
