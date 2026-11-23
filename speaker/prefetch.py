"""처음 설치할 때 필요한 모델을 미리 내려받는다 (setup.sh 가 부른다).

    uv run python -m speaker.prefetch

음성 인식(Whisper)과 음성 합성(Qwen3-TTS) 모델은 합쳐 수 GB 라서, 스피커를 처음 켤 때 받으면
한참 동안 아무 반응이 없어 보인다. 여기서 진행 상황을 보여 주며 미리 받아 둔다.
이미 받은 모델은 건너뛴다.
"""

from __future__ import annotations

import sys

from .config import load_config


def main() -> int:
    cfg = load_config()
    from huggingface_hub import snapshot_download

    repos = [("음성 인식", cfg.whisper_model)]
    if cfg.tts_backend in ("auto", "qwen"):
        repos.append(("음성 합성", cfg.qwen_tts_model))
    for label, repo in repos:
        print(f"\n▶ {label} 모델: {repo}", flush=True)
        try:
            snapshot_download(repo)
        except Exception as exc:
            print(f"✗ 내려받지 못했습니다: {exc}\n  인터넷 연결을 확인하고 다시 실행해 주세요.")
            return 1

    if cfg.wake_word.lower() not in ("off", "none", ""):
        print(f"\n▶ 호출어 모델: {cfg.wake_word}", flush=True)
        from .audio_in import _resolve_wake_model

        try:
            _resolve_wake_model(cfg.wake_word)
        except Exception as exc:
            print(f"✗ 호출어 모델을 준비하지 못했습니다: {exc}")
            return 1

    print("\n✓ 모델 준비 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
