"""MLX 를 쓰는 모델들(Whisper 음성 인식, Qwen 음성 합성)이 GPU 를 동시에 쓰지 않도록 하는 공용 잠금."""

import threading

MLX_LOCK = threading.Lock()
