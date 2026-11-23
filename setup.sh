#!/bin/zsh
# mac-home-speaker 처음 설치. 여러 번 실행해도 괜찮습니다 (이미 된 단계는 건너뜀).
#   ./setup.sh
cd "$(dirname "$0")" || exit 1

step() { print -P "\n%B▶ $1%b"; }
fail() { print -P "\n%F{red}✗ $1%f"; exit 1; }
ok() { print -P "%F{green}✓%f $1"; }

# ------------------------------------------------------------ 1. 이 맥에서 돌 수 있는지
step "맥 확인"
[[ $(uname -m) == arm64 ]] || fail "Apple Silicon 맥(M1 이상)이 필요합니다. 음성 인식·합성 모델이 Apple Silicon 전용(MLX)입니다."
macos_major=$(sw_vers -productVersion | cut -d. -f1)
(( macos_major >= 14 )) || fail "macOS 14(Sonoma) 이상이 필요합니다. 지금: $(sw_vers -productVersion)"
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
ok "Apple Silicon, macOS $(sw_vers -productVersion), 메모리 ${mem_gb}GB"
(( mem_gb >= 16 )) || print "  메모리가 16GB 보다 적으면 느리거나 모델을 불러오지 못할 수 있습니다."
free_gb=$(df -g . | awk 'NR==2 {print $4}')
(( free_gb >= 15 )) || fail "디스크 여유 공간이 15GB 이상 필요합니다 (모델 약 9GB + 프로그램). 지금: ${free_gb}GB"
ok "디스크 여유 ${free_gb}GB"

# ------------------------------------------------------------ 2. Homebrew 와 도구
step "도구 설치 (Homebrew: uv, mpv, node)"
if ! command -v brew >/dev/null; then
  [[ -x /opt/homebrew/bin/brew ]] && eval "$(/opt/homebrew/bin/brew shellenv)"
fi
command -v brew >/dev/null || fail "Homebrew 가 필요합니다. https://brew.sh 의 설치 명령을 터미널에 붙여 넣어 설치한 뒤 다시 ./setup.sh 를 실행하세요."
for pkg in uv mpv node; do
  if command -v $pkg >/dev/null; then
    ok "$pkg"
  else
    brew install $pkg || fail "$pkg 설치에 실패했습니다."
  fi
done

# ------------------------------------------------------------ 3. 파이썬 패키지
step "파이썬 패키지 설치 (uv sync, 처음엔 몇 분 걸려요)"
uv sync || fail "패키지 설치에 실패했습니다. 메시지를 확인하고 다시 실행해 주세요."
ok "패키지 설치 완료"

# ------------------------------------------------------------ 4. 설정 파일 (.env)
step "설정 (.env)"
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  ok ".env 를 만들었습니다."
fi

set_env() {  # set_env 이름 값 : .env 에서 그 줄을 바꾸거나 없으면 추가한다
  KEY="$1" VALUE="$2" uv run --quiet python - <<'PY'
import os, re
from pathlib import Path
p = Path(".env"); key, value = os.environ["KEY"], os.environ["VALUE"]
text = p.read_text(encoding="utf-8")
line = f"{key}={value}"
pat = re.compile(rf"^#?\s*{re.escape(key)}=.*$", re.M)
text = pat.sub(lambda m: line, text, count=1) if pat.search(text) else text.rstrip("\n") + "\n" + line + "\n"
p.write_text(text, encoding="utf-8")
PY
}
get_env() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }

if [[ -z "$(get_env ANTHROPIC_API_KEY)" ]]; then
  print "Claude API 키가 필요합니다. https://console.anthropic.com 에서 로그인 → API Keys → Create Key 로 만들고,"
  print "결제 수단을 등록해 두세요. 키는 이 맥의 .env 파일에만 저장됩니다."
  print -n "API 키를 붙여 넣고 Enter (화면에 보이지 않아요): "
  read -rs api_key; print
  [[ -n "$api_key" ]] && set_env ANTHROPIC_API_KEY "$api_key" && ok "API 키 저장"
  unset api_key
else
  ok "API 키가 이미 있습니다."
fi

if [[ -z "$(get_env LOCATION_CITY)" ]]; then
  print "날씨와 지역 정보를 찾을 기준 도시를 알려 주세요. (건너뛰려면 그냥 Enter)"
  print -n "도시 (예: Seoul): "; read -r city
  if [[ -n "$city" ]]; then
    print -n "국가 코드 두 글자 (예: KR, US): "; read -r country
    set_env LOCATION_CITY "$city"
    [[ -n "$country" ]] && set_env LOCATION_COUNTRY "${country:u}"
    ok "사는 곳 저장: $city ${country:u}"
  fi
fi

# ------------------------------------------------------------ 5. 모델 내려받기
step "음성 모델 내려받기 (약 9GB, 처음 한 번만)"
uv run python -m speaker.prefetch || fail "모델을 내려받지 못했습니다. 다시 ./setup.sh 를 실행하면 이어서 받습니다."

# ------------------------------------------------------------ 6. 점검
step "점검"
uv run python -m speaker --check || fail "위에서 ✗ 표시된 항목을 해결한 뒤 다시 실행해 주세요."

print -P "\n%B설치가 끝났어요.%b 이제 스피커를 켜세요:\n\n  ./start.command\n"
print "처음 켤 때 터미널이 마이크 권한을 물으면 '허용'을 눌러 주세요."
print "자세한 사용법은 README.md 를 보세요."
