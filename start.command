#!/bin/zsh
# 더블클릭하거나 로그인 항목에 등록해 스피커를 켠다.
# caffeinate 로 감싸서 실행 중에는 맥이 잠들지 않게 한다 (시스템 설정은 바꾸지 않는다).
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

while true; do
  caffeinate -is uv run python -m speaker "$@"
  code=$?
  if [[ $code -eq 75 ]]; then
    # 코드가 바뀌었거나 ./restart.sh 로 재시작을 요청함: 곧바로 다시 켠다
    echo "스피커를 다시 시작합니다..."
    continue
  fi
  # 0(정상 종료)이나 130(Ctrl-C)이면 끝내고, 그 밖의 오류로 죽으면 잠시 뒤 다시 켠다.
  if [[ $code -eq 0 || $code -eq 130 ]]; then
    break
  fi
  echo "스피커가 종료 코드 $code 로 멈췄습니다. 10초 뒤 다시 시작합니다. (그만두려면 Ctrl-C)"
  sleep 10 || break
done
