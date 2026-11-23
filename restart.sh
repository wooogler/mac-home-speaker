#!/bin/zsh
# 켜져 있는 스피커를 다시 시작한다 (start.command 로 켠 경우). 원격으로 작업할 때 쓴다.
cd "$(dirname "$0")" || exit 1
pid_file=logs/speaker.pid
if [[ ! -f $pid_file ]] || ! kill -0 "$(cat $pid_file)" 2>/dev/null; then
  echo "켜져 있는 스피커가 없습니다. ./start.command 로 켜 주세요."
  exit 1
fi
kill -HUP "$(cat $pid_file)" && echo "재시작을 요청했습니다. 모델을 불러오느라 30초쯤 걸립니다. (로그: tail -f logs/speaker.log)"
