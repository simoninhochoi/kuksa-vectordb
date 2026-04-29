"""Windows 절전·디스플레이 절전 차단 (장시간 인덱싱·패치 작업용).

실행 중인 동안만 시스템·디스플레이 절전이 차단됨. Ctrl+C 또는 프로세스 종료 시 자동 해제.
사용:
  uv run python src/keep_awake.py &
  # 작업 끝나면 taskkill 또는 Ctrl+C
"""
from __future__ import annotations

import ctypes
import signal
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

_AWAKE_FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED


def set_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(_AWAKE_FLAGS)


def reset() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _on_signal(signum, frame):
    print(f"[keep_awake] signal {signum} → resetting", flush=True)
    reset()
    sys.exit(0)


def main() -> int:
    set_awake()
    print(f"[keep_awake] PID {ctypes.windll.kernel32.GetCurrentProcessId()} "
          f"holding ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED. "
          f"Stop with: taskkill //PID <pid> //F", flush=True)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    # OS가 ES_CONTINUOUS 상태를 무한 유지하므로 sleep 짧게 반복하며 시그널 대기
    try:
        while True:
            time.sleep(60)
            # 안전 차원에서 재요청 (다른 프로세스가 reset 했을 가능성 차단)
            set_awake()
    except KeyboardInterrupt:
        pass
    finally:
        reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
