import cProfile
import pstats
from pathlib import Path
from analyzer.log_analyzer import analyze_logs

LOG = Path(__file__).resolve().parents[1] / "data" / "logs" / "stdout.log.20260407.txt"

if __name__ == '__main__':
    raw = ""
    if LOG.exists():
        raw = LOG.read_text(encoding='utf-8')
    else:
        print('No log file found at', LOG)

    pr = cProfile.Profile()
    pr.enable()
    analyze_logs(raw)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(50)
