"""Before/after for two source states, alternating runs under the machine lock.

    pixi run python bench/compare_states.py A_root B_root repeat case...

Each root has ``lib.so`` and ``python/``.  A is run, then B, in separate
processes, three times over, and the median of the three process medians is
reported with the best of all minima.  Alternating keeps a drifting host
roughly common to both sides.
"""

import os
import statistics
import subprocess
import sys


def run(root, case, repeat):
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(
        [sys.executable, os.path.join(here, "state.py"), root, case, str(repeat)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return float(out[0]), float(out[1])


def main() -> int:
    a_root, b_root, repeat = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cases = sys.argv[4:]
    print("| case | before med | before min | after med | after min |"
          " after/before med | after/before min |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for case in cases:
        am, an, bm, bn = [], [], [], []
        for _ in range(3):
            m, n = run(a_root, case, repeat)
            am.append(m)
            an.append(n)
            m, n = run(b_root, case, repeat)
            bm.append(m)
            bn.append(n)
        amed, amin = statistics.median(am), min(an)
        bmed, bmin = statistics.median(bm), min(bn)
        print(f"| {case} | {amed:.3f} | {amin:.3f} | {bmed:.3f} | {bmin:.3f} |"
              f" {bmed / amed:.2f}x | {bmin / amin:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
