# scripts/compare_retention.py
import sys
from pathlib import Path
from collections import defaultdict

PROBLEM = {204, 233, 241, 245, 252, 259, 276, 278}

def stats(det_dir, trk_dir):
    rows = []
    for det in sorted(Path(det_dir).glob("*.txt")):
        seq = det.stem
        n_det = sum(1 for _ in det.open())
        trk = Path(trk_dir) / f"{seq}.txt"
        ids, n_trk = set(), 0
        if trk.exists():
            for line in trk.open():
                n_trk += 1
                ids.add(int(line.split(",")[1]))
        ret = 100 * n_trk / n_det if n_det else 0
        num = int("".join(c for c in seq if c.isdigit()))
        rows.append((seq, n_det, n_trk, ret, len(ids), num in PROBLEM))
    return rows

if __name__ == "__main__":
    det_dir, trk_dir = sys.argv[1], sys.argv[2]
    rows = sorted(stats(det_dir, trk_dir), key=lambda r: r[3])
    print(f"{'seq':<16}{'#det':>8}{'#trk':>8}{'ret%':>7}{'#ids':>6}  cluster")
    for seq, nd, nt, ret, nid, prob in rows:
        print(f"{seq:<16}{nd:>8}{nt:>8}{ret:>7.1f}{nid:>6}  {'*' if prob else ''}")
