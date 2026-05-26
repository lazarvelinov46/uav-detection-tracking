# scripts/patch_trackeval.py
"""
Patch a freshly cloned TrackEval for modern NumPy: np.float/np.int/np.bool were
removed in NumPy >= 1.24. Run once after cloning TrackEval. Idempotent.
"""
import re
from pathlib import Path

ROOT = Path("TrackEval/trackeval")
SUBS = [(r"\bnp\.float\b", "float"),
        (r"\bnp\.int\b", "int"),
        (r"\bnp\.bool\b", "bool")]


def main():
    if not ROOT.exists():
        raise SystemExit(f"{ROOT} not found - clone TrackEval into the project root first")
    n = 0
    for p in ROOT.rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        new = s
        for pat, repl in SUBS:
            new = re.sub(pat, repl, new)
        if new != s:
            p.write_text(new, encoding="utf-8")
            print("patched", p.relative_to(ROOT.parent))
            n += 1
    print(f"Done: {n} file(s) patched")


if __name__ == "__main__":
    main()
