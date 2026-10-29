"""公開前の検査: PARC2026 由来のファイル・重み・語句がリポジトリに混じっていないか。

    python tests/check_no_parc.py

git 管理下のファイルの中身と名前を、PARC 由来を示す語句で検索する。1件でも見つかれば終了コード 1。
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [
    r"parc", r"PARC", r"LIBERO[-_]?plus", r"libero_plus", r"Sylvest", r"openvla", r"OpenVLA", r"\bOFT\b",
    r"pi05_libero", r"policy_server", r"submission_", r"parc_kit", r"PARC_TEST", r"PARC_資料",
    r"Track ?[123]\b", r"\bT3\b", r"check_no_leaks", r"ckpt_16000", r"prismatic", r"Moo Jin",
]
# 検査スクリプト自身と、混入防止を説明する文書は除く
ALLOW = {"tests/check_no_parc.py"}
BIG = 20 * 1024 * 1024  # 重みのような大きいファイルが紛れ込んでいないか


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def main() -> int:
    bad = []
    rx = re.compile("|".join(PATTERNS))
    for f in tracked_files():
        if f in ALLOW:
            continue
        p = ROOT / f
        if rx.search(f):
            bad.append(f"{f}: ファイル名が一致")
        if p.stat().st_size > BIG:
            bad.append(f"{f}: {p.stat().st_size / 1e6:.0f}MB（大きすぎる）")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = rx.search(line)
            if m:
                bad.append(f"{f}:{i}: '{m.group(0)}' → {line.strip()[:100]}")
    if bad:
        print("★ PARC 由来の可能性があるもの:")
        print("\n".join("  " + b for b in bad))
        return 1
    print(f"OK: {len(tracked_files())} ファイルに PARC 由来の語句・大きなファイルは無い")
    return 0


if __name__ == "__main__":
    sys.exit(main())
