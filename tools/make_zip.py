"""Colab へ持っていく zip を作る（git に載せる予定のファイルだけ。.gitignore の対象は入れない）。

    python tools/make_zip.py            # → outputs/pai-final.zip（Google Drive の pai_final/ に置く）
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                       cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
out = ROOT / "outputs" / "pai-final.zip"
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(ROOT / f, f)
print(f"wrote {out} ({len(files)} files)")
