"""提出物（動画と説明資料 PDF）を作る。

    .venv/bin/python tools/build_submission.py --account <omnicampusアカウント名> \
        [--video-url URL] [--repo-url URL] [--skip-video] [--skip-pdf]

入力:
  docs/submission/{slides,report,diagram}.html   スライドと説明資料の元
  results/main_*/results.jsonl                   本評価の結果（グラフ用）
  outputs/colab/demo_*/videos/*.mp4              デモの撮影（colab_driver.py で回収したもの）
出力（outputs/submission/）:
  PAI最終課題_<account>.mp4                       提出動画（1920x1080、30fps、等倍速）
  説明資料_PAI最終課題_<account>.pdf              説明資料

HTML の描画と PDF への印刷には Windows 側の Microsoft Edge をヘッドレスで使う（WSL 前提）。
別の Chromium 系ブラウザを使うときは環境変数 PAI_BROWSER に実行ファイルを指定する。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import av
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "submission"
DEMO = ROOT / "outputs" / "colab"
OUT = ROOT / "outputs" / "submission"
BUILD = OUT / "build"
sys.path.insert(0, str(ROOT))
from runner.analyze import success_table  # noqa: E402

BROWSER = os.environ.get("PAI_BROWSER", "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
FONTS = [os.environ.get("PAI_FONT", ""), "/mnt/c/Windows/Fonts/NotoSansJP-VF.ttf",
         "/usr/share/fonts/truetype/noto/NotoSansJP-VF.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]

W, H, FPS, SRC_FPS = 1920, 1080, 30, 20
BG = (16, 18, 22)
INK, INK2, INK3 = (243, 243, 241), (169, 173, 181), (111, 116, 125)
BASE, MON, OK, HELP, BAD = (75, 143, 224), (235, 104, 52), (70, 183, 104), (180, 124, 224), (120, 124, 132)

# 動画のデモ場面。step は撮影した各 step（20 step = 1 秒）。pause はその step で両方を止めて説明を出す
SCENES = [
    dict(name="demo_sp_miss", key="libero_spatial_t5_i5", d="n", title="自然に起きた空振り", suite="LIBERO-Spatial",
         task="ラメキンの上の黒いボウルを取って、皿に置く",
         caps=[(0, "左右は同じ初期状態・同じ乱数。違いは監視層の有無だけ"),
               (30, "VLA がボウルに手を伸ばして掴みにいく"),
               (51, "右：開く → 上がる → 少しずらして、VLA に任せ直す"),
               (130, "右はやり直して成功。左は同じ空振りをくり返して時間切れ")],
         pauses={50: (2.0, "空振り！ 右の監視層は、閉じきった指の開き幅（3 mm 未満）で気づく")}),
    dict(name="demo_sp_dist", key="libero_spatial_t3_i9", d="d", title="持ち上げた物を取り上げる妨害", suite="LIBERO-Spatial",
         task="クッキー箱の上の黒いボウルを取って、皿に置く",
         caps=[(0, "持ち上げた瞬間に、物をシミュレータで元の場所へ戻す（人が取り上げるのを模す）"),
               (30, "ボウルを掴んで持ち上げる"),
               (62, "右：落下に気づいて、退避してから取り直す"),
               (140, "右は成功。左は取り直しても同じ所で外し続ける")],
         pauses={58: (2.0, "妨害：ボウルが元の場所に戻され、手は空になった"),
                 61: (1.6, "右の監視層：持っていた指が閉じきった → 落とした")}),
    dict(name="demo_ob_miss", key="libero_object_t1_i6", d="n", title="自然に起きた空振り", suite="LIBERO-Object",
         task="クリームチーズを取って、かごに入れる",
         caps=[(0, "別のタスクでも、同じ初期状態・同じ乱数で比べる"),
               (72, "右：開く → 上がる → 少しずらして、VLA に任せ直す"),
               (185, "右はやり直して成功。左は空振りから立て直せず時間切れ")],
         pauses={71: (2.0, "空振り！ 右の監視層が気づく")}),
    dict(name="demo_ob_dist", key="libero_object_t1_i5", d="d", title="持ち上げた物を取り上げる妨害", suite="LIBERO-Object",
         task="クリームチーズを取って、かごに入れる",
         caps=[(0, "持ち上げた瞬間に、チーズを元の場所へ戻す"),
               (73, "右：落下に気づいて、退避してから取り直す"),
               (190, "右は成功。左は空の手のまま運び続ける")],
         pauses={64: (2.0, "妨害：チーズが元の場所に戻され、手は空になった"),
                 72: (1.6, "右の監視層：落とした と判定")}),
    dict(name="demo_sp_help", key="libero_spatial_t1_i9", d="n", title="どうしても掴めないとき", suite="LIBERO-Spatial",
         task="ラメキンの隣の黒いボウルを取って、皿に置く",
         caps=[(0, "掴みにくい配置。どちらも空振りをくり返す"),
               (69, "右：1 回目のやり直し"),
               (131, "右：2 回目のやり直し"),
               (192, "右は止まって助けを求めた（HELP）。左は時間切れまで空振りを続ける")],
         pauses={191: (2.2, "2 回やり直しても駄目 → 無限に再試行せず、安全に止まる")}),
]
SLIDES = [("title", 4.5), ("problem", 8.0), ("method", 11.0), None, ("results", 13.0), ("summary", 9.0)]  # None = デモ


# ---------------------------------------------------------------- 図

def chart_svg(rows: list[dict]) -> str:
    """成功率の棒グラフ。色は差し込み先の CSS 変数（--base / --mon / --ink 系）に従う。"""
    idx = {(r["suite"], r["disturb"], r["cond"]): r for r in rows}
    groups = [("libero_spatial", False), ("libero_object", False), ("libero_spatial", True), ("libero_object", True)]
    name = {"libero_spatial": "Spatial", "libero_object": "Object"}
    w, h, l, r, t, b = 1000, 560, 70, 10, 50, 86
    pw, ph = w - l - r, h - t - b
    y = lambda v: t + ph * (1 - v / 100)
    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="inherit">']
    for v in (0, 25, 50, 75, 100):
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{y(v):.1f}" y2="{y(v):.1f}" style="stroke:var(--line)" stroke-width="1"/>')
        out.append(f'<text x="{l - 12}" y="{y(v) + 7:.1f}" text-anchor="end" font-size="20" style="fill:var(--ink2)">{v}%</text>')
    gw, bw, gap = pw / len(groups), 74, 6
    for i, (suite, dist) in enumerate(groups):
        cx = l + gw * (i + 0.5)
        for j, cond in enumerate(("base", "monitor")):
            row = idx[(suite, dist, cond)]
            x = cx - bw - gap / 2 if j == 0 else cx + gap / 2
            v, lo, hi = row["rate"] * 100, row["ci95"][0] * 100, row["ci95"][1] * 100
            color = "var(--base)" if cond == "base" else "var(--mon)"
            out.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw}" height="{y(0) - y(v):.1f}" rx="3" style="fill:{color}"/>')
            m = x + bw / 2
            out.append(f'<path d="M{m:.1f} {y(lo):.1f}V{y(hi):.1f}M{m - 8:.1f} {y(lo):.1f}h16M{m - 8:.1f} {y(hi):.1f}h16" '
                       f'style="stroke:var(--ink2)" stroke-width="1.6" fill="none"/>')
            out.append(f'<text x="{m:.1f}" y="{y(hi) - 10:.1f}" text-anchor="middle" font-size="22" font-weight="700" '
                       f'style="fill:var(--ink)">{v:.0f}%</text>')
        out.append(f'<text x="{cx:.1f}" y="{h - b + 34}" text-anchor="middle" font-size="22" style="fill:var(--ink)">{name[suite]}</text>')
        out.append(f'<text x="{cx:.1f}" y="{h - b + 62}" text-anchor="middle" font-size="19" style="fill:var(--ink2)">'
                   f'{"妨害あり" if dist else "妨害なし"}</text>')
    for k, (label, color) in enumerate((("素の VLA", "var(--base)"), ("VLA ＋ 監視層", "var(--mon)"))):
        x0 = l + 10 + k * 190
        out.append(f'<rect x="{x0}" y="8" width="20" height="20" rx="3" style="fill:{color}"/>')
        out.append(f'<text x="{x0 + 30}" y="26" font-size="21" style="fill:var(--ink)">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def frame(path: Path, idx: int) -> Image.Image:
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == idx:
                return f.to_image()
    raise IndexError(idx)


def video_path(scene_name: str, key: str, cond: str, d: str) -> Path:
    return DEMO / scene_name / "videos" / f"{key}_{cond}_{d}.mp4"


def make_assets() -> None:
    """スライドと説明資料が参照する画像を build/ に置く。撮影画像の左半分（俯瞰カメラと状態バー）を切り出す。"""
    BUILD.mkdir(parents=True, exist_ok=True)
    frame(video_path("demo_ob_dist", "libero_object_t1_i5", "base", "d"), 110).crop((0, 68, 720, 788)).save(BUILD / "problem.png")
    src = video_path("demo_sp_miss", "libero_spatial_t5_i5", "monitor", "n")
    for i, f in enumerate((45, 52, 74, 150)):
        frame(src, f).crop((0, 0, 720, 788)).resize((480, 525), Image.LANCZOS).save(BUILD / f"strip{i}.jpg", quality=90)


FILMSTRIP = [("① 閉じたが何も挟んでいない", "VLA はそのまま運ぼうとする"),
             ("② 空振りを検知", "閉じきった指の開き幅が 3 mm 未満"),
             ("③ 開いて上がり、少しずらす", "同じ状態からの再開を避ける"),
             ("④ VLA に任せ直して成功", "やり直しは 1 回")]


def fill(template: str, links: str = "") -> str:
    s = (SRC / template).read_text()
    rows = success_table([json.loads(l) for d in sorted((ROOT / "results").glob("main_*"))
                          for l in (d / "results.jsonl").read_text().splitlines()])
    strip = "\n".join(f'<div><img src="strip{i}.jpg" alt=""><b>{a}</b><br>{b}</div>' for i, (a, b) in enumerate(FILMSTRIP))
    return (s.replace("<!--DIAGRAM-->", (SRC / "diagram.html").read_text())
             .replace("<!--CHART-->", chart_svg(rows))
             .replace("<!--FILMSTRIP-->", strip)
             .replace("<!--LINKS-->", links))


# ---------------------------------------------------------------- ブラウザ（Windows 側の Edge）

def win_temp() -> Path:
    t = subprocess.run(["cmd.exe", "/c", "echo %TEMP%"], capture_output=True, text=True, cwd="/mnt/c").stdout.strip()
    return Path(subprocess.run(["wslpath", "-u", t], capture_output=True, text=True).stdout.strip()) / "pai_submission"


def wpath(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True).stdout.strip()


def browser(work: Path, *args: str) -> None:
    # 普段使いのプロファイルに触れないよう、作業用のプロファイルで起動する
    subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                    "--no-default-browser-check", "--disable-extensions", "--disable-sync",
                    f"--user-data-dir={wpath(work / 'profile')}", *args],
                   check=True, capture_output=True, timeout=180)


def url(p: Path, frag: str = "") -> str:
    return "file:///" + wpath(p).replace("\\", "/") + (f"#{frag}" if frag else "")


def render_html(slides: bool, pdf_name: str | None, links: str) -> None:
    work = win_temp()
    if work.exists():
        shutil.rmtree(work / "site", ignore_errors=True)
    shutil.copytree(BUILD, work / "site", dirs_exist_ok=True)
    if slides:
        (work / "site" / "slides.html").write_text(fill("slides.html"))
        for item in SLIDES:
            if item is None:
                continue
            sid = item[0]
            browser(work, "--window-size=1920,1080", "--virtual-time-budget=3000",
                    f"--screenshot={wpath(work / 'site' / f'slide_{sid}.png')}", url(work / "site" / "slides.html", sid))
            shutil.copy(work / "site" / f"slide_{sid}.png", BUILD / f"slide_{sid}.png")
    if pdf_name:
        (work / "site" / "report.html").write_text(fill("report.html", links))
        browser(work, "--no-pdf-header-footer", "--virtual-time-budget=3000",
                f"--print-to-pdf={wpath(work / 'site' / 'report.pdf')}", url(work / "site" / "report.html"))
        shutil.copy(work / "site" / "report.pdf", OUT / pdf_name)
        (BUILD / "report.html").write_text(fill("report.html", links))


# ---------------------------------------------------------------- 動画

def font(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    for p in FONTS:
        if p and Path(p).exists():
            f = ImageFont.truetype(p, size)
            try:
                f.set_variation_by_axes([weight])
            except OSError:
                pass  # 可変フォントでなければそのまま
            return f
    raise FileNotFoundError("日本語フォントが見つからない。PAI_FONT に .ttf / .ttc を指定する")


class Clip:
    """20fps の撮影を、指定した step 番号のフレームとして順に取り出す（先へ進むだけ）。"""

    def __init__(self, path: Path, size: tuple[int, int]):
        self.c = av.open(str(path))
        self.n = self.c.streams.video[0].frames
        self.it = self.c.decode(video=0)
        self.size, self.i, self.img = size, -1, None

    def get(self, idx: int) -> Image.Image:
        idx = min(idx, self.n - 1)
        while self.i < idx:
            f = next(self.it)
            self.i += 1
            if self.i == idx:
                self.img = f.to_image().resize(self.size, Image.BILINEAR)
        return self.img

    def close(self) -> None:
        self.c.close()


def text_center(d: ImageDraw.ImageDraw, cx: int, y: int, s: str, f, fill) -> None:
    d.text((cx - d.textlength(s, font=f) / 2, y), s, font=f, fill=fill)


def badge(img: Image.Image, x: int, y: int, w: int, s: str, color) -> None:
    f = font(46, 700)
    d = ImageDraw.Draw(img, "RGBA")
    tw = d.textlength(s, font=f)
    bx0, by0 = x + (w - tw) / 2 - 34, y
    d.rounded_rectangle([bx0, by0, bx0 + tw + 68, by0 + 82], radius=14, fill=color + (235,))
    d.text((bx0 + 34, by0 + 10), s, font=f, fill=(255, 255, 255))


PW, PH = 936, 512          # 1 本の撮影（1440x788）を縮めた大きさ
PX = (16, W - 16 - PW)
PY = 236


def scene_background(sc: dict, no: int, total: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((48, 40), f"デモ {no} / {total}", font=font(30, 700), fill=MON)
    d.text((48, 80), sc["title"], font=font(56, 700), fill=INK)
    d.text((48 + d.textlength(sc["title"], font=font(56, 700)) + 36, 100), f"{sc['suite']}　指示：{sc['task']}",
           font=font(30), fill=INK2)
    for x, (label, color) in zip(PX, (("素の VLA（監視なし）", BASE), ("VLA ＋ 監視層", MON))):
        d.ellipse([x + 4, 190, x + 28, 214], fill=color)
        d.text((x + 42, 180), label, font=font(34, 700), fill=INK)
    d.text((48, H - 50), "シミュレータ：LIBERO（MuJoCo）　方策：lerobot/smolvla_libero（公開重み・推論のみ）　等倍速",
           font=font(24), fill=INK3)
    return img


def scene_frames(sc: dict, no: int, total: int):
    bg = scene_background(sc, no, total)
    base = Clip(video_path(sc["name"], sc["key"], "base", sc["d"]), (PW, PH))
    mon = Clip(video_path(sc["name"], sc["key"], "monitor", sc["d"]), (PW, PH))
    # 出力フレームごとに (step, 一時停止中の説明) を並べる
    timeline: list[tuple[int, str | None]] = []
    for step in range(max(base.n, mon.n) + SRC_FPS // 2):
        if step in sc["pauses"]:
            sec, msg = sc["pauses"][step]
            timeline += [(step, msg)] * int(sec * FPS)
        timeline += [(step, None)] * (FPS // SRC_FPS) + ([(step, None)] if step % 2 else [])  # 20fps → 30fps
    f_cap, f_pause = font(44, 700), font(28, 700)
    for step, paused in timeline:
        img = bg.copy()
        img.paste(base.get(step), (PX[0], PY))
        img.paste(mon.get(step), (PX[1], PY))
        if step >= mon.n - SRC_FPS:  # 撮影の最後の 1 秒は結果の表示
            ok = "SUCCESS" if not sc["name"].endswith("help") else "HELP"
            badge(img, PX[1], PY + PH - 110, PW, "成功" if ok == "SUCCESS" else "停止して助けを求める（HELP）",
                  OK if ok == "SUCCESS" else HELP)
        if step >= base.n - SRC_FPS:
            badge(img, PX[0], PY + PH - 110, PW, "失敗（時間切れ）", BAD)
        d = ImageDraw.Draw(img)
        cap = paused or [c for s, c in sc["caps"] if s <= step][-1]
        y = PY + PH + 60
        if paused:
            hw = d.textlength("一時停止して説明", font=f_pause) / 2 + 22
            d.rounded_rectangle([W / 2 - hw, y - 8, W / 2 + hw, y + 40], radius=8, fill=(60, 64, 72))
            text_center(d, W // 2, y - 4, "一時停止して説明", f_pause, INK)
            y += 66
        text_center(d, W // 2, y, cap, f_cap, MON if paused else INK)
        yield img
    base.close()
    mon.close()


def fade(frames, n: int = 8):
    """場面の始まりを背景色から立ち上げる。"""
    bg = Image.new("RGB", (W, H), BG)
    for i, f in enumerate(frames):
        yield Image.blend(bg, f, (i + 1) / (n + 1)) if i < n else f


def still(path: Path, sec: float):
    img = Image.open(path).convert("RGB")
    assert img.size == (W, H), (path, img.size)
    for _ in range(int(sec * FPS)):
        yield img


def build_video(out: Path) -> float:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.Popen([ff, "-loglevel", "error", "-y",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                          "-shortest", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = 0
    for item in SLIDES:
        parts = ([still(BUILD / f"slide_{item[0]}.png", item[1])] if item else
                 [scene_frames(sc, i + 1, len(SCENES)) for i, sc in enumerate(SCENES)])
        for part in parts:
            for f in fade(part):
                p.stdin.write(f.tobytes())
                n += 1
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return n / FPS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, help="omnicampus のアカウント名（個人名なら アカウント ID）")
    ap.add_argument("--video-url", default="")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    make_assets()
    links = (f'<span><b>動画</b>{args.video_url or "提出フォームに記載"}</span>'
             f'<span><b>コード</b>{args.repo_url or "提出フォームに記載"}</span>')
    render_html(slides=not args.skip_video, pdf_name=None if args.skip_pdf else f"説明資料_PAI最終課題_{args.account}.pdf",
                links=links)
    if not args.skip_video:
        mp4 = OUT / f"PAI最終課題_{args.account}.mp4"
        sec = build_video(mp4)
        print(f"wrote {mp4}  ({sec:.1f} s)")
    if not args.skip_pdf:
        print(f"wrote {OUT / f'説明資料_PAI最終課題_{args.account}.pdf'}")


if __name__ == "__main__":
    main()
