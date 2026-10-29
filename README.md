# しくじりに気づくロボット — VLA に後付けする失敗検知・回復レイヤ

Physical AI 応用1 最終課題。

VLA（視覚・言語・行動モデル）は自分の失敗に気づかない。空振りしたまま空の手で運び続けたり、
同じ失敗を繰り返したりする。本作品では、**どの VLA にも外付けできる監視層**を作り、
失敗に気づいて立て直すことで成功率がどれだけ戻るかを、シミュレータで定量的に示す。

- ロボット / 環境: Franka Panda / [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)（MuJoCo、lerobot 同梱版）
- 方策: 公開 VLA [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero)（Apache-2.0、推論のみ。追加学習なし）
- 実行: Google Colab（T4）

## 仕組み

```
LIBERO ──観測──▶ MonitoredController ──行動──▶ LIBERO
                  │  L1 把持失敗: 閉じ指令を出して指が止まったとき、開き幅が閾値未満 → 空振り / 落下
                  │  L2 停滞・迷走: 直近 60 step の手先の移動範囲が小さい / グリッパの開閉を繰り返す
                  ▼
                回復: 開く → 8cm 上がる → 水平に 1〜2.5cm ランダムにずらす → VLA の行動キューを捨てて再開
                      2 回やり直しても駄目なら HELP（止まって助けを求める）
```

- 監視層が見るのは**ロボット自身が観測できる量だけ**（手先位置・指の位置・直前の行動）。
  シミュレータの正解（物体の位置、指と物体の接触）は**採点と妨害の演出にだけ**使う
- 監視層は方策の外側にあるので、VLA を差し替えても同じコードで動く
- ずらしてから再開する理由: 掴めるかどうかは mm 単位の状態差で決まることが多く、
  同じ状態に戻すと VLA は同じ行動を出して同じ失敗を繰り返す

### 妨害テスト
ロボットが物を持ち上げた瞬間に、シミュレータで物を元の場所へ戻す（人が手から取り上げて置き直すのを模す）。
素の VLA は空の手で運び続け、監視層ありでは落下に気づいて取り直す。

## 結果

（Colab での本評価のあとに記入）

## 構成

```
monitor/            監視層（シミュレータに依存しない）
  detectors.py        L1 把持失敗 / L2 停滞・迷走
  recovery.py         退避 → ずらす
  wrapper.py          MonitoredController（方策の外側に付ける）
runner/             LIBERO での実験
  libero_setup.py     LIBERO の設定ファイルを先に書き、初回 import 時の対話を飛ばす
  policy.py           公開 VLA の読み込み（前処理は lerobot-eval と同じ経路）
  sim_tools.py        正解判定と妨害（評価専用）
  run_eval.py         条件をそろえた比較実験。中断しても再開できる
  analyze.py          成功率・検知の精度・閾値の較正
  overlay.py          動画用のフレーム合成
notebooks/
  run_colab.ipynb     Colab での実行手順（build_notebook.py で生成）
tests/                監視層の単体テスト
```

## 再現手順

Colab（T4）で `notebooks/run_colab.ipynb` を上から実行する。ローカル（Linux、CPU でも動くが遅い）では:

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/python -m pytest tests
.venv/bin/python -m runner.run_eval --suite libero_spatial --tasks 0 --episodes 1 --conditions base,monitor --out outputs/smoke --video all
.venv/bin/python -m runner.analyze outputs/smoke
```

## 使用した公開物とライセンス

| 名前 | 用途 | ライセンス |
|---|---|---|
| lerobot 0.6.1 | 方策の読み込み・LIBERO 環境 | Apache-2.0 |
| lerobot/smolvla_libero | VLA（推論のみ） | Apache-2.0 |
| LIBERO（hf-libero 0.1.4） | シミュレーション環境 | MIT |
| lerobot/libero-assets | LIBERO の物体・テクスチャ（初回に自動取得） | データセットカードに記載なし（LIBERO 付属アセット） |
