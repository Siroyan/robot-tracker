# robot-tracker

水中ロボット動画から、オレンジ色のスラスタを手掛かりにロボット位置をフレーム単位で追跡するツールです。
プール四隅の画素座標と実寸を設定すると、射影変換によってピクセル座標をメートル座標へ変換できます。

現状の実装は GUI なしのヘッドレス実行です。主要処理は [`main.py`](./main.py) に集約されています。

## 機能

- 動画からオレンジ色領域を HSV 閾値で抽出
- 輪郭検出による候補抽出
- 初期点と前フレーム位置を使ったロボット候補の選定
- 近傍候補のクラスタ平均による位置推定
- 平滑化による位置の安定化
- プール四隅と実寸に基づくメートル座標への変換
- フレームごとの CSV 出力
- 注釈付き動画の出力
- 参照フレーム画像、オレンジ候補プレビュー画像の出力

## 動作環境

- Python 3.12 以上

依存パッケージ:

- `opencv-python-headless`
- `numpy`

## セットアップ

`pip` を使う場合:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_tracker.txt
```

`uv` を使う場合:

```bash
uv sync
```

注意:

- `pyproject.toml` の `dependencies` は未記載です。
- 現状の依存パッケージ定義は [`requirements_tracker.txt`](./requirements_tracker.txt) ベースです。

## 典型的な使い方

### 1. 参照フレームを書き出す

設定に使う基準フレームを画像として保存します。

```bash
python3 main.py movie/20260509_173657.mp4 \
  --reference-frame 0 \
  --export-reference-frame reference.jpg
```

### 2. オレンジ候補を確認する

HSV 閾値でどの領域が候補になるかを画像で確認します。

```bash
python3 main.py movie/20260509_173657.mp4 \
  --config tracker_config_example.json \
  --reference-frame 0 \
  --export-orange-preview orange_preview.jpg
```

必要に応じて CLI から HSV を上書きできます。

```bash
python3 main.py movie/20260509_173657.mp4 \
  --config tracker_config_example.json \
  --hsv-lower 5 80 50 \
  --hsv-upper 30 255 255 \
  --export-orange-preview orange_preview.jpg
```

### 3. 設定ファイルを作る

GUI はないため、参照画像を見ながらプール四隅と初期位置を手入力します。

```bash
python3 main.py movie/20260509_173657.mp4 \
  --make-config tracker_config.json \
  --reference-frame 0 \
  --pool-width-m 2.0 \
  --pool-height-m 3.0 \
  --pool-corners-px 145,236 585,237 913,1097 -173,1088 \
  --init-point-px 420,360
```

`--pool-corners-px` の順序は必ず以下です。

1. top-left
2. top-right
3. bottom-right
4. bottom-left

設定例は [`tracker_config_example.json`](./tracker_config_example.json) を参照してください。

### 4. トラッキングを実行する

```bash
python3 main.py movie/20260509_173657.mp4 \
  --config tracker_config.json \
  --csv positions.csv \
  --annotated annotated.mp4
```

処理後に以下が出力されます。

- 追跡結果 CSV
- 任意で注釈付き MP4
- 標準出力に総フレーム数、検出率、メートル座標範囲

## 出力 CSV

CSV には以下の列が出力されます。

- `frame`: フレーム番号
- `time_s`: 動画先頭からの経過秒
- `detected`: そのフレームで位置が確定したか
- `px_x`, `px_y`: 平滑化後の画素座標
- `pool_x_m`, `pool_y_m`: 射影変換後のプール座標
- `speed_mps`: 前回有効点からの速度
- `orange_area_px2`: 採用クラスタの面積合計
- `cluster_contours`: 採用クラスタに含まれた輪郭数
- `num_orange_candidates`: そのフレームで見つかったオレンジ候補数

プール四隅またはプール実寸が未設定の場合、`pool_x_m`、`pool_y_m`、`speed_mps` は `NaN` になります。

## 設定項目

[`TrackerConfig`](./main.py) で定義されている主な設定値です。

- `hsv_lower`, `hsv_upper`
  - オレンジ抽出用の HSV 閾値
- `min_area_px`, `max_area_px`
  - 輪郭候補として採用する面積範囲
- `cluster_radius_px`
  - 選択した候補の近傍輪郭をまとめる半径
- `max_jump_px`
  - 前フレーム予測位置から許容する最大移動量
- `smoothing_alpha`
  - 平滑化係数
- `pool_corners_px`
  - プール領域の四隅画素座標
- `pool_width_m`, `pool_height_m`
  - プール実寸
- `init_point_px`
  - 初期フレームでロボットを選ぶための基準点
- `reference_frame`
  - 設定作成時の基準フレーム番号

## CLI オプション

主要オプション:

- `video`
  - 入力動画パス
- `--config`
  - JSON 設定ファイル
- `--make-config`
  - 設定ファイルを生成して終了
- `--reference-frame`
  - 参照フレーム番号
- `--export-reference-frame`
  - 参照フレーム画像を書き出して終了
- `--export-orange-preview`
  - オレンジ候補付き画像を書き出して終了
- `--pool-corners-px`
  - プール四隅の画素座標
- `--init-point-px`
  - 初期ロボット位置
- `--pool-width-m`, `--pool-height-m`
  - プール実寸
- `--csv`
  - CSV 出力先
- `--annotated`
  - 注釈付き動画出力先
- `--hsv-lower`, `--hsv-upper`
  - HSV 閾値の CLI 上書き

## 実装上の制約

- 色ベース追跡なので、水面反射や照明変化、類似色の物体に影響を受けます。
- `init_point_px` が未設定だと、最初のフレームでは最大のオレンジ領域をロボットとみなします。
- ロボットが急に大きく移動した場合、`max_jump_px` を超えると未検出になります。
- トラッキングは単一オブジェクト前提です。
- 現状はテストコードがありません。

## ファイル構成

- [`main.py`](./main.py)
  - 追跡処理、設定処理、CSV/動画出力、CLI をまとめた本体
- [`tracker_config_example.json`](./tracker_config_example.json)
  - 設定例
- [`requirements_tracker.txt`](./requirements_tracker.txt)
  - 依存パッケージ一覧
- [`positions_pixels.csv`](./positions_pixels.csv)
  - 出力例
- `movie/`
  - 入力動画や生成動画
