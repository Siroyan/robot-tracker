# robot-tracker

水中ロボット動画からオレンジ色のスラスタを検出し、各フレームのロボット位置を CSV と注釈動画として出力するツールです。
プール四隅の画素座標と実寸を設定すると、射影変換で画素座標をメートル座標に変換できます。

現状は GUI なしのヘッドレス実行です。`main.py` は薄いエントリポイントで、実装本体は `src/` 配下に分割しています。

## 機能

- 動画からオレンジ色領域を抽出
- 初期フレームからスラスタ点を自動初期化
- スラスタごとの個別 ROI 追跡
- ROI 内で見失った場合の `hold` 表示
- 全体再推定に落ちた場合の `global` 表示
- `num_thrusters` に応じた可変スラスタ数対応
- プール四隅と実寸に基づくメートル座標への変換
- フレームごとの CSV 出力
- 注釈付き動画出力
- 参照フレーム画像とオレンジ候補プレビュー画像の出力

## 動作環境

- Python 3.12 以上

ランタイム依存:

- `opencv-python-headless`
- `numpy`

## セットアップ

このリポジトリは `uv` で管理しています。

```bash
uv sync
```

補足:

- 依存関係は `pyproject.toml` と `uv.lock` で管理しています。

## 典型的な使い方

### 1. 参照フレームを書き出す

```bash
uv run python main.py movie/20260509_173657.mp4 \
  --reference-frame 0 \
  --export-reference-frame reference.jpg
```

### 2. オレンジ候補を確認する

```bash
uv run python main.py movie/20260509_173657.mp4 \
  --config tracker_config_example.json \
  --reference-frame 0 \
  --export-orange-preview orange_preview.jpg
```

HSV 閾値は CLI から上書きできます。

```bash
uv run python main.py movie/20260509_173657.mp4 \
  --config tracker_config_example.json \
  --hsv-lower 5 80 50 \
  --hsv-upper 30 255 255 \
  --export-orange-preview orange_preview.jpg
```

### 3. 設定ファイルを作る

```bash
uv run python main.py movie/20260509_173657.mp4 \
  --make-config tracker_config.json \
  --reference-frame 0 \
  --pool-width-m 2.0 \
  --pool-height-m 3.0 \
  --pool-corners-px 145,236 585,237 913,1097 -173,1088 \
  --water-area-corners-px 160,250 570,250 870,1040 -120,1030 \
  --init-point-px 420,360
```

`--pool-corners-px` の順序:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

`--water-area-corners-px` も同じ順序で指定します。未指定の場合は、従来どおり `pool_corners_px` の内側だけをスラスタ検出範囲として使います。

設定例は [`tracker_config_example.json`](./tracker_config_example.json) を参照してください。

### 4. トラッキングを実行する

```bash
uv run python main.py movie/20260509_173657.mp4 \
  --config tracker_config.json
```

`--csv` を省略した場合もCSVは必ず生成されます。出力先は `csv/<入力動画ファイル名>_csv.csv` です。例えば `movie/20260509_173657.mp4` なら `csv/20260509_173657_csv.csv` に保存されます。
`--annotated` を省略した場合も注釈動画は必ず生成されます。出力先は `annotation/<入力動画ファイル名>_annotation.mp4` です。例えば `movie/20260509_173657.mp4` なら `annotation/20260509_173657_annotation.mp4` に保存されます。

標準出力には以下を出します。

- 総フレーム数
- 検出率
- 有効フレームのメートル座標範囲

### 5. `movie/` 内の動画をまとめて解析する

```bash
tools/analyze_all_movies.sh
```

デフォルトでは [`tracker_config_example.json`](./tracker_config_example.json) を使います。別の設定ファイルを使う場合は、第一引数に指定します。

```bash
tools/analyze_all_movies.sh tracker_config.json
```

## 注釈動画の見方

- スラスタ番号の色
  - オレンジ: `tracking=roi`
  - 黄色: `tracking=hold`
  - 赤: `tracking=global`
- `tracking=roi`
  - 各スラスタを ROI 内で実際に再検出できた状態
- `tracking=hold`
  - ROI 内で見つからず、前フレーム位置を保持した状態
- `tracking=global`
  - 局所 ROI 追跡に失敗し、全体再推定にフォールバックした状態
- ROI 円
  - 内側の青円: 通常探索半径 `thruster_search_radius_px`
  - 外側の水色円: 再取得半径 `thruster_reacquire_radius_px`
- 四角形
  - 水色: 距離変換に使う `pool_corners_px`
  - 紫: スラスタ検出範囲に使う `water_area_corners_px`

## 追跡アルゴリズム

現在の追跡は、大きく以下の段階で構成しています。

0. 検出範囲の制限
- オレンジ抽出は、まず水面領域マスクの内側に限定します。
- `water_area_corners_px` が設定されている場合は、その四隅を検出範囲として使います。
- 未設定の場合は後方互換のため、`pool_corners_px` の内側を検出範囲として使います。
- `pool_corners_px` は引き続きメートル座標変換用の基準として使います。

1. 初期フレームのスラスタ点決定
- 初期フレームのオレンジ detection を使って、`num_thrusters` 個の初期スラスタ点を決めます。
- detection 数が不足する場合は、1 detection を `kmeans` で分割して候補を補います。
- それでも足りない場合は、オレンジマスクの `distance transform` ピーク候補を追加します。

2. ROI 中心の予測
- 各スラスタについて、前フレーム位置を基準に次フレームの ROI 中心を予測します。
- 予測には
  - そのスラスタ自身の前回移動量
  - ロボット重心の移動量
  を混ぜて使います。
- 速度ベクトルは `thruster_max_step_px` で上限制限します。

3. 個別 ROI 追跡
- 各スラスタを個別に ROI 内で探します。
- まず strict mask で探索し、見つからなければ relaxed mask でも探索します。
- ROI 内では連結成分ごとの重心を候補とし、予測位置に近い候補を採用します。
- 既に採用した点の周辺はマスクから除去し、同じ候補を別スラスタへ再利用しないようにしています。

4. hold
- ROI 内でスラスタを見つけられなかった場合、その点は前フレーム位置を保持します。
- この状態が `tracking=hold` です。

5. global フォールバック
- ROI 追跡で必要点数が揃わない場合は、フレーム全体からスラスタ点を再推定します。
- 全体再推定でも
  - `distance transform` のピーク候補
  - 必要に応じた `kmeans`
  を使います。
- この状態が `tracking=global` です。

6. ロボット位置の出力
- 採用したスラスタ点の重心をロボット位置として扱います。
- その後、`smoothing_alpha` で平滑化し、CSV と注釈動画に出力します。

## `num_thrusters`

`num_thrusters` を設定すると、スラスタ数に応じて処理と出力列数が変わります。

- `orange_preview` の番号付きマーカー表示上限
- CSV の `thruster_i_x`, `thruster_i_y` 列数
- 初期化と追跡で扱うスラスタ点数

例:

```json
{
  "num_thrusters": 5
}
```

```bash
uv run python main.py movie/20260509_173721.mp4 \
  --config /tmp/tracker_config_5.json \
  --reference-frame 1 \
  --export-orange-preview orange_preview_5.jpg
```

```bash
uv run python main.py movie/20260509_173721.mp4 \
  --config /tmp/tracker_config_5.json
```

## 出力 CSV

CSV は `--csv` の指定がない場合も生成されます。デフォルトの出力先は `csv/<入力動画ファイル名>_csv.csv` です。`--csv` を指定した場合は、そのパスへ出力します。

CSV には以下の列を出力します。

- `frame`: フレーム番号
- `time_s`: 動画先頭からの経過秒
- `detected`: そのフレームで位置が確定したか
- `tracking`: 追跡状態。`init`, `roi`, `hold`, `global`, `none` のいずれか
- `px_x`, `px_y`: 平滑化後の重心画素座標
- `pool_x_m`, `pool_y_m`: 射影変換後のプール座標
- `speed_mps`: 前回有効点からの速度
- `orange_area_px2`: そのフレームで採用したオレンジ領域の面積合計
- `cluster_contours`: 採用スラスタ点数
- `num_orange_candidates`: そのフレームで見つかったオレンジ候補数
- `thruster_min_distance_px`, `thruster_max_distance_px`: 採用スラスタ点同士の最小/最大距離
- `thruster_1_x`, `thruster_1_y` ... `thruster_N_x`, `thruster_N_y`: 各スラスタの画素座標

プール四隅またはプール実寸が未設定の場合、`pool_x_m`、`pool_y_m`、`speed_mps` は `NaN` です。

## 出力 注釈動画

注釈動画は `--annotated` の指定がない場合も生成されます。デフォルトの出力先は `annotation/<入力動画ファイル名>_annotation.mp4` です。`--annotated` を指定した場合は、そのパスへ出力します。

## 設定項目

主な設定値:

- `hsv_lower`, `hsv_upper`
  - オレンジ抽出用 HSV 閾値
- `min_area_px`, `max_area_px`
  - 輪郭候補として採用する面積範囲
- `cluster_radius_px`
  - 全体再推定時の探索に使う近傍スケール
- `max_jump_px`
  - ロボット重心として許容する最大移動量
- `smoothing_alpha`
  - 重心平滑化係数
- `num_thrusters`
  - 追跡対象のスラスタ数
- `min_thruster_distance_px`, `max_thruster_distance_px`
  - スラスタ点同士の距離制約
- `thruster_search_radius_px`, `thruster_reacquire_radius_px`
  - 個別 ROI 追跡の通常探索半径と再取得半径
- `thruster_max_step_px`
  - ROI 予測に使うスラスタ移動量の上限
- `orange_clahe_clip_limit`
  - 前処理の CLAHE 強度
- `orange_red_minus_green_min`, `orange_green_minus_blue_min`
  - RGB 成分差によるオレンジ判定条件
- `orange_min_red`, `orange_min_green`
  - RGB 成分の最小値条件
- `pool_corners_px`
  - メートル座標変換に使うプール四隅の画素座標
- `water_area_corners_px`
  - スラスタ検出範囲として使う水面領域の四隅画素座標
  - 未指定の場合は `pool_corners_px` を検出範囲として使う
- `pool_width_m`, `pool_height_m`
  - プール実寸
- `init_point_px`
  - 初期フレームでロボット近傍を示す基準点
- `reference_frame`
  - 参照フレーム番号

## CLI オプション

主要オプション:

- `video`
- `--config`
- `--make-config`
- `--reference-frame`
- `--export-reference-frame`
- `--export-orange-preview`
- `--pool-corners-px`
- `--water-area-corners-px`
- `--init-point-px`
- `--pool-width-m`, `--pool-height-m`
- `--csv`
- `--annotated`
- `--hsv-lower`, `--hsv-upper`

## 実装上の制約

- 色ベース追跡なので、水面反射、照明変化、類似色のノイズに影響を受けます。
- 初期フレームのスラスタ自動初期化が不安定だと、後続追跡にも影響します。
- `hold` が多い場合は、ROI 内でオレンジ抽出が弱いことを意味します。
- `global` が出るフレームでは、局所追跡に失敗して全体再推定へ落ちています。
- 現状は単一ロボット前提です。
- 現状はテストコードがありません。

## ファイル構成

- [`main.py`](./main.py)
  - 薄い CLI エントリポイント
- [`src/config.py`](./src/config.py)
  - 設定 dataclass と設定ファイル入出力
- [`src/geometry.py`](./src/geometry.py)
  - 距離計算、プールマスク、射影変換
- [`src/detection.py`](./src/detection.py)
  - オレンジ抽出と輪郭候補生成
- [`src/tracking.py`](./src/tracking.py)
  - 初期スラスタ選定と個別 ROI 追跡
- [`src/output.py`](./src/output.py)
  - CSV 出力、速度計算、注釈描画
- [`src/pipeline.py`](./src/pipeline.py)
  - 参照画像出力、プレビュー、設定生成、動画処理本体
- [`src/cli.py`](./src/cli.py)
  - CLI 引数定義と実行フロー
- [`src/tracker_types.py`](./src/tracker_types.py)
  - 共有する型定義
- [`tools/analyze_all_movies.sh`](./tools/analyze_all_movies.sh)
  - `movie/` 内の動画をまとめて解析するシェルスクリプト
- [`tracker_config_example.json`](./tracker_config_example.json)
  - 設定例
