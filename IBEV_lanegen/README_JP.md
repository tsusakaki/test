# IBEV LaneGen Production V7

Production V7では、反射強度Laneを優先保持したまま、
不足時だけRGBBEVの局所コントラストから白線・黄線候補を追加します。

- `RGB_LANE_FUSION_JP.md`
- `PRODUCTION_V7_CHANGELOG.md`
- `PRODUCTION_V7_TEST_REPORT.json`

---

# IBEV LaneGen Production V6

現在のProduction Versionは **6.0.0-production-v6** です。

Production V6では、V5のSQLiteバッチ、ストリーミングPCD処理、
交差点・長距離誤接続対策、コンソール表示修正を維持しながら、
検出したLane Line・curb boundaryを**直線またはロバスト曲線で平滑化し、
頂点数を大幅に削減**する処理を追加しました。

## SQLiteバッチ実行

```bat
python lanegen_run.py ^
  --input "D:\path\BEV_OD_processing.sqlite" ^
  --output "D:\sakaki\BEV_results20260716"
```

既定設定では、Stage3で以下を自動実行します。

```text
直線に近いライン       ロバスト直交回帰直線
曲率があるライン       ロバスト3次スプライン
スプラインの出力       曲率に応じた適応頂点配置
最大頂点間隔           5.0 m
Laneの近似許容          0.05 m
curbの近似許容          0.08 m
```

主要な調整引数:

```text
--geometry-smoothing auto
--output-point-spacing 5.0
--straight-line-tolerance 0.05
--spline-smoothing-tolerance 0.04
--curve-approximation-tolerance 0.05
--curb-straight-line-tolerance 0.08
--curb-spline-smoothing-tolerance 0.06
--curb-curve-approximation-tolerance 0.08
```

詳細:

- `POLYLINE_SMOOTHING_JP.md`
- `SQLITE_BATCH_JP.md`
- `PRODUCTION_V6_CHANGELOG.md`
- `PRODUCTION_V6_TEST_REPORT.json`

---

# IBEV 1パス自動ライン生成パイプライン — Production V1

この版は、元の3段パイプラインを量産運用へ近づける最初の堅牢化版です。
検出本数を機械的に増減するのではなく、**再現性、レビュー根拠、Annotator連携、
センサー欠測と破線空白の区別**を追加しています。

## Production V1 の主な変更

1. **路面Z基準のロバスト化**
   セル内の単一最小Zではなく、下位k点平均を使います。
   低い孤立ノイズで正常な路面点が除外される問題を抑えます。
2. **破線空白とセンサー欠測を分離**
   `bg_valid` / `z_occupied` をトラックに保持し、観測不能区間を
   破線のOFF区間として扱いません。
3. **レビュー根拠を明示**
   `observed_ratio`, `interpolated_ratio`, `max_gap_m`,
   `max_unknown_gap_m`, `review_reasons`, `review_priority` を出力します。
4. **短いLaneを信頼度で減点しない**
   長さをconfidenceから外し、自動生成根拠の強さを評価します。
5. **構造推定を通常検出と区別**
   `structure_infill` / `structure_virtual` はreview対象になります。
6. **Global Z対応の修正**
   簡略化後のZは走行軌跡中心ではなく、簡略化前のLane自身の密なXYから戻します。
7. **BEV Lane Annotator用JSONを直接出力**
   `*_annotator.json`, `*_annotator_main.json`, `*_annotator_review.json`
   を生成します。`map_world` のGlobal XYZです。
8. **再現性記録**
   入力PCD、mapping_pose、Stage NPZ、各PythonコードのSHA-256をmetaへ保存します。
9. **Stage1キャッシュの誤用防止**
   `--skip-stage1` 時に入力SHA-256を照合し、不一致なら停止します。
10. **自動回帰試験**
    `lanegen_regression_test.py` で精度、JSONスキーマ、ID一意性、再実行の決定性を確認します。

---

1 パス分の点群と mapping_pose だけから、`lane_line` と `curb_boundary` の
CVAT XML を自動生成する。既存の `BevCvatConverter.cvat_xml_to_lanes()` で
そのまま読み戻せる。

---

## 1. 構成

```
lanegen_core.py            共通基盤 (PCD入出力 / 軌跡 / Frenet / CVAT XML)
lanegen_stage1_extract.py  Stage1 抽出   (重い・決定的) -> NPZ にキャッシュ
lanegen_stage2_lines.py    Stage2 対応付け (軽い) -> トラック
lanegen_stage3_export.py   Stage3 出力   (軽い) -> CVAT XML / CSV / preview
lanegen_run.py             3段一括実行
lanegen_evaluate.py        予測 XML と正解 XML の突き合わせ評価
make_synthetic_ibev.py     合成テストデータ生成 (回帰テスト用)
```

**Stage1 と Stage2 の間でキャッシュを切っているのが要点。**
点群の読み込みと Frenet 投影が処理時間の大半を占める一方、
そこは決定的で、閾値調整では一切変わらない。
実際、前回の 2 パッケージ (`z融合版` と `z+rgb融合版`) は
抽出結果が完全に一致していて、差は分類層だけにあった。

閾値を触るときは Stage1 を再実行しないこと。

---

## 2. 使い方

### 初回

```bat
python lanegen_run.py "D:\data\IBEV.pcd" "D:\data\mapping_pose.txt" ^
  -o "D:\data\lanegen_out" ^
  --z-pcd "D:\data\RGBBEV.pcd" ^
  --stem IBEV
```

量産向けの主要追加引数:

```text
--surface-low-k 3                 路面基準に使う下位Z点数
--review-gap-threshold 30         長い補間をreviewへ送る閾値[m]
--review-interpolated-ratio 0.70  補間率review閾値
--review-unknown-gap-threshold 30 センサー無効連続区間のreview閾値[m]
--skip-stage1                     Stage1キャッシュ再利用（署名照合あり）
--force-cache                     署名不一致キャッシュを意図的に利用
```

出力:

```
IBEV_stage1.npz        Frenet 証拠 (キャッシュ)
IBEV_stage1.meta.json  抽出統計。まずこれを見る
IBEV_stage2.npz        トラック
IBEV_stage2.meta.json
IBEV.xml               本体。信頼度が閾値以上のもの
IBEV_review.xml        要確認。読み込むかは作業者が選ぶ
IBEV_tracks.csv        全トラックの特徴量と信頼度内訳
IBEV_preview.png       Frenet 空間の確認画像
IBEV_stage3.meta.json
IBEV_annotator.json          全候補（Annotator直接読込用）
IBEV_annotator_main.json     通常候補のみ
IBEV_annotator_review.json   review候補のみ
```

### 閾値調整 (数秒で回る)

```bat
python lanegen_stage2_lines.py IBEV_stage1.npz -o IBEV_stage2.npz ^
  --sigma-threshold 2.2 --curb-step-max 0.28
python lanegen_stage3_export.py IBEV_stage1.npz IBEV_stage2.npz -o . --stem IBEV
```

### アノテーター表示画像に座標を合わせる

```bat
python lanegen_stage3_export.py IBEV_stage1.npz IBEV_stage2.npz -o . --stem IBEV ^
  --image-meta-json "D:\data\RGBBEV_composite_meta.json" ^
  --xml-image-name "RGBBEV_composite.png"
```

### 合成データでの動作確認

```bat
python make_synthetic_ibev.py -o synth
python lanegen_run.py synth\IBEV_synth.pcd synth\mapping_pose_synth.txt ^
  -o out --z-pcd synth\RGBBEV_synth.pcd --stem IBEV
python lanegen_evaluate.py out\IBEV.xml synth\truth_gt.xml
```

同梱の合成データでの実測値:

| 指標 | lane_line | curb_boundary |
|---|---|---|
| recall (横許容 0.20m) | 0.941 | 0.873 |
| precision | 1.000 | 0.878 |
| 横方向 RMSE | 0.026 m | 0.088 m |
| **正解1本あたりの予測本数** | **1.00** | **1.00** |

導流帯 3 本は本体 XML から除外、破線の 25m 欠測区間は 1 本に連結。

---

## 3. 方式

### Stage1 — 土台

反射強度が真の intensity フィールドではなく 8bit packed RGB でしか
手に入らない前提で、実効分解能を稼ぐ処理を入れてある。

1. **セル集約を max ではなく上位k平均 (float32)**
   1 セルあたり約 10 点あるので、量子化ノイズが √k 分の 1 になる。
   8bit → 約 9.5bit 相当。max は 1 点しか使わないうえ外れ値に最も弱い。
2. **中央を除外した annulus 背景を引く**
   自セル ±0.25m を除外し ±0.80m の平均を背景とする。
   細い高反射帯が自分自身の背景を押し上げない。
   幅の広い明部 (白っぽい舗装・コンクリート路肩) はここで消える。
3. **s 列ごとのロバスト散布度 (1.4826·MAD) で正規化**
   しきい値がカウント値ではなく「路面比 n シグマ」になる。
   距離・入射角・走行条件によるベースライン変動をスケール側で吸収する。
4. **Z は s 列ごとに横断勾配を最小二乗で除去してから符号付き段差を出す**
   大きさだけの勾配では、縁石・壁・法面・側溝を区別できない。

### Stage2 — 対応付け

大きな Closing と細線化トレースは使わない。あれは交差点・分岐で必ず壊れる。

```
反射リッジ点 (各 s 列で 1 本に縮約、幅上限つき)
  -> 塗装片 (開始s / 終了s / 中央d / 傾き / 強度)
  -> 塗装片グラフ (最長経路 DP)
  -> トラック
```

* 短ギャップ (破線の空白) と長ギャップ (オクルージョン) を同じ枠組みで扱う。
  実データで 25〜70m の分断が観測されているので、破線周期だけを
  想定した接続では届かない。既定の上限は 90m。
* 横位置許容差は空白長に応じて緩める (`0.30 + 0.004 × gap`)。
  90m先でも0.66mに抑えるため通常の平行車線への飛び移りは起こりにくい。
  ただし交差点・分合流・往復軌跡近接では横位置だけでは保証できないため、
  長距離補間はStage3でreview理由と最大ギャップを記録する。

**レーン構造の事前知識**は、センサ条件に一切依存しない推論なので
悪条件でも効く。

* 間隔 2.5m 未満で 3 本以上並び、かつ互いの s 区間がほぼ一致する群
  → 導流帯・横断歩道として除外。
  「s 区間がほぼ一致」という条件が重要で、これが無いと
  導流帯の隣を走る本物の車線境界線まで巻き込む。
* 間隔 5.5〜7.6m の空き → その帯だけ閾値を下げて再探索。
  全体を下げないので誤検出は増えない。
  ノイズの拾い上げは、見つかった横位置のばらつき (既定 0.12m) で棄却する。

**縁石**は符号付き左右レベル差で追跡する。

```
|外側レベル − 内側レベル| が 0.08〜0.30m で、
勾配ピークがあり、符号が s 方向に一貫している
```

段差 0.30m 超は擁壁・法面として落とす。
反射リッジと段差リッジは**独立に追跡**し、
どちらかを他方の「支持情報」として吸収することはしない。
近接している場合だけ重複を落とす (既定 0.18m)。

### Stage3 — 出力

* **RGB は主判定に使わない。** 線種との整合チェック (信頼度) と
  白/黄の色属性だけに使う。
  夜間は色が壊れ、雨天は反射強度が壊れるので、片方に依存しない構成にしてある。
* **RGB の自己診断**を入れてある。白画素比率が 0.4%〜25% の範囲を
  外れたら「信用できない条件」と判断して重みを 0 にし、
  理由を `stage3.meta.json` の `rgb_health` に残す。
  静かに劣化するのではなく、降りたことが記録に残る。
* **低信頼のラインは捨てない。** 本体 XML と `_review.xml` に分ける。
  ラベルスキーマを変更せずに済み、作業者が読み込むかを選べる。
  プリラベルでは「描き直す」より「消す」方が速いので recall を優先する。
* `lane_number` は d の並び順から自動採番する (既定は自車基準 L1/R1…)。

信頼度の内訳は CSV の `conf_*` 列に全部出る。

---

## 4. 主要パラメータ

| 引数 | 既定 | 意味 |
|---|---|---|
| `--sigma-threshold` | 2.5 | 路面比 何シグマ以上を塗装候補とするか |
| `--topk` | 3 | セル集約に使う上位点数 |
| `--bg-inner / --bg-outer` | 0.25 / 0.80 | annulus 背景の内外半径 [m] |
| `--max-gap` | 90.0 | 塗装片を接続する最大空白長 [m] |
| `--gap-penalty` | 0.08 | 空白 1m あたりのスコア減点 |
| `--hatch-max-spacing` | 2.5 | この間隔未満の並走群は車線境界線ではない [m] |
| `--infill-min/max-spacing` | 5.5 / 7.6 | 穴埋め再探索を起動する間隔 [m] |
| `--curb-step-min/max` | 0.08 / 0.30 | 縁石とみなす段差 [m] |
| `--confidence-threshold` | 0.55 | 本体 XML に入れる下限 |
| `--curb-edge` | gutter | 縁石をどのエッジに置くか |

### 調整の順番

1. `stage1.meta.json` の `intensity.sigma_p99` を見る。
   4 未満なら塗装と路面が分離できていないので、`--sigma-threshold` を
   下げる前に `--surface-band` と `--topk` を疑う。
2. 線が足りない → `--sigma-threshold` を下げる (2.5 → 2.0)。
3. 線が分断される → `--max-gap` を上げるか `--gap-penalty` を下げる。
4. 縁石が出すぎる → `--curb-step-min` を上げる、`--curb-min-coverage` を上げる。
5. 縁石が壁を拾う → `--curb-step-max` を下げる (0.30 → 0.25)。

---

## 5. 先に決めておくべきこと (規約側)

コードより先に決まっていないと手戻りする。

1. **縁石ポリラインはガター側エッジか天端側エッジか。**
   `--curb-edge {ridge,gutter,top}` で切り替えられるようにしてあるが、
   15〜30cm の系統誤差になるので規約で固定すべき。既定は `gutter`。
2. **導流帯・横断歩道の扱い。** 別クラスか除外か。
   現状は本体 XML から除外し、`--emit-hatch` で review 側へ出せる。
3. **`lane_number` の採番規約。** 現状は自車基準 (L1/R1…)。
   `--lane-number-style {ego,left,none}` で変更可。
4. **`review_required` をラベルスキーマに追加するか、XML 2 本で運用するか。**
   現状は XML 2 本 + `_review_required` 属性の併用。

---

## 回帰試験

```bat
python lanegen_regression_test.py -o regression_work
```

確認内容:

- 合成PCDからStage1/2/3が完走する
- Lane recall/precisionの最低基準
- Curb recall/precisionの最低基準
- Annotator JSONのgeometry、属性値、ID一意性
- `--skip-stage1` 再実行でXMLとAnnotator JSONのSHA-256が一致する

同梱版での結果は `PRODUCTION_V1_TEST_REPORT.json` を参照してください。

---

## 6. 制限

* Production V1ではまだASCII PCDのみ対応。binary / binary_compressedは次段階で対応予定。
* 合成データでの検証しか行っていない。実データでの閾値は
  `stage1.meta.json` の統計を見てから決めること。
* **真の intensity フィールドが取れるなら、そちらを使うべき。**
  Stage1 は `FIELDS` に `intensity` / `reflectivity` があれば自動的に
  そちらを優先する。packed RGB 経由では、路面と塗装の差が
  10 カウント台しかない状態から始めることになる。
* 単一パス前提。同一路線の複数パスを持っているなら、
  昼・乾燥パスで作ったジオメトリを夜間・雨天パスへ
  mapping_pose 経由で転写する方が確実。
