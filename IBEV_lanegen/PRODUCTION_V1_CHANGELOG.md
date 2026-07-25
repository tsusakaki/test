# Production V1 変更点

Version: `1.0.0-production-v1`

## 信頼性・再現性

- 入力PCD、mapping_pose、Z/RGB PCDのSHA-256をmetaへ記録
- コードSHA-256とProduction Versionを各Stage metaへ記録
- `--skip-stage1`使用時に入力署名を照合
- 不一致キャッシュを原則拒否し、明示的な`--force-cache`のみ許可
- 同一入力・同一設定でXMLとAnnotator JSONが一致する回帰試験を追加

## Stage1

- 路面基準Zをセル最小値から下位k点平均へ変更
- `--surface-low-k`を追加
- 孤立した低いZノイズで路面点が除外される問題を軽減

## Stage2

- センサー有効区間と無効区間を保持
- 破線のOFF区間とセンサー欠測を分離
- 各Trackへ以下を記録
  - observed_ratio
  - observed_on_valid_ratio
  - sensor_valid_ratio
  - interpolated_ratio
  - max_gap_m
  - max_valid_gap_m
  - max_unknown_gap_m

## Stage3

- Lane/Curb confidenceから線の長さを除外
- 長いLaneを優遇せず、短い実在Laneを不当に低評価しない
- 出力点Zを走行軌跡中心ではなく、簡略化前のLane自身のXY/Zから対応付け
- 長距離補間、構造推定、センサー欠測等をreview理由として明示
- 数値review_priorityを追加
- CVAT XMLに加えてBEV Lane Annotator用Global XYZ JSONを直接出力
- 自動生成根拠、観測率、補間率、最大ギャップ、生成条件をJSONへ保持

## 回帰試験

`python lanegen_regression_test.py -o regression_work`

- Stage1/2/3完走
- Lane/Curb精度下限
- Annotator JSON schema
- ID一意性
- 再実行時の決定性

詳細は`PRODUCTION_V1_TEST_REPORT.json`を参照。

## Production V1で未対応

- binary / binary_compressed PCD
- 複数候補Frenet投影
- Yaw/Z/連続性を使った交差点曖昧性解消
- Greedy追跡から全体対応付けへの変更
- 実IBEVデータでの最終閾値調整
