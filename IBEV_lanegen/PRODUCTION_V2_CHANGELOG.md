# Production V2 変更点

Version: `2.0.0-production-v2`

Production V1の再現性・レビュー情報・Annotator JSON直接出力を維持し、
交差点、Uターン、往復軌跡近接、長距離欠測、平行線対応の安全性を強化しました。

## Frenet投影

- XY最近傍1候補から、既定12候補の分岐比較へ変更
- XY最近傍を通常の局所分岐として維持
- sが離れた別分岐だけ、Z差とmapping_pose yaw整合性を使って比較
- 別分岐が明確に優れる場合だけ切替
- ほぼ同点の別分岐候補がある点は、既定で証拠集約から除外
- 曖昧点分布をStage1 NPZへ保存

追加:
- `traj_yaw`
- `traj_geometric_yaw`
- `traj_heading_error`
- `frenet_ambiguous`
- `frenet_ambiguous_count`

## リッジ対応

Lane:
- 同一s列の複数リッジをHungarian法で一括1対1対応
- activeトラック処理順による対応入替えを低減

Curb:
- 段差ノイズ片が多いため、回帰試験で安定した全ペアcost順Greedyを既定使用
- `--curb-assignment-method`で変更可能

## 長距離接続

15m以上の接続では次を追加確認:

- 前方・後方の両側外挿
- 厳しい横位置許容差
- 厳しい傾き差
- 他の検出片を横切らないこと
- 長距離接続cost増加

縁石は欠測特性が異なるため、既定では長距離条件を緩和し、
`--curb-strict-long-gap`指定時のみLaneと同じ制約を適用します。

## レビュー情報

各Trackへ追加:

- `frenet_ambiguity_ratio`
- `max_frenet_ambiguity_run_m`

閾値超過時:

- `review_reasons += ["frenet_branch_ambiguity"]`
- review priority上昇
- CVAT属性、CSV、Annotator JSONへ出力

## 回帰試験

- 合成PCD Stage1→Stage2→Stage3
- Lane/Curb精度
- Annotator JSON schema
- ID一意性
- Stage1キャッシュ再利用時の決定性
- 交差点Frenet曖昧点の除外
- 他線を横切る長距離接続の拒否

詳細は `PRODUCTION_V2_TEST_REPORT.json` を参照してください。

## Production V3で残る項目

- ASCII / binary / binary_compressed PCD統一読込
- 大容量PCDチャンク集約
- 中断・再開
- Stage1部分キャッシュ
- Annotator GUIからの実行、進捗、キャンセル
- 実IBEVデータでの閾値最終調整
