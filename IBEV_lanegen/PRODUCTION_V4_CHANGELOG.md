# Production V4 変更内容

Version: `4.0.0-production-v4`

## 1. Stage1を完全ストリーミング集約へ変更

V3:
- PCDはチャンク読込み
- ただし有効点を連結してからFrenet証拠を生成
- 点数に比例してメモリが増加

V4:
- 現在のPCDチャンクだけを保持
- 各チャンクをFrenetセルへ即時集約
- 全点XYZ/intensity/RGB配列を保持しない
- 点群部分のメモリ計算量を `O(chunk_points)` に制限
- グリッド部分は `O(ns * nd * k)`

既定値:

```text
--stage1-aggregation-mode streaming
```

比較・互換確認用に、従来方式も残しています。

```text
--stage1-aggregation-mode batch
```

## 2. 厳密なストリーミング統計

新規 `lanegen_streaming.py`:

- `StreamingTopK`
- `StreamingMoments`
- intensity evidence
- Z evidence
- RGB evidence

セルごとの上位k・下位kを保持し、合成fixtureではbatch方式と
対象配列が完全一致することを確認しています。

## 3. 複数パスPCD Reader

`PcdChunkReader`を追加しました。

- 同じPCDを複数回走査可能
- ASCIIは再度チャンク読込み
- binaryはmemmapを再利用
- binary_compressedは一度展開し、一時memmapを再利用
- context終了時に一時ファイルを削除
- キャンセル確認を行単位からチャンク境界へ変更

intensityは2パス、Zは1パス、RGBは2パスで集約します。

## 4. parameter-aware resume

`--resume`でキャッシュを再利用する際に、入力SHA-256だけでなく、
各Stageに影響するパラメータも照合します。

例:

- `--sigma-threshold`変更
  - Stage1は再利用
  - Stage2以降を再生成
- Stage1解像度変更
  - Stage1から再生成

各Stage metadataへparameter fingerprintを保存します。

## 5. stale lock回復

異常終了でlockファイルが残った場合:

- 同一ホストでPIDが存在しない → 即時回復
- 別ホストまたは判定不能 → `--stale-lock-hours`経過後に回復
- 既定値: 24時間

## 6. 進捗ファイルをjob stateから分離

- `<stem>_job_state.json`
- `<stem>_progress.json`

Stage1の内部進捗を一括処理全体の進捗率へマッピングします。

## 7. BEV Lane Annotator UI統合

`annotator_integration/`へ以下を追加:

- `main_window.py`
- `lane_line_panel.py`
- `lanegen_qt_runner.py`

上部ツールバー:

- `初期生成 / Generate`
- `キャンセル / Cancel`
- 既存進捗バー

完了後:

- `IBEV_annotator.json`を自動取込
- 手動・半自動・通常importデータを保持
- 前回LaneGen生成recordだけ置換
- ID衝突時は一意IDを再付与
- `auto_track_id`と`generation` metadataを保持
- 全件検証成功後に一括反映

インストーラ:

```text
install_annotator_integration.py
```

既存ファイルを日時付きでバックアップしてから置換します。

## 8. 回帰試験

- ASCII / binary / binary_compressed PCD
- StreamingTopKとbatch top-kの一致
- compressed memmapの複数パス再利用と削除
- streaming / batch Stage1対象配列の完全一致
- parameter-aware resume
- stale lock回復
- 協調キャンセル
- Frenet交差曖昧性
- 長距離橋渡し交差拒否
- 出力決定性

詳細は `PRODUCTION_V4_TEST_REPORT.json` を参照してください。

## 現時点の制限

- 実IBEV.pcd本体が現在の実行コンテナへマウントされていないため、
  実データでの速度・最大RSS・閾値最終評価は未実施
- streaming方式は複数パスのため、点数が小さいデータではbatchより遅い
- 証拠グリッド自体のサイズは走行距離・横幅・解像度に比例する
