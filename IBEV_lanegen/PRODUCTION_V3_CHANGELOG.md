# Production V3 変更点

Version: `3.0.0-production-v3`

Production V2の検出・誤接続対策を維持したまま、
量産運用に必要なPCD入力、途中再開、キャンセル、二重実行防止、
原子的保存、GUI連携ブリッジを追加しました。

## PCD入力
- `DATA ascii`
- `DATA binary`
- `DATA binary_compressed`（LZF）
- ASCIIは行チャンク、binaryは`numpy.memmap`
- packed `rgb` / `rgba`のFLOAT32・UINT32を認識
- `intensity`, `reflectivity`, `i`, `intensities`を認識
- `--pcd-chunk-points`

V3は巨大ASCIIの全文文字列化を廃止しました。
ただしStage1は証拠生成前に有効点配列を連結します。
セル集約までの完全ストリーミング化は次段階です。
`binary_compressed`はPCD仕様上、展開バッファが必要です。

## 中断・再開
`lanegen_run.py`へ以下を追加しました。

- `--resume`
- `--state-file`
- `--progress-json`
- `--cancel-file`
- `--reset-cancel-file`
- `--lock-file`

各Stageの入力SHA-256と成果物を検証し、一致する完了Stageだけ再利用します。

## 安全なキャンセル
- Stage1 PCD読込中は点チャンク境界でキャンセル
- Stage1後、Stage2後にも確認
- キャンセル時終了コード `2`
- 状態JSONへ `CANCELLED`

## 二重実行防止
同じ出力先・stemへの同時実行をlockファイルで拒否します。

## 原子的保存
一時ファイルへ書き、完了後に置換します。

対象:
- Stage1 / Stage2 NPZ
- JSON / meta / job state / progress
- CVAT XML
- tracks CSV

## PyQt5連携
新規 `lanegen_qt_runner.py`:

- `QProcess`による別プロセス実行
- progress JSON監視
- ログ・完了・失敗signal
- 協調キャンセル
- 再開

UI文字列はAnnotator側の既存` t()`で日本語・英語化します。

## 量産用補助
- `lanegen_self_check.py`
- `lanegen_job_control.py`

## 回帰試験
- ASCII / binary / binary_compressed PCD一致
- 小チャンク境界
- `--resume`
- 協調キャンセル
- Frenet交差曖昧性
- 長距離橋渡し交差拒否
- XML / Annotator JSON決定性

## 次段階
- Stage1セル集約の完全ストリーミング化
- binary_compressed展開先の一時memmap化
- Annotator MainWindowへの実UI統合
- 実IBEVでの速度・メモリ・閾値評価
