# Production V5 変更内容

Version: `5.0.0-production-v5`

## SQLiteバッチ入力

`lanegen_run.py`へ次の実行形式を追加しました。

```text
--input <SQLite>
--output <BEV results root>
```

内部では新規`lanegen_sqlite_batch.py`へ委譲します。
個別PCDを直接指定する従来形式との互換性は維持しています。

## DBパス列

優先順位:

1. `lidar_path`
2. `ladar_path`
3. `lidar_source`

空文字とNULLは使用しません。
添付SQLiteでは`lidar_path`が全行NULLだったため、
`lidar_source`へ自動フォールバックすることを確認しました。

## resultルート

選択したLiDARパスから祖先`result`を抽出します。
`result_bagEx`などは兄弟`result`へ正規化します。

入力:

```text
result/mapqr_input/IBEV.pcd
result/mapqr_input/RGBBEV.pcd
result/mapping/mapping_pose.txt
```

## 出力先

`--output`以下を再帰検索し、次を探します。

```text
<resultの親フォルダ名>_output
```

出力ディレクトリ:

```text
<clip>_output/work_dir/bevld
```

- `work_dir/bevld`は自動作成
- 同名`<clip>_output`が複数ある場合はエラー
- 見つからない場合は既定でエラー
- `--create-output-container`で`--output`直下への作成も可能

## 重複排除

複数DB行が同じresultへ解決される場合は1回だけ処理します。
特に`result`と`result_bagEx`の重複行を統合します。
元の全DB行IDはsource metadataへ保存します。

## 追跡情報

各clip:

```text
IBEV_lanegen_source.json
```

バッチ全体:

```text
bevld_lanegen_batch_latest.json
```

へ、DB行、入力、出力、状態、終了コード、処理時間を保存します。

## 運用オプション

- `--task-id`: 指定DB IDのみ処理
- `--max-tasks`: 最大処理clip数
- `--dry-run`: 実行せずパス対応を確認
- `--fail-fast`: 最初の失敗で停止
- `--no-resume`: Stageキャッシュを使用しない
- `--create-output-container`: `<clip>_output`がない場合に作成

未知のLaneGen閾値オプションは各clipの子プロセスへ転送します。

## コンソールメッセージ修正

- 軌跡不足を処理失敗ではなく `SKIPPED` として扱う
- `停止Pose除去後の軌跡点が不足しています` ではTracebackを通常表示しない
- SQLiteバッチでは `[skip] <clip>: 理由。次のタスクへ進みます。` と表示
- 予期しない失敗は `[failed]` と表示し、`[error]` を使用しない
- Tracebackは `--debug-traceback` 指定時だけ表示
- タスク終了時に `[ok]` / `[skip]` / `[failed]` / `[cancelled]` を表示
- バッチ終了時に件数サマリーを表示
- スキップのみのバッチは `COMPLETED_WITH_SKIPS`、終了コード0
