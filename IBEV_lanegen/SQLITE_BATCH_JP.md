# SQLite入力によるBEV Lane一括生成

Production V6では、個別PCDパスをコマンドへ直接書く方法に加えて、
SQLiteを`--input`で指定するバッチ実行に対応しています。

## 入力パスの解決規則

SQLiteの`match_result_tasks`を読み、次の順でLiDARパスを選択します。

1. `lidar_path`
2. `ladar_path`（綴り違いのDBにも対応）
3. `lidar_source`

選択したパスの祖先にある`result`を入力ルートとします。
`result_bagEx`などの`result*`は、同じclipの兄弟フォルダ`result`へ正規化します。

`result`以下は固定で次を読みます。

```text
<clip>/result/mapqr_input/IBEV.pcd
<clip>/result/mapqr_input/RGBBEV.pcd
<clip>/result/mapping/mapping_pose.txt
```

## 出力先の解決規則

`--output`以下を再帰検索し、入力`result`の親フォルダ名に
`_output`を付けたフォルダを探します。

例:

```text
入力result:
D:\...\clip_2026_05_29_13_55_57_0013\result

検索するフォルダ名:
clip_2026_05_29_13_55_57_0013_output
```

見つかったフォルダの下へ、次を作成して保存します。

```text
<clip>_output\work_dir\bevld\
```

同名フォルダが複数見つかった場合は、誤ったclipへ保存しないよう
そのタスクをエラーにします。

見つからない場合も既定ではエラーです。`--create-output-container`を指定すると、
`--output`直下へ`<clip>_output`を作成できます。

## 添付SQLiteを使う例

全レコードを処理します。

```bat
python IBEV_lanegen_production_v6\lanegen_run.py ^
  --input "D:\path\BEV_OD_processing.sqlite" ^
  --output "D:\sakaki\BEV_results20260716"
```

指定したDB IDだけを処理します。今回例示されたclipはID 13です。

```bat
python IBEV_lanegen_production_v6\lanegen_run.py ^
  --input "D:\path\BEV_OD_processing.sqlite" ^
  --output "D:\sakaki\BEV_results20260716" ^
  --task-id 13
```

実行せず、入力・出力の対応だけ確認します。

```bat
python IBEV_lanegen_production_v6\lanegen_run.py ^
  --input "D:\path\BEV_OD_processing.sqlite" ^
  --output "D:\sakaki\BEV_results20260716" ^
  --task-id 13 ^
  --dry-run
```

複数IDは`--task-id`を繰り返します。

```bat
--task-id 13 --task-id 14 --task-id 15
```

LaneGen本体の閾値は、そのまま末尾へ追加できます。

```bat
python IBEV_lanegen_production_v6\lanegen_run.py ^
  --input "D:\path\BEV_OD_processing.sqlite" ^
  --output "D:\sakaki\BEV_results20260716" ^
  --sigma-threshold 2.5 ^
  --pcd-chunk-points 500000
```

## 出力されるファイル

各clipの`work_dir\bevld`へ次を保存します。

```text
IBEV.xml
IBEV_review.xml
IBEV_annotator.json
IBEV_annotator_main.json
IBEV_annotator_review.json
IBEV_tracks.csv
IBEV_preview.png
IBEV_stage1.npz
IBEV_stage2.npz
IBEV_stage1.meta.json
IBEV_stage2.meta.json
IBEV_stage3.meta.json
IBEV_job_state.json
IBEV_progress.json
IBEV_lanegen_source.json
```

`IBEV_lanegen_source.json`には、SQLiteパス、使用したDB行ID、
入力result、実際に解決した3入力ファイル、出力先を記録します。

バッチ全体の結果は次へ保存します。

```text
<--output>\bevld_lanegen_batch_latest.json
```

## 重複レコード

同じclipについて`result`と`result_bagEx`の行が両方ある場合、
同じ`result`ルートへ正規化して1回だけ処理します。
バッチ結果と`IBEV_lanegen_source.json`には、統合した全DB行IDを残します。

## 再開

既定で`--resume`相当の動作を行います。同じ入力・同じ設定なら、
完了済みStageを再利用します。

最初から再生成する場合は次を指定します。

```text
--no-resume
```
