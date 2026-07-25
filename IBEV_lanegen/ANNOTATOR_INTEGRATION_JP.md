# BEV Lane AnnotatorへのProduction V6統合

## 推奨配置

Annotatorのルート直下へ、ZIP内のフォルダをそのまま配置します。

```text
BevLaneAnnotatorCvat004/
├─ ui/
│  └─ main_window.py
├─ lane_line_panel.py
└─ IBEV_lanegen_production_v6/
   ├─ lanegen_run.py
   ├─ lanegen_qt_runner.py
   └─ annotator_integration/
```

## 自動インストール

```bash
python IBEV_lanegen_production_v6/install_annotator_integration.py \
  --annotator-root "C:\path\to\BevLaneAnnotatorCvat004"
```

既存の次のファイルは、日時付きバックアップを作ってから置換します。

- `ui/main_window.py`
- `lane_line_panel.py`

確認だけ行う場合:

```bash
python IBEV_lanegen_production_v6/install_annotator_integration.py \
  --annotator-root "C:\path\to\BevLaneAnnotatorCvat004" --dry-run
```

## UI動作

上部ツールバーへ追加されます。

- **初期生成 / Generate**
- **キャンセル / Cancel**
- 既存の進捗バー

初期生成を押すと、開いているルートフォルダから次を検索します。

- `IBEV.pcd`
- `mapping_pose.txt`
- `RGBBEV.pcd` または `RGBBEV(1).pcd`

出力先:

```text
<project root>/lanegen_output/
```

完了後、`IBEV_annotator.json`を自動的に作業アノテーションへ取り込みます。

## 既存作業データの扱い

再生成時には次のルールで取り込みます。

- 手動アノテーション: 保持
- 半自動アノテーション: 保持
- 通常importデータ: 保持
- 前回のLaneGen生成データだけ: 置換
- 表示IDが衝突した場合: 新しい一意IDを付与
- `auto_track_id`と`generation` metadata: 保持

すべての新規recordのgeometry検証に成功してから、一括で反映します。途中で不正データが見つかった場合、現在の作業データは変更しません。

## 中断と再開

キャンセルは強制Killではなく協調キャンセルです。PCDチャンク境界またはStage境界で安全に停止します。

再度 **初期生成** を押すと`--resume`で実行し、入力ファイルとパラメータが一致する完了Stageを再利用します。

## 日本語・英語

ボタン・ダイアログはAnnotatorの既存言語設定`get_language()`へ追従します。LaneGen本体の状態JSONは言語非依存です。
