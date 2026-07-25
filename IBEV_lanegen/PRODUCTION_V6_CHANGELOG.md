# Production V6 変更内容

Version: `6.0.0-production-v6`

## ポリライン平滑化

- 密な検出点列をStage3で自動整形
- 直線性が高いラインはロバスト直交回帰直線
- 曲線は外れ値に強い反復再重み付け3次スプライン
- 曲率に応じた適応サンプリング
- 直線部の不要な頂点を削減
- Laneとcurbで異なる既定許容値
- Zは元の密なラインから割り当て、XY平滑化でZを作り変えない

## 頂点削減

合成回帰データでは、Stage3対象11ラインの総頂点数が次のようになりました。

```text
2852 -> 197  (-93.1%)
```

直線単体試験:

```text
401 -> 22  (-94.5%)
```

曲線単体試験:

```text
401 -> 33  (-91.8%)
```

## 出力情報

- CSVへ近似方式・削減前後頂点数・近似誤差を追加
- Annotator JSONの`generation.geometry_fit`へ同情報を追加
- CVAT XML内部属性へ近似方式と頂点数を追加
- Stage3 metaへ全体削減率を追加

## 再開処理

平滑化パラメータをStage3 parameter fingerprintへ追加しました。
設定を変更して`--resume`した場合、Stage1/2は再利用し、Stage3だけ再生成します。

## 維持した機能

- SQLiteバッチ入力
- `<clip>_output/work_dir/bevld`への出力
- 正常スキップ時に`[error]`やTracebackを出さないコンソール表示
- streaming Stage1
- Frenet交差曖昧性対策
- 長距離誤接続対策
- 原子的保存・再開・キャンセル

## V5キャッシュ互換性

Stage1/Stage2のアルゴリズムは変更していないため、Production V5で生成済みの
`IBEV_stage1.npz`と`IBEV_stage2.npz`を`--resume`で再利用できます。
V6ではStage3だけを再生成し、平滑化済みXML/JSONへ更新します。
