# Production V7変更内容

Version: `7.0.0-production-v7`

## 目的

反射強度では弱いがRGBBEV上では視認できる白線・黄線を、
反射強度Laneを一切犠牲にせず追加検出します。

## 主な変更

- Stage2へRGB局所コントラストLane補完を追加
- 既定は `--rgb-lane-fusion auto`
- 反射強度Laneの検出量が十分ならRGB処理を省略
- RGBは加点・追加専用で、反射強度Laneの削除条件にはしない
- 重複除去では反射強度Laneを固定し、RGB候補側だけを落とす
- RGB候補に弱い反射強度支持を要求
- 非常に強いRGB局所証拠は単独採用可能
- Z段差近傍のRGB候補を除外し、縁石と白線を分離
- 白・黄の色条件と最大ペイント幅を使用
- Stage2キャッシュ判定へ全RGB補完パラメータを追加
- SQLiteバッチの `--resume` ではV5/V6 Stage1を再利用し、Stage2だけ再生成可能

## 今回の実データStage1キャッシュでの確認

対象:
`clip_2026_05_29_13_56_57_0014`

反射強度のみ:
- paint ridge: 1,412
- Lane track: 10
- 最終Lane: 9

RGB局所コントラスト補完:
- RGB ridge: 9,513
- RGB追加Track: 48
- RGB重複候補除外: 13
- 最終Lane: 53
- solid: 12
- dashed: 37

Stage3:
- Lane 53
- curb 25
- Annotator record 78

最終的な誤検出の有無はAnnotator上で確認が必要ですが、
反射強度で欠落していた候補を回復できることを確認しました。
