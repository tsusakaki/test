# RGB局所コントラストLane補完

Production V7では、RGBを反射強度Laneの拒否条件には使用しません。

処理順:

1. 反射強度だけでLane候補を生成
2. Lane検出量が不足している場合だけRGB局所コントラスト補完を起動
3. RGB候補には次を要求
   - 周囲路面より局所的に明るい
   - 白または黄色として妥当
   - 弱くても近傍に反射強度支持がある、またはRGB証拠が非常に強い
   - Z段差近傍ではない
   - ペイント幅として細い
4. 反射強度Laneを固定して残す
5. RGB候補のうち重複する候補だけを削除

`RGBBEV.pcd`全体が明るく、Stage3のRGB healthが不健全になっても、
反射強度Laneは削除されません。RGB補完はFrenet局所背景との差を使うため、
画像全体のwhite_ratioとは別の判定です。

## 既定動作

```text
--rgb-lane-fusion auto
```

反射強度のLane総延長が軌跡長に対して少ない、または100m当たりの
Lane本数が少ない場合だけ自動的に補完します。

無効化:

```text
--rgb-lane-fusion off
```

常時有効:

```text
--rgb-lane-fusion on
```
