# ポリライン平滑化・頂点削減

## 処理方針

Stage2までの密な検出点は保持し、Stage3の出力形状だけを整形します。
検出・接続・confidence・review判定には影響しません。

1. 連続する重複点を除去
2. ロバスト直交回帰で直線性を評価
3. 直線に近い場合は直線へ近似
4. それ以外は累積距離を媒介変数にしたロバスト3次スプラインへ近似
5. 曲率と近似誤差に応じて必要な位置だけ頂点を配置
6. Zは平滑化で作り変えず、元の密なライン上の最寄りZを割り当て

短いラインを除外したり、Z急変を理由にrejectしたりする処理ではありません。

## 既定値

| 対象 | 直線判定 | スプライン平滑化 | ポリライン近似 |
|---|---:|---:|---:|
| Lane Line | 0.05m | 0.04m | 0.05m |
| curb boundary | 0.08m | 0.06m | 0.08m |

最大頂点間隔は5mです。約100mの直線なら、おおむね20～22頂点になります。
道路の曲率が大きい部分には自動的に頂点が追加されます。

## さらに頂点を減らす

```bat
--output-point-spacing 10.0
```

最大頂点間隔を10mにします。ただし急カーブでは近似誤差条件が優先されるため、
必ず10m間隔になるわけではありません。

## 形状をより厳密に保つ

```bat
--output-point-spacing 3.0 ^
--straight-line-tolerance 0.03 ^
--spline-smoothing-tolerance 0.03 ^
--curve-approximation-tolerance 0.03
```

## 平滑化を無効化

```bat
--geometry-smoothing none
```

この場合もDouglas-Peucker簡略化と最大間隔調整は行います。

## 出力される記録

`IBEV_tracks.csv`、Annotator JSONの`generation.geometry_fit`、
CVAT XMLの内部属性へ次を記録します。

- `mode`: line / spline / rdp_fallback / none
- `vertices_before`
- `vertices_after`
- `reduction_ratio`
- `fit_rmse_m`
- `fit_p95_m`
- `fit_max_error_m`

コンソールには次の形式で集計を表示します。

```text
[info] geometry smoothing: line=2, spline=9; vertices=2852->197 (-93.1%)
```
