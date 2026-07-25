# BEV Lane Annotatorへの組込みガイド

```python
from pathlib import Path
from lanegen_qt_runner import LaneGenQtRunner

self._lanegen_runner = LaneGenQtRunner(self)
self._lanegen_runner.log.connect(self._on_lanegen_log)
self._lanegen_runner.progress.connect(self._on_lanegen_progress)
self._lanegen_runner.finished.connect(self._on_lanegen_finished)

self._lanegen_runner.start_job(
    package_dir=Path(__file__).resolve().parent / "IBEV_lanegen_production_v3",
    intensity_pcd=root_dir / "IBEV.pcd",
    mapping_pose=root_dir / "mapping_pose.txt",
    z_pcd=root_dir / "RGBBEV.pcd",
    output_dir=root_dir / "lanegen_output",
    stem="IBEV",
    resume=True,
)
```

進捗:

```python
def _on_lanegen_progress(self, payload: dict) -> None:
    fraction = float(payload.get("fraction", 0.0))
    self._toolbar_progress.setValue(round(fraction * 100))
```

キャンセル:

```python
self._lanegen_runner.request_cancel()
```

完了後:

```python
def _on_lanegen_finished(self, exit_code: int, state: dict) -> None:
    if exit_code == 0 and state.get("status") == "COMPLETED":
        annotator_json = Path(state["outputs"]["annotator"])
        # LaneLinePanelの既存JSONロード処理へ渡す
```

表示文字列は既存の `t()` で日本語・英語対応してください。
