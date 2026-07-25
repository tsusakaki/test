# ui/main_window.py


from __future__ import annotations

import os
import sys
import tempfile
import traceback
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

#   
try:
    from core.i18n import t, set_language, get_language, on_language_changed
except ImportError:
    # : 
    def t(key: str) -> str:
        return key

    def set_language(lang: str) -> None:
        pass

    def get_language() -> str:
        return "ja"

    def on_language_changed(cb) -> None:
        pass

try:
    from core.data_loader import DataLoader, ImageLayer
except ImportError:
    DataLoader = None  # type: ignore
    ImageLayer = None  # type: ignore

try:
    from core.cvat_converter_bev import BevCvatConverter
except ImportError:
    BevCvatConverter = None  # type: ignore

try:
    from core.cvat_client import CvatClient, CvatConfig
except ImportError:
    CvatClient = None  # type: ignore
    CvatConfig = None  # type: ignore

try:
    from core.quality_checker import QualityChecker, QualityConfig
except ImportError:
    QualityChecker = None  # type: ignore
    QualityConfig = None  # type: ignore

try:
    from core.attribute_assigner import AttributeAssigner
except ImportError:
    AttributeAssigner = None  # type: ignore

try:
    from core.priority_classifier import PriorityClassifier
except ImportError:
    PriorityClassifier = None  # type: ignore

try:
    from core.lane_id_assigner import LaneIdAssigner
except ImportError:
    LaneIdAssigner = None  # type: ignore

#  UI  
try:
    from ui.gl_widget import BevGLWidget
except Exception as _e_gl:
    print(f"[WARNING] BevGLWidget import failed: {_e_gl}")
    BevGLWidget = None  # type: ignore

try:
    from ui.quality_panel import QualityPanel
except Exception as _e_qp:
    print(f"[WARNING] QualityPanel import failed: {_e_qp}")
    QualityPanel = None  # type: ignore

try:
    from ui.image_window import ImageWindow
except Exception as _e_iw:
    print(f"[WARNING] ImageWindow import failed: {_e_iw}")
    ImageWindow = None  # type: ignore

try:
    from ui.cvat_dialog import CvatLoginDialog, CvatDialog
except Exception:
    CvatLoginDialog = None  # type: ignore
    CvatDialog = None  # type: ignore

#  LaneLinePanel ( lane_line_panel.py) 
try:
    import sys as _sys
    from pathlib import Path as _Path
    _BASE = str(_Path(__file__).resolve().parent.parent)
    if _BASE not in _sys.path:
        _sys.path.insert(0, _BASE)
    from lane_line_panel import LaneLinePanel
except Exception as _e_llp:
    print(f"[WARNING] LaneLinePanel import failed: {_e_llp}")
    LaneLinePanel = None  # type: ignore

#   
try:
    from register_cvat_tasks import (
        register_tasks,
        identify_tasks_dual,
        generate_task_names,
        generate_project_names,
    )
except ImportError:
    register_tasks = None  # type: ignore
    identify_tasks_dual = None  # type: ignore
    generate_task_names = None  # type: ignore
    generate_project_names = None  # type: ignore

# カメラ画像ストリップパネル（下段）
try:
    from ui.camera_strip_panel import CameraStripPanel
except Exception as _e_csp:
    print(f"[WARNING] CameraStripPanel import failed: {_e_csp}")
    CameraStripPanel = None  # type: ignore

# カメラフォルダ自動検索
try:
    from core.camera_finder import find_camera_dirs, build_camera_image_list
except Exception as _e_cf:
    print(f"[WARNING] camera_finder import failed: {_e_cf}")
    find_camera_dirs = None  # type: ignore
    build_camera_image_list = None  # type: ignore

# フレーム時刻同期
try:
    from core.frame_sync import FrameSyncTable, find_pcd_timestamp_path
except Exception as _e_fs:
    print(f"[WARNING] frame_sync import failed: {_e_fs}")
    FrameSyncTable = None  # type: ignore
    find_pcd_timestamp_path = None  # type: ignore

# アノテーション投影
try:
    from core.annotation_projector import AnnotationProjector
except Exception as _e_ap:
    print(f"[WARNING] annotation_projector import failed: {_e_ap}")
    AnnotationProjector = None  # type: ignore


#  locale  
_LOCALE_DIR = Path(__file__).resolve().parent.parent / "locale"


def _build_dark_palette() -> QPalette:
    
    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 30, 30))
    p.setColor(QPalette.WindowText, QColor(204, 204, 204))
    p.setColor(QPalette.Base, QColor(25, 25, 25))
    p.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    p.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    p.setColor(QPalette.ToolTipText, QColor(204, 204, 204))
    p.setColor(QPalette.Text, QColor(204, 204, 204))
    p.setColor(QPalette.Button, QColor(45, 45, 45))
    p.setColor(QPalette.ButtonText, QColor(204, 204, 204))
    p.setColor(QPalette.BrightText, QColor(255, 50, 50))
    p.setColor(QPalette.Link, QColor(42, 110, 187))
    p.setColor(QPalette.Highlight, QColor(42, 110, 187))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return p



# 
#  MainWindow
# 
class MainWindow(QMainWindow):
    

    # ------------------------------------------------------------------
    # 7.1  
    # ------------------------------------------------------------------
    def __init__(
        self,
        cvat_config: Optional["CvatConfig"] = None,
        show_login: bool = True,
    ):
        super().__init__()
        self.setWindowTitle(t("app_title"))
        self.resize(1400, 900)

        #   
        self.setPalette(_build_dark_palette())
        self.setStyleSheet(
            "QMainWindow { background: #1e1e1e; }"
            "QMenuBar { background: #2a2a2a; color: #ccc; }"
            "QMenuBar::item:selected { background: #3a3a3a; }"
            "QMenuBar::item:disabled { color: #555; }"
            "QMenu { background: #2a2a2a; color: #ccc; border: 1px solid #555; }"
            "QMenu::item:selected { background: #2a6ebb; }"
            "QStatusBar { background: #1e1e1e; color: #888; }"
        )

        #  CVAT 
        self._cvat_client: Optional["CvatClient"] = None
        self._cvat_task_id: Optional[int] = None
        self._cvat_project_id: Optional[int] = None
        self._cvat_project_name: str = ""
        self._cvat_config = cvat_config

        #  PCD
        # 2D pkl export 
        self._pcd_z_resolver: Optional[object] = None  # PcdZResolver | None

        #  Z 
        self._bev_overlay_path: Optional[str] = None  # z_gradient 

        #   () 
        self._image_path: Optional[str] = None
        self._lane_path: Optional[str] = None        # 
        self._trajectory_path: Optional[str] = None  # 
        self._mapping_poses: list = []                # mapping_pose.txt 
        self._has_unsaved_changes: bool = False
        self._sampling_lane_hint: Optional[str] = None  # ルートフォルダ開時のsampling_lane候補パス
        self._pending_lane_path: Optional[str] = None   # ルートフォルダ開時の遅延ロード用clusterパス
        self._root_folder_path: Optional[str] = None    # 最後に開いたルートフォルダ

        #   
        self._image_window: Optional["ImageWindow"] = None
        self._camera_images: list[dict] = []  # 
        self._camera_strip_panel: Optional["CameraStripPanel"] = None  # 下段カメラパネル

        # ── フレーム同期 ─────────────────────────────────────────
        self._frame_sync_table = None   # FrameSyncTable | None
        self._current_frame_idx: int = 0

        # ── 2D車両マーカードラッグによるフレーム移動 ─────────────
        self._ego_drag_last_frame: Optional[int] = None

        # ── 描画位置へのカメラ画像/フレーム自動追従 ─────────────
        # UIチェックボックスのデフォルトはON。
        self._draw_position_follow_enabled: bool = True

        # ── LaneGen Production V6 ───────────────────────────────
        self._lanegen_runner = None
        self._lanegen_package_dir: Optional[Path] = None
        self._lanegen_output_dir: Optional[Path] = None
        self._lanegen_last_log: str = ""

        # ── アノテーション投影 ───────────────────────────────────
        self._annotation_projector = None   # AnnotationProjector | None
        if AnnotationProjector is not None:
            self._annotation_projector = AnnotationProjector()

        #  LaneLinePanel  MainWindow  
        # pointcloud_viewer  MainWindow 
        self.solo_layers: dict = {}   # {name: (layer, row_widget)}
        self.groups: dict = {}        # {name: (group, row_widget)}

        #  11.2 
        # 
        self._last_line_type: str = "solid"
        self._last_line_color: str = "white"
        self._last_line_count: str = "single"
        self._last_class: str = "lane_line"

        #  GL  + LaneLinePanel 
        self._gl: Optional["BevGLWidget"] = None
        self._panel: Optional["LaneLinePanel"] = None

        self._setup_central_widget()
        self._build_menus()
        self._build_status_bar()

        #   
        on_language_changed(self._retranslate_ui)

        #   CVAT  
        if show_login and CvatLoginDialog is not None:
            self._show_login_dialog()

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------
    def _setup_central_widget(self) -> None:
        
        #  :  | : GL | : 
        splitter = QSplitter(Qt.Horizontal)

        #  :  
        self._layer_panel = None
        try:
            from ui.layer_panel import LayerPanel as _LayerPanel
            # GL  None ｰ
            self._layer_panel_cls = _LayerPanel
        except ImportError:
            self._layer_panel_cls = None

        #  : GL  +  
        gl_container = QWidget()
        gl_layout = QVBoxLayout(gl_container)
        gl_layout.setContentsMargins(0, 0, 0, 0)
        gl_layout.setSpacing(0)

        #  2D/3D  
        view_toolbar = QWidget()
        view_toolbar.setFixedHeight(30)
        view_toolbar.setStyleSheet("background: #252525;")
        vt_layout = QHBoxLayout(view_toolbar)
        vt_layout.setContentsMargins(6, 2, 6, 2)
        vt_layout.setSpacing(4)

        _BTN_VIEW_ON = (
            "QPushButton{background:#1a3a5a;color:#7ac8ff;"
            "border:2px solid #3a7aaa;border-radius:3px;"
            "padding:2px 10px;font-size:11px;font-weight:bold;}")
        _BTN_VIEW_OFF = (
            "QPushButton{background:#2a2a2a;color:#888;"
            "border:1px solid #444;border-radius:3px;"
            "padding:2px 10px;font-size:11px;}"
            "QPushButton:hover{background:#3a3a3a;color:#ccc;}")

        self._btn_view_2d = QPushButton("2D")
        self._btn_view_2d.setFixedHeight(24)
        self._btn_view_2d.setStyleSheet(_BTN_VIEW_ON)
        self._btn_view_2d.clicked.connect(lambda: self._set_view_mode("2d"))
        vt_layout.addWidget(self._btn_view_2d)

        self._btn_view_3d = QPushButton("3D")
        self._btn_view_3d.setFixedHeight(24)
        self._btn_view_3d.setStyleSheet(_BTN_VIEW_OFF)
        self._btn_view_3d.clicked.connect(lambda: self._set_view_mode("3d"))
        vt_layout.addWidget(self._btn_view_3d)

        #   
        _BTN_FIT = (
            "QPushButton{background:#2a2a2a;color:#aaa;"
            "border:1px solid #555;border-radius:3px;"
            "padding:2px 10px;font-size:11px;}"
            "QPushButton:hover{background:#3a4a3a;color:#8fc88f;}")
        self._btn_fit_all = QPushButton(t("btn_fit_all"))
        self._btn_fit_all.setFixedHeight(24)
        self._btn_fit_all.setStyleSheet(_BTN_FIT)
        self._btn_fit_all.setToolTip(t("tooltip_fit_all"))
        self._btn_fit_all.clicked.connect(self._fit_to_all)
        vt_layout.addWidget(self._btn_fit_all)

        #   
        _BTN_HOME = (
            "QPushButton{background:#2a2a2a;color:#aaa;"
            "border:1px solid #555;border-radius:3px;"
            "padding:2px 10px;font-size:11px;}"
            "QPushButton:hover{background:#3a3a4a;color:#99ccff;}")
        self._btn_home_view = QPushButton(t("btn_home_view"))
        self._btn_home_view.setFixedHeight(24)
        self._btn_home_view.setStyleSheet(_BTN_HOME)
        self._btn_home_view.setToolTip(t("tooltip_home_view"))
        self._btn_home_view.clicked.connect(self._reset_to_home_view)
        vt_layout.addWidget(self._btn_home_view)

        #   + 
        _BTN_COLOR = (
            "QPushButton{background:#2a2a2a;color:#f0c060;"
            "border:1px solid #665500;border-radius:3px;"
            "padding:2px 10px;font-size:11px;}"
            "QPushButton:hover{background:#3a3a2a;color:#ffd070;}")
        self._btn_auto_color = QPushButton(t("btn_auto_color"))
        self._btn_auto_color.setFixedHeight(24)
        self._btn_auto_color.setStyleSheet(_BTN_COLOR)
        self._btn_auto_color.setToolTip(t("tooltip_auto_color"))
        self._btn_auto_color.clicked.connect(self._auto_color_adjust)
        vt_layout.addWidget(self._btn_auto_color)

        # 
        _gamma_lbl = QLabel(":")
        _gamma_lbl.setStyleSheet("color:#f0c060; font-size:11px; padding:0 2px;")
        _gamma_lbl.setFixedHeight(24)
        vt_layout.addWidget(_gamma_lbl)
        self._gamma_label = _gamma_lbl  # retranslate 

        # DoubleSpinBox
        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 5.0)
        self._gamma_spin.setSingleStep(0.05)
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setValue(1.05)
        self._gamma_spin.setFixedSize(68, 24)
        self._gamma_spin.setToolTip(t("tooltip_gamma"))
        self._gamma_spin.setStyleSheet(
            "QDoubleSpinBox{"
            "  background:#2a2a2a; color:#f0c060;"
            "  border:1px solid #665500; border-radius:3px;"
            "  font-size:11px; padding:1px 2px;}"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button{"
            "  width:14px; background:#333; border:none;}"
        )
        vt_layout.addWidget(self._gamma_spin)

        # ── LaneGen Production V6 ───────────────────────────────
        _BTN_LANEGEN = (
            "QPushButton{background:#263b2b;color:#9fe6ad;"
            "border:1px solid #4c8056;border-radius:3px;"
            "padding:2px 8px;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#315039;}"
            "QPushButton:disabled{background:#292929;color:#666;border-color:#444;}"
        )
        self._btn_lanegen_generate = QPushButton("初期生成")
        self._btn_lanegen_generate.setFixedHeight(24)
        self._btn_lanegen_generate.setStyleSheet(_BTN_LANEGEN)
        self._btn_lanegen_generate.clicked.connect(self._start_lanegen_v4)
        vt_layout.addWidget(self._btn_lanegen_generate)

        self._btn_lanegen_cancel = QPushButton("キャンセル")
        self._btn_lanegen_cancel.setFixedHeight(24)
        self._btn_lanegen_cancel.setStyleSheet(_BTN_LANEGEN)
        self._btn_lanegen_cancel.setEnabled(False)
        self._btn_lanegen_cancel.clicked.connect(self._cancel_lanegen_v4)
        vt_layout.addWidget(self._btn_lanegen_cancel)

        #  
        self._toolbar_progress = QProgressBar()
        self._toolbar_progress.setRange(0, 100)
        self._toolbar_progress.setValue(0)
        self._toolbar_progress.setFixedSize(100, 16)
        self._toolbar_progress.setVisible(False)  # 
        self._toolbar_progress.setStyleSheet(
            "QProgressBar{"
            "  background:#1a1a1a; border:1px solid #444; border-radius:3px;"
            "  color:#888; font-size:10px; text-align:center;}"
            "QProgressBar::chunk{"
            "  background:#2e6da4; border-radius:2px;}"
        )
        vt_layout.addWidget(self._toolbar_progress)

        vt_layout.addStretch()
        gl_layout.addWidget(view_toolbar)

        # 
        self._BTN_VIEW_ON = _BTN_VIEW_ON
        self._BTN_VIEW_OFF = _BTN_VIEW_OFF

        # GL 
        if BevGLWidget is not None:
            self._gl = BevGLWidget(self)
            self._gl.set_ego_drag_callback(
                self._on_ego_marker_drag
            )
            gl_layout.addWidget(self._gl, stretch=1)
        else:
            placeholder = QWidget()
            placeholder.setStyleSheet("background: #1e1e1e;")
            gl_layout.addWidget(placeholder, stretch=1)

        #  GL  
        if self._layer_panel_cls is not None and self._gl is not None:
            self._layer_panel = self._layer_panel_cls(self._gl)
            # Z
            overlay_row = self._layer_panel._rows.get("bev_overlay")
            if overlay_row is not None:
                overlay_row.set_slider_callback(self._on_overlay_threshold_changed)
        else:
            self._layer_panel = None

        # :  : GL
        if self._layer_panel is not None:
            splitter.addWidget(self._layer_panel)
        splitter.addWidget(gl_container)

        # : LaneLinePanel + QualityPanel 
        right_splitter = QSplitter(Qt.Vertical)

        if LaneLinePanel is not None and self._gl is not None:
            self._panel = LaneLinePanel(self._gl, main_win=self)
            self._gl.set_lane_panel(self._panel)
            right_splitter.addWidget(self._panel)
            self._wrap_auto_generate()
        else:
            placeholder = QWidget()
            placeholder.setStyleSheet("background: #1e1e1e;")
            right_splitter.addWidget(placeholder)

        # アノテーション変更時の再投影コールバックを登録
        if self._gl is not None:
            self._gl.set_annotations_changed_callback(
                self._on_annotations_changed
            )

        self._quality_panel: Optional["QualityPanel"] = None
        if QualityPanel is not None:
            self._quality_panel = QualityPanel(self)
            right_splitter.addWidget(self._quality_panel)
            self._quality_panel.hide()

        splitter.addWidget(right_splitter)

        # :  180px GL 65%, Panel 30%
        if self._layer_panel is not None:
            splitter.setSizes([180, 840, 380])
        else:
            splitter.setSizes([980, 420])

        # ── 縦方向スプリッター: 上段（BEV+パネル） | 下段（カメラストリップ） ──
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(splitter)

        # 下段カメラストリップパネル
        if CameraStripPanel is not None:
            self._camera_strip_panel = CameraStripPanel(self)

            # 起動直後からカメラ画像を十分大きく表示する。
            # 旧設定 [700, 180] は下段が約20%しかなく、画像領域が小さすぎた。
            # 添付イメージに合わせ、上段:下段 ≒ 62:38 をデフォルトとする。
            self._camera_strip_panel.setMinimumHeight(220)
            v_splitter.addWidget(self._camera_strip_panel)

            # リサイズ時もおおむね同じ比率で伸縮させる。
            v_splitter.setStretchFactor(0, 62)
            v_splitter.setStretchFactor(1, 38)
            v_splitter.setChildrenCollapsible(False)

            # 初期表示比率（実ピクセル値はウィンドウサイズに応じて再配分される）
            v_splitter.setSizes([620, 380])
        else:
            self._camera_strip_panel = None

        # ── フレームコントロールバー（時間軸マスタースライダー）──
        frame_bar = self._build_frame_control_bar()
        self._frame_control_bar = frame_bar

        # 全体を縦に並べる
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(v_splitter, stretch=1)
        central_layout.addWidget(frame_bar)
        self.setCentralWidget(central_widget)

    def _build_frame_control_bar(self) -> QWidget:
        """フレームコントロールバーを生成して返す。

        構成:
          [◀ 前] [▶ 次] [====スライダー====] [0000/0000] [追従 ○] [自由 ○]
        """
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(
            "QWidget{background:#1a1a2e;}"
            "QLabel{color:#aaa;font-size:11px;}"
            "QPushButton{background:#2a2a3e;color:#ccc;border:1px solid #444;"
            "  border-radius:3px;padding:2px 8px;font-size:11px;}"
            "QPushButton:hover{background:#3a3a5e;}"
            "QCheckBox{color:#aaa;font-size:10px;}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        # ── ラベル ──
        lbl = QLabel("フレーム:")
        lbl.setFixedWidth(55)
        layout.addWidget(lbl)

        # ── 前後スキップ ──
        _BTN_STYLE = (
            "QPushButton{background:#252535;color:#ccc;border:1px solid #444;"
            "border-radius:3px;padding:2px 5px;font-size:10px;min-width:28px;}"
            "QPushButton:hover{background:#3a3a5e;}"
        )

        def _add_skip_button(text: str, delta: int, tooltip: str):
            btn = QPushButton(text)
            btn.setFixedHeight(24)
            btn.setStyleSheet(_BTN_STYLE)
            btn.setToolTip(tooltip)
            btn.clicked.connect(
                lambda _checked=False, d=delta: self._frame_skip(d)
            )
            layout.addWidget(btn)
            return btn

        self._btn_frame_m100 = _add_skip_button(
            "-100", -100, "100フレーム戻る"
        )
        self._btn_frame_m10 = _add_skip_button(
            "-10", -10, "10フレーム戻る"
        )

        btn_prev = QPushButton("◀")
        btn_prev.setFixedHeight(24)
        btn_prev.setStyleSheet(_BTN_STYLE)
        btn_prev.setToolTip("1フレーム戻る (←)")
        btn_prev.clicked.connect(self._frame_prev)
        layout.addWidget(btn_prev)
        self._btn_frame_prev = btn_prev

        btn_next = QPushButton("▶")
        btn_next.setFixedHeight(24)
        btn_next.setStyleSheet(_BTN_STYLE)
        btn_next.setToolTip("1フレーム進む (→)")
        btn_next.clicked.connect(self._frame_next)
        layout.addWidget(btn_next)
        self._btn_frame_next = btn_next

        self._btn_frame_p10 = _add_skip_button(
            "+10", 10, "10フレーム進む"
        )
        self._btn_frame_p100 = _add_skip_button(
            "+100", 100, "100フレーム進む"
        )

        layout.addSpacing(4)

        # ── 描画位置への自動追従 ──
        # スライダーの左側に配置し、その分スライダー幅を短くする。
        self._chk_draw_position_follow = QCheckBox("描画・選択に追従")
        self._chk_draw_position_follow.setChecked(True)
        self._chk_draw_position_follow.setToolTip(
            "ON: 新規ポリライン・ポリゴン・ポイントの1点目、"
            "およびアノテーション一覧選択時だけ、"
            "黄色い車両矩形と下段カメラ画像を追従させます。"
            "2Dビューをパン/ズームしても自動追従しません。"
        )
        self._chk_draw_position_follow.setStyleSheet(
            "color:#88ddaa;font-size:10px;"
        )
        self._chk_draw_position_follow.stateChanged.connect(
            self._on_draw_position_follow_changed
        )
        layout.addWidget(self._chk_draw_position_follow)

        # ── 現在地 ──
        # 2Dビュー中心に最も近いFrameSync/mapping_poseフレームへ、
        # 黄色い自車矩形と下段カメラ画像だけを移動する。
        # 2Dビューの表示範囲・Zoom・回転は変更しない。
        self._btn_frame_current = QPushButton("現在地")
        self._btn_frame_current.setFixedHeight(24)
        self._btn_frame_current.setStyleSheet(_BTN_STYLE)
        self._btn_frame_current.setToolTip(
            "現在の2Dビュー中心に最も近い走行フレームへ移動"
        )
        self._btn_frame_current.clicked.connect(
            self._on_current_location_clicked
        )
        layout.addWidget(self._btn_frame_current)

        # ── スライダー ──
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.setValue(0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.setStyleSheet(
            "QSlider::groove:horizontal{"
            "  height:6px; background:#333; border-radius:3px;}"
            "QSlider::handle:horizontal{"
            "  width:14px; height:14px; margin:-4px 0;"
            "  background:#4488ff; border-radius:7px;}"
            "QSlider::sub-page:horizontal{"
            "  background:#2255aa; border-radius:3px;}"
            "QSlider:disabled::handle:horizontal{background:#555;}"
        )
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self._frame_slider, stretch=1)

        # ── スピンボックス ──
        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(0)
        self._frame_spin.setMaximum(0)
        self._frame_spin.setFixedWidth(65)
        self._frame_spin.setFixedHeight(24)
        self._frame_spin.setEnabled(False)
        self._frame_spin.setStyleSheet(
            "QSpinBox{background:#1c1c2e;color:#ccc;border:1px solid #444;"
            "border-radius:2px;font-size:11px;padding:0 2px;}"
            "QSpinBox::up-button,QSpinBox::down-button{width:14px;}"
        )
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        layout.addWidget(self._frame_spin)

        # ── フレーム情報ラベル ──
        self._lbl_frame_info = QLabel("— / —")
        self._lbl_frame_info.setFixedWidth(80)
        self._lbl_frame_info.setStyleSheet("color:#888;font-size:10px;")
        layout.addWidget(self._lbl_frame_info)

        layout.addSpacing(8)

        # ── 車両追従モード切替 ──
        lbl_mode = QLabel("表示:")
        lbl_mode.setStyleSheet("color:#88aacc;font-size:10px;")
        layout.addWidget(lbl_mode)

        self._chk_follow = QCheckBox("車両追従")
        self._chk_follow.setChecked(True)
        self._chk_follow.setStyleSheet("color:#88aacc;font-size:10px;")
        self._chk_follow.stateChanged.connect(self._on_follow_mode_changed)
        layout.addWidget(self._chk_follow)

        self._chk_free = QCheckBox("自由表示")
        self._chk_free.setChecked(False)
        self._chk_free.setStyleSheet("color:#aaa;font-size:10px;")
        self._chk_free.stateChanged.connect(self._on_free_mode_changed)
        layout.addWidget(self._chk_free)

        layout.addSpacing(12)

        # ── アノテーション投影 ON/OFF ──
        self._chk_project_anno = QCheckBox("投影")
        self._chk_project_anno.setChecked(True)
        self._chk_project_anno.setToolTip("アノテーションをカメラ画像に投影する")
        self._chk_project_anno.setStyleSheet("color:#88ccaa;font-size:10px;")
        self._chk_project_anno.stateChanged.connect(self._on_project_anno_changed)
        layout.addWidget(self._chk_project_anno)

        return bar

    def _wrap_auto_generate(self) -> None:
        """LaneLinePanelの自動生成ボタンに、遅延ロード＋プログレスバー処理をラップする。"""
        if self._panel is None:
            return

        # クラスターコンボのUI要素を非表示（ルートフォルダ開時は自動設定するため不要）
        if hasattr(self._panel, '_cmb_cluster'):
            self._panel._cmb_cluster.hide()
        from PyQt5.QtWidgets import QPushButton, QLabel as _QLabel
        for lbl in self._panel.findChildren(_QLabel):
            if "cluster" in lbl.text().lower() or "クラスター" in lbl.text():
                lbl.hide()
                break

        # 全自動生成ボタンをラップ（現在無効化中のためスキップ）
        # for btn in self._panel.findChildren(QPushButton):
        #     orig_text = btn.text()
        #     if ("全自動生成" in orig_text or "Full Auto" in orig_text
        #             or "✨ 自動生成" in orig_text or "✨ Auto" in orig_text):
        #         try:
        #             btn.clicked.disconnect()
        #         except Exception:
        #             pass
        #         btn.clicked.connect(self._on_auto_generate_with_load)
        #         break

        # 半自動生成ボタンをラップ（プログレスバー付き処理に差し替え）
        for btn in self._panel.findChildren(QPushButton):
            orig_text = btn.text()
            if "半自動生成" in orig_text or "Semi-Auto" in orig_text or "🔧" in orig_text:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(self._on_semi_auto_generate)
                break

    def _on_auto_generate_with_load(self) -> None:
        """自動生成ボタン押下時の処理。

        1. 遅延ロードパス（_pending_lane_path / _sampling_lane_hint）があれば読み込む
        2. プログレスバーで進捗を表示しながら自動生成を実行する
        """
        if self._panel is None or self._gl is None:
            return

        self._toolbar_progress.setValue(0)
        self._toolbar_progress.setVisible(True)
        QApplication.processEvents()

        try:
            # ── ステップ1: cluster.txt の遅延ロード ──────────────────
            self._toolbar_progress.setValue(10)
            QApplication.processEvents()

            pending = getattr(self, '_pending_lane_path', None)
            if pending and Path(pending).exists():
                self._status.showMessage(f"{t('status_loading_lane_points')} {Path(pending).name}")
                QApplication.processEvents()
                lane_points = DataLoader.parse_lane_points(pending)
                if len(lane_points) > 0:
                    colors = DataLoader.generate_cluster_colors(lane_points[:, 3])
                    self._gl.set_lane_points(lane_points[:, :3], colors)
                self._set_cluster_file_to_panel(pending)
                if self._layer_panel is not None:
                    self._layer_panel.set_loaded("lane_points", True)
                self._pending_lane_path = None  # 読み込み済みフラグ
                self._toolbar_progress.setValue(30)
                QApplication.processEvents()

            # cluster.txt がパネルにセットされているかチェック
            paths = getattr(self._panel, '_cmb_cluster_paths', [])
            has_cluster = paths and paths[0] is not None and paths[0] != ""

            # ── ステップ2: sampling_lane の遅延ロード（cluster非存在時） ──
            self._toolbar_progress.setValue(40)
            QApplication.processEvents()

            if not has_cluster:
                hint = getattr(self, '_sampling_lane_hint', None)
                if hint and Path(hint).exists():
                    self._status.showMessage(f"{t('status_loading_sampling_lane')} {Path(hint).name}")
                    QApplication.processEvents()
                    self._auto_import_sampling_lane(hint)
                    self._toolbar_progress.setValue(60)
                    QApplication.processEvents()
                    # sampling_lane をインポートした場合は自動生成不要（ポリライン追加済み）
                    self._toolbar_progress.setValue(100)
                    QApplication.processEvents()
                    self._status.showMessage(t("status_sampling_lane_done"))
                    return

            # ── ステップ3: cluster.txt がない場合はintensity/PCDから生成 ──
            if not has_cluster and self._image_path:
                self._toolbar_progress.setValue(50)
                QApplication.processEvents()
                self._status.showMessage(t("status_generating_cluster"))
                img_dir = Path(self._image_path).parent
                cluster_path = self._find_existing_cluster_txt(img_dir)
                if cluster_path is None:
                    cluster_path = self._generate_cluster_from_intensity(img_dir)
                if cluster_path is None:
                    intensity_path = self._generate_intensity_from_pcd(img_dir)
                    if intensity_path is not None:
                        cluster_path = self._generate_cluster_from_intensity(img_dir)
                if cluster_path:
                    lane_points = DataLoader.parse_lane_points(cluster_path)
                    if len(lane_points) > 0:
                        colors = DataLoader.generate_cluster_colors(lane_points[:, 3])
                        self._gl.set_lane_points(lane_points[:, :3], colors)
                    self._set_cluster_file_to_panel(cluster_path)
                    if self._layer_panel is not None:
                        self._layer_panel.set_loaded("lane_points", True)
                    has_cluster = True

            self._toolbar_progress.setValue(70)
            QApplication.processEvents()

            # ── ステップ4: 自動生成実行 ──────────────────────────────
            paths = getattr(self._panel, '_cmb_cluster_paths', [])
            has_cluster = paths and paths[0] is not None and paths[0] != ""
            if not has_cluster:
                QMessageBox.warning(self, t("dialog_warning"), t("warn_no_lane_file"))
                return

            self._status.showMessage(t("status_auto_generating"))
            QApplication.processEvents()
            self._panel._auto_generate_lanes()

            self._toolbar_progress.setValue(80)
            QApplication.processEvents()

            # ── ステップ5: Snap工程 — lane_line_cluster.txt で法線方向補正 ──
            # _auto_generate_lanes() で生成した source="sampling_auto" のレコードに対して
            # lane_line_cluster.txt 点群を使い、法線方向（横方向）のみ座標を補正する。
            # ファイルが存在しない場合はプライアオフセットのまま継続する。
            self._apply_snap_to_auto_lanes()

            self._toolbar_progress.setValue(90)
            QApplication.processEvents()

            # レーン点群レイヤーを非表示にする
            self._gl.set_layer_visible("lane_points", False)
            self._gl.update()

            self._toolbar_progress.setValue(100)
            QApplication.processEvents()

            n_lanes = len(self._panel._lanes)
            self._status.showMessage(t("status_auto_gen_done").format(n_lanes))

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_auto_generate')} {e}")
            self._status.showMessage(t("status_auto_gen_fail_short"))
        finally:
            self._toolbar_progress.setVisible(False)

        # 自動生成ボタンをラップ(文字列マッチは文字化けのためスキップ)
        # for btn in self._panel.findChildren(QPushButton):
        #     if "自動生成" in btn.text() or "Auto" in btn.text():

    def _on_semi_auto_generate(self) -> None:
        """半自動生成ボタン押下時の処理。

        手動付与済みポリラインのオフセットを参照し、
        mapping_pose.txt と平行なラインをプログレスバー付きで生成する。
        """
        if self._panel is None or self._gl is None:
            return

        self._toolbar_progress.setValue(0)
        self._toolbar_progress.setVisible(True)
        QApplication.processEvents()

        try:
            self._toolbar_progress.setValue(20)
            QApplication.processEvents()

            # mapping_pose が読み込まれているか確認
            traj_path = getattr(self, '_trajectory_path', None)
            if not traj_path or not Path(traj_path).exists():
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, t("dialog_warning"), t("warn_semi_autogen_no_pose"))
                return

            self._toolbar_progress.setValue(40)
            QApplication.processEvents()

            self._status.showMessage(t("status_semi_auto_generating"))
            QApplication.processEvents()

            self._panel._semi_auto_generate_lanes()

            self._toolbar_progress.setValue(80)
            QApplication.processEvents()
            self._gl.update()

            # ── 編集用間引き: 5m間隔に頂点削減 ──────────────────────
            self._thin_auto_lanes_for_edit()

            self._toolbar_progress.setValue(100)
            QApplication.processEvents()

            n_lanes = len([r for r in self._panel._lanes
                           if r.get("source") == "semi_auto"])
            self._status.showMessage(t("status_semi_auto_gen_done").format(n_lanes))

        except Exception as e:
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_auto_generate')} {e}")
            self._status.showMessage(t("status_auto_gen_fail_short"))
        finally:
            self._toolbar_progress.setVisible(False)
        #         ... (文字化けによりスキップ)

    def _lanegen_ui_text(self, ja: str, en: str) -> str:
        """既存言語設定に合わせたLaneGen専用短文を返す。"""
        return ja if get_language() == "ja" else en

    def _find_lanegen_package_dir(self) -> Optional[Path]:
        """LaneGen Production V6 packageを既知の配置から検索する。"""
        candidates = []
        if self._lanegen_package_dir is not None:
            candidates.append(self._lanegen_package_dir)
        app_root = Path(__file__).resolve().parent.parent
        candidates += [
            app_root / "IBEV_lanegen_production_v6",
            app_root / "tools" / "IBEV_lanegen_production_v6",
            app_root / "IBEV_lanegen_production_v5",
            app_root / "tools" / "IBEV_lanegen_production_v5",
            app_root / "IBEV_lanegen_production_v4",
            app_root / "tools" / "IBEV_lanegen_production_v4",
        ]
        if self._root_folder_path:
            root = Path(self._root_folder_path)
            candidates += [
                root / "IBEV_lanegen_production_v6",
                root / "tools" / "IBEV_lanegen_production_v6",
                root / "IBEV_lanegen_production_v5",
                root / "tools" / "IBEV_lanegen_production_v5",
                root / "IBEV_lanegen_production_v4",
                root / "tools" / "IBEV_lanegen_production_v4",
            ]
        for candidate in candidates:
            if (candidate / "lanegen_run.py").is_file():
                self._lanegen_package_dir = candidate.resolve()
                return self._lanegen_package_dir

        selected = QFileDialog.getExistingDirectory(
            self,
            self._lanegen_ui_text(
                "IBEV LaneGen Production V6フォルダを選択",
                "Select IBEV LaneGen Production V6 folder",
            ),
            str(app_root),
        )
        if selected and (Path(selected) / "lanegen_run.py").is_file():
            self._lanegen_package_dir = Path(selected).resolve()
            return self._lanegen_package_dir
        return None

    def _load_lanegen_runner_class(self, package_dir: Path):
        """package内runnerを動的ロードし、Annotator側へのコピーを不要にする。"""
        module_path = package_dir / "lanegen_qt_runner.py"
        if not module_path.is_file():
            raise FileNotFoundError(module_path)
        module_name = "_bev_lanegen_qt_runner_v5"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(str(module_path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        return module.LaneGenQtRunner

    def _ensure_lanegen_runner(self, package_dir: Path):
        if self._lanegen_runner is not None:
            return self._lanegen_runner
        runner_cls = self._load_lanegen_runner_class(package_dir)
        runner = runner_cls(self)
        runner.started.connect(self._on_lanegen_started)
        runner.progress.connect(self._on_lanegen_progress)
        runner.log.connect(self._on_lanegen_log)
        runner.finished.connect(self._on_lanegen_finished)
        runner.failed.connect(self._on_lanegen_failed)
        self._lanegen_runner = runner
        return runner

    @staticmethod
    def _find_project_file(root: Path, names) -> Optional[Path]:
        """一般的な配置を優先し、最後にexact filenameを再帰検索する。"""
        bases = [root, root / "PCD", root / "pcd", root / "bev", root / "BEV"]
        for base in bases:
            for name in names:
                candidate = base / name
                if candidate.is_file():
                    return candidate.resolve()
        for name in names:
            for candidate in root.rglob(name):
                if "lanegen_output" not in candidate.parts and candidate.is_file():
                    return candidate.resolve()
        return None

    def _resolve_lanegen_inputs(self):
        if not self._root_folder_path:
            raise RuntimeError(
                self._lanegen_ui_text(
                    "先にルートフォルダを開いてください。",
                    "Open a project root folder first.",
                )
            )
        root = Path(self._root_folder_path).resolve()
        intensity = self._find_project_file(root, ["IBEV.pcd"])
        z_pcd = self._find_project_file(
            root, ["RGBBEV.pcd", "RGBBEV(1).pcd"]
        )
        mapping = (
            Path(self._trajectory_path).resolve()
            if self._trajectory_path and Path(self._trajectory_path).is_file()
            else self._find_project_file(root, ["mapping_pose.txt"])
        )
        missing = []
        if intensity is None:
            missing.append("IBEV.pcd")
        if mapping is None:
            missing.append("mapping_pose.txt")
        if z_pcd is None:
            missing.append("RGBBEV.pcd")
        if missing:
            raise FileNotFoundError(
                self._lanegen_ui_text(
                    "必要ファイルが見つかりません: ",
                    "Required files not found: ",
                ) + ", ".join(missing)
            )
        return root, intensity, mapping, z_pcd

    def _start_lanegen_v4(self) -> None:
        """Production V5を別プロセスで開始し、完了後JSONを自動取込する。"""
        try:
            package_dir = self._find_lanegen_package_dir()
            if package_dir is None:
                raise FileNotFoundError(
                    self._lanegen_ui_text(
                        "LaneGen Production V6フォルダが見つかりません。",
                        "LaneGen Production V6 folder was not found.",
                    )
                )
            root, intensity, mapping, z_pcd = self._resolve_lanegen_inputs()
            runner = self._ensure_lanegen_runner(package_dir)
            if runner.is_running:
                QMessageBox.information(
                    self,
                    "LaneGen",
                    self._lanegen_ui_text(
                        "初期生成はすでに実行中です。",
                        "Initial generation is already running.",
                    ),
                )
                return
            output_dir = root / "lanegen_output"
            self._lanegen_output_dir = output_dir
            runner.start_job(
                package_dir=package_dir,
                intensity_pcd=intensity,
                mapping_pose=mapping,
                z_pcd=z_pcd,
                rgb_pcd=z_pcd,
                output_dir=output_dir,
                stem="IBEV",
                resume=True,
                extra_args=(
                    "--stage1-aggregation-mode",
                    "streaming",
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "LaneGen", str(exc))

    def _cancel_lanegen_v4(self) -> None:
        runner = self._lanegen_runner
        if runner is None or not runner.is_running:
            return
        runner.request_cancel()
        self._btn_lanegen_cancel.setEnabled(False)
        self._status.showMessage(
            self._lanegen_ui_text(
                "LaneGen: 安全な停止を要求しました…",
                "LaneGen: cooperative cancellation requested...",
            )
        )

    def _on_lanegen_started(self) -> None:
        self._toolbar_progress.setValue(0)
        self._toolbar_progress.setVisible(True)
        self._btn_lanegen_generate.setEnabled(False)
        self._btn_lanegen_cancel.setEnabled(True)
        self._status.showMessage(
            self._lanegen_ui_text(
                "LaneGen Production V6: 初期アノテーション生成中…",
                "LaneGen Production V6: generating initial annotations...",
            )
        )

    def _on_lanegen_progress(self, payload: dict) -> None:
        fraction = float(payload.get("fraction", 0.0) or 0.0)
        self._toolbar_progress.setValue(
            max(0, min(100, round(fraction * 100)))
        )
        stage = payload.get("stage", "-")
        message = str(payload.get("message", payload.get("status", "")))
        self._status.showMessage(f"LaneGen Stage {stage}: {message}")

    def _on_lanegen_log(self, line: str) -> None:
        self._lanegen_last_log = str(line)
        print(f"[LaneGen] {line}")

    def _on_lanegen_finished(self, exit_code: int, state: dict) -> None:
        self._btn_lanegen_generate.setEnabled(True)
        self._btn_lanegen_cancel.setEnabled(False)
        status = str(state.get("status", ""))
        if exit_code == 2 or status == "CANCELLED":
            self._toolbar_progress.setVisible(False)
            self._status.showMessage(
                self._lanegen_ui_text(
                    "LaneGen: キャンセルしました。再開可能です。",
                    "LaneGen: cancelled. The job can be resumed.",
                )
            )
            return
        if exit_code != 0 or status != "COMPLETED":
            self._toolbar_progress.setVisible(False)
            QMessageBox.critical(
                self,
                "LaneGen",
                self._lanegen_ui_text(
                    "初期生成に失敗しました。\n",
                    "Initial generation failed.\n",
                ) + str(state.get("message", self._lanegen_last_log)),
            )
            return

        self._toolbar_progress.setValue(100)
        try:
            output = None
            outputs = state.get("outputs", {})
            if isinstance(outputs, dict) and outputs.get("annotator"):
                output = Path(outputs["annotator"])
            if output is None or not output.is_file():
                output_dir = self._lanegen_output_dir or Path(state["output_dir"])
                output = Path(output_dir) / "IBEV_annotator.json"
            if self._panel is None or not hasattr(
                self._panel, "import_lanegen_json"
            ):
                raise RuntimeError(
                    self._lanegen_ui_text(
                        "LaneLinePanelがLaneGen JSON取込に対応していません。",
                        "LaneLinePanel does not support LaneGen JSON import.",
                    )
                )
            result = self._panel.import_lanegen_json(
                str(output), replace_previous=True
            )
            self._status.showMessage(
                self._lanegen_ui_text(
                    f"LaneGen完了: {result['imported']}件取込、"
                    f"{result['replaced']}件置換",
                    f"LaneGen completed: imported {result['imported']}, "
                    f"replaced {result['replaced']}",
                )
            )
            QMessageBox.information(
                self,
                "LaneGen Production V6",
                self._lanegen_ui_text(
                    f"初期アノテーションを{result['imported']}件取り込みました。\n"
                    f"以前のLaneGen生成データは{result['replaced']}件置換しました。",
                    f"Imported {result['imported']} initial annotations.\n"
                    f"Replaced {result['replaced']} previous LaneGen records.",
                ),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "LaneGen Import",
                self._lanegen_ui_text(
                    "生成は完了しましたが、JSON取込に失敗しました。\n",
                    "Generation completed, but JSON import failed.\n",
                ) + str(exc),
            )
        finally:
            self._toolbar_progress.setVisible(False)

    def _on_lanegen_failed(self, message: str) -> None:
        self._btn_lanegen_generate.setEnabled(True)
        self._btn_lanegen_cancel.setEnabled(False)
        self._toolbar_progress.setVisible(False)
        QMessageBox.critical(self, "LaneGen", str(message))

    def _set_view_mode(self, mode: str) -> None:
        
        if self._gl is not None:
            self._gl.set_view_mode(mode)
        # 
        if hasattr(self, '_btn_view_2d'):
            self._btn_view_2d.setStyleSheet(
                self._BTN_VIEW_ON if mode == "2d" else self._BTN_VIEW_OFF)
            self._btn_view_3d.setStyleSheet(
                self._BTN_VIEW_ON if mode == "3d" else self._BTN_VIEW_OFF)

    def _fit_to_all(self) -> None:
        
        if self._gl is not None:
            self._gl.fit_to_all()

    def _reset_to_home_view(self) -> None:
        """基準向きに戻す。軌跡が読み込まれている場合は進行方向を画面上向きに回転する。
        ズームレベルとパンは変更しない。

        OpenGL の glRotatef(rot_z, 0,0,1) は反時計回りにワールドを回転させるため、
        カメラから見た地図は「時計回り rot_z 度」回転して見える。
        「進行方向が画面上向き」にするには: rot_z = heading_deg - 90
        """
        if self._gl is None:
            return
        self._gl.rot_x = 0.0
        # ズームとパンは維持する（変更しない）

        rot_z = 0.0
        poses = getattr(self, '_mapping_poses', None)
        if poses and len(poses) >= 2:
            import math as _math
            # 軌跡の最初の数点から進行方向を算出
            n = min(10, len(poses))
            x0 = poses[0]['x'];  y0 = poses[0]['y']
            x1 = poses[n - 1]['x']; y1 = poses[n - 1]['y']
            dx = x1 - x0;  dy = y1 - y0
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                # 進行方向の角度（度）: 0度=東(+X), 90度=北(+Y)
                heading_deg = _math.degrees(_math.atan2(dy, dx))
                # glRotatef(rot_z, 0,0,1) はワールドを反時計回りに回す
                # 見た目は地図が時計回りに rot_z 度回転する
                # 「進行方向が画面上 (+Y)」にするには: rot_z = heading_deg - 90
                rot_z = heading_deg - 90.0

        self._gl.rot_z = rot_z
        self._gl.update()

    def _set_cluster_file_to_panel(self, filepath: str) -> None:
        
        if self._panel is None:
            return
        self._panel._cmb_cluster.blockSignals(True)
        self._panel._cmb_cluster.clear()
        self._panel._cmb_cluster.addItem(Path(filepath).name)
        self._panel._cmb_cluster_paths = [filepath]
        self._panel._cmb_cluster.setCurrentIndex(0)
        self._panel._cmb_cluster.blockSignals(False)

    def _on_before_auto_generate(self) -> None:
        
        if self._panel is None:
            return
        paths = getattr(self._panel, '_cmb_cluster_paths', [])
        has_valid = paths and paths[0] is not None and paths[0] != ""
        if not has_valid:
            self._open_lane()
            # _open_lane 
            paths = getattr(self._panel, '_cmb_cluster_paths', [])
            if not paths or paths[0] is None:
                return
        self._panel._auto_generate_lanes()

    def _on_after_auto_generate(self) -> None:
        
        if self._gl is not None:
            self._gl.set_layer_visible("lane_points", False)
            self._gl.update()

    def get_default_attributes_for_class(self, cls_name: str) -> dict:
        
        from core.cvat_converter_bev import CLASS_ATTR_MAP

        attrs = {}
        attr_keys = CLASS_ATTR_MAP.get(cls_name, [])

        for key in attr_keys:
            if key == "line_type":
                attrs[key] = self._last_line_type
            elif key == "line_color":
                attrs[key] = self._last_line_color
            elif key == "line_count":
                attrs[key] = self._last_line_count
            elif key == "lane_number":
                attrs[key] = ""  # ID
            elif key == "boundary_id":
                attrs[key] = self._get_next_boundary_id(cls_name)
            elif key in ("start_line_ids", "end_line_ids"):
                attrs[key] = ""  # 

        return attrs

    def _get_next_boundary_id(self, cls_name: str) -> str:
        
        if self._panel is None:
            return "1"

        lanes = self._panel._lanes
        max_id = 0
        for rec in lanes:
            if rec.get("class") == cls_name:
                try:
                    bid = int(rec.get("boundary_id", "0"))
                    max_id = max(max_id, bid)
                except (ValueError, TypeError):
                    pass

        return str(max_id + 1)

    def update_last_attributes(self, record: dict) -> None:
        
        if record.get("class") == "lane_line":
            line_type = record.get("line_type", "")
            if line_type:
                self._last_line_type = line_type
            line_color = record.get("line_color", "")
            if line_color:
                self._last_line_color = line_color
            line_count = record.get("line_count", "")
            if line_count:
                self._last_line_count = line_count

        self._last_class = record.get("class", self._last_class)

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------
    def _build_menus(self) -> None:

        menubar = self.menuBar()

        # ================================================================
        # ファイルメニュー（4サブメニュー構成）
        # ================================================================
        self._menu_file = menubar.addMenu(t("menu_file"))

        # ---- [1] プロジェクト ----------------------------------------
        self._submenu_project = self._menu_file.addMenu(t("menu_file_project"))

        self._act_project_new = QAction(t("menu_project_new"), self)
        self._act_project_new.triggered.connect(self._project_new)
        self._submenu_project.addAction(self._act_project_new)

        self._act_project_open = QAction(t("menu_project_open"), self)
        self._act_project_open.triggered.connect(self._project_open)
        self._submenu_project.addAction(self._act_project_open)

        self._act_project_save = QAction(t("menu_project_save"), self)
        self._act_project_save.triggered.connect(self._project_save)
        self._submenu_project.addAction(self._act_project_save)

        self._menu_recent = self._submenu_project.addMenu(t("menu_project_recent"))
        self._rebuild_recent_menu()

        # ---- [2] ルート（フォルダ単位で開く） -----------------------
        self._menu_file.addSeparator()
        self._submenu_root = self._menu_file.addMenu(t("menu_submenu_root"))

        self._act_open_root_folder = QAction(t("menu_open_root_folder"), self)
        self._act_open_root_folder.triggered.connect(self._open_root_folder)
        self._submenu_root.addAction(self._act_open_root_folder)

        self._menu_recent_folders = self._submenu_root.addMenu(t("menu_recent_folders"))
        self._rebuild_recent_folders_menu()

        # サンプルPCDを開く
        self._submenu_root.addSeparator()
        self._act_open_sample_pcd = QAction(t("menu_open_sample_pcd"), self)
        self._act_open_sample_pcd.triggered.connect(self._open_sample_pcd)
        self._submenu_root.addAction(self._act_open_sample_pcd)

        # ---- [3] カメラ ----------------------------------------------
        self._menu_file.addSeparator()
        self._submenu_camera = self._menu_file.addMenu(t("menu_submenu_camera"))

        self._act_open_camera = QAction(t("menu_open_camera"), self)
        self._act_open_camera.triggered.connect(self._open_camera_folder)
        self._submenu_camera.addAction(self._act_open_camera)

        # ---- [4] アノテーション -------------------------------------
        self._menu_file.addSeparator()
        self._submenu_anno = self._menu_file.addMenu(t("menu_file_annotation_group"))

        # 作業データ（開く・保存のみ）
        self._submenu_anno_work = self._submenu_anno.addMenu(t("menu_anno_workdata"))

        self._act_open_annotation = QAction(t("menu_open_annotation"), self)
        self._act_open_annotation.triggered.connect(self._open_annotation)
        self._submenu_anno_work.addAction(self._act_open_annotation)

        self._act_save = QAction(t("menu_save_annotations"), self)
        self._act_save.triggered.connect(self._save_annotations)
        self._submenu_anno_work.addAction(self._act_save)

        # サンプルデータ
        self._submenu_anno_work.addSeparator()
        self._act_open_sample_json = QAction(t("menu_open_sample_json"), self)
        self._act_open_sample_json.triggered.connect(self._open_sample_json)
        self._submenu_anno_work.addAction(self._act_open_sample_json)

        self._act_open_sample_pkl = QAction(t("menu_open_sample_pkl"), self)
        self._act_open_sample_pkl.triggered.connect(self._open_sample_pkl)
        self._submenu_anno_work.addAction(self._act_open_sample_pkl)

        # 最終アノテーション出力
        self._submenu_anno.addSeparator()
        self._submenu_anno_output = self._submenu_anno.addMenu(t("menu_anno_final_output"))

        self._act_export_gt_json = QAction(t("menu_export_gt_json"), self)
        self._act_export_gt_json.triggered.connect(self._export_gt_json)
        self._submenu_anno_output.addAction(self._act_export_gt_json)

        self._act_export_pkl = QAction(t("menu_export_pkl"), self)
        self._act_export_pkl.triggered.connect(self._export_lane_pkl)
        self._submenu_anno_output.addAction(self._act_export_pkl)

        # ---- 終了 ---------------------------------------------------
        self._menu_file.addSeparator()
        self._act_exit = QAction(t("menu_exit"), self)
        self._act_exit.triggered.connect(self.close)
        self._menu_file.addAction(self._act_exit)

        #  CVAT  
        self._menu_cvat = menubar.addMenu(t("menu_cvat"))

        self._act_cvat_connect = QAction(t("cvat_connect"), self)
        self._act_cvat_connect.triggered.connect(self._cvat_connect)
        self._menu_cvat.addAction(self._act_cvat_connect)

        self._act_cvat_register = QAction(t("cvat_register_tasks"), self)
        self._act_cvat_register.triggered.connect(self._cvat_register_tasks)
        self._menu_cvat.addAction(self._act_cvat_register)

        self._menu_cvat.addSeparator()

        self._act_cvat_upload = QAction(t("cvat_upload"), self)
        self._act_cvat_upload.triggered.connect(self._cvat_upload)
        self._menu_cvat.addAction(self._act_cvat_upload)

        self._act_cvat_complete = QAction(t("cvat_complete_job"), self)
        self._act_cvat_complete.triggered.connect(self._cvat_complete_job)
        self._menu_cvat.addAction(self._act_cvat_complete)

        #   
        self._menu_lang = menubar.addMenu(t("menu_language"))

        self._act_lang_ja = QAction(t("menu_lang_ja"), self)
        self._act_lang_ja.triggered.connect(lambda: self._switch_language("ja"))
        self._menu_lang.addAction(self._act_lang_ja)

        self._act_lang_en = QAction(t("menu_lang_en"), self)
        self._act_lang_en.triggered.connect(lambda: self._switch_language("en"))
        self._menu_lang.addAction(self._act_lang_en)

        #   
        self._menu_quality = menubar.addMenu(t("menu_quality"))

        self._act_quality_run = QAction(t("menu_quality_run"), self)
        self._act_quality_run.triggered.connect(self._run_quality_check)
        self._menu_quality.addAction(self._act_quality_run)

        self._act_quality_settings = QAction(t("menu_quality_settings"), self)
        self._act_quality_settings.triggered.connect(self._show_quality_settings)
        self._menu_quality.addAction(self._act_quality_settings)

        self._act_quality_export = QAction(t("menu_quality_export"), self)
        self._act_quality_export.triggered.connect(self._export_quality_report)
        self._menu_quality.addAction(self._act_quality_export)

        #   
        self._menu_annotation = menubar.addMenu(t("menu_annotation"))

        self._act_auto_assign_all = QAction(t("menu_auto_assign_all"), self)
        self._act_auto_assign_all.triggered.connect(self._auto_assign_all)
        self._menu_annotation.addAction(self._act_auto_assign_all)

        self._menu_annotation.addSeparator()

        self._act_renumber_ids = QAction(t("menu_renumber_ids"), self)
        self._act_renumber_ids.triggered.connect(self._renumber_ids)
        self._menu_annotation.addAction(self._act_renumber_ids)

        self._menu_annotation.addSeparator()

        self._act_auto_assign_n = QAction(t("menu_auto_assign_n"), self)
        self._act_auto_assign_n.triggered.connect(self._auto_assign_n)
        self._menu_annotation.addAction(self._act_auto_assign_n)

        self._act_auto_assign_x = QAction(t("menu_auto_assign_x"), self)
        self._act_auto_assign_x.triggered.connect(self._auto_assign_x)
        self._menu_annotation.addAction(self._act_auto_assign_x)

        self._act_auto_assign_y = QAction(t("menu_auto_assign_y"), self)
        self._act_auto_assign_y.triggered.connect(self._auto_assign_y)
        self._menu_annotation.addAction(self._act_auto_assign_y)

        self._menu_annotation.addSeparator()

        self._act_auto_boundary_ids = QAction(t("menu_auto_boundary_ids"), self)
        self._act_auto_boundary_ids.triggered.connect(self._auto_assign_boundary_ids)
        self._menu_annotation.addAction(self._act_auto_boundary_ids)

        self._menu_annotation.addSeparator()

        self._act_suggest_split_merge = QAction(t("menu_suggest_split_merge"), self)
        self._act_suggest_split_merge.triggered.connect(
            self._suggest_split_merge_ids
        )
        self._menu_annotation.addAction(self._act_suggest_split_merge)

        # ────────────────────────────────────────────
        # ツールメニュー
        # ────────────────────────────────────────────
        self._menu_tools = menubar.addMenu(t("menu_tools"))

        self._act_measure_toggle = QAction(t("menu_measure_distance"), self)
        self._act_measure_toggle.setCheckable(True)
        self._act_measure_toggle.setChecked(False)
        self._act_measure_toggle.triggered.connect(self._toggle_measure_mode)
        self._menu_tools.addAction(self._act_measure_toggle)

        self._act_measure_reset = QAction(t("menu_measure_reset"), self)
        self._act_measure_reset.triggered.connect(self._reset_measure)
        self._menu_tools.addAction(self._act_measure_reset)

        # CVAT / 品質チェック / アノテーション支援 / ツール をグレーアウト
        # スタイルシートを各メニューに個別設定して文字色を明示的にグレーにする
        _DISABLED_MENU_STYLE = (
            "QMenu { color: #555555; }"
            "QMenu::item { color: #555555; }"
            "QMenu::item:disabled { color: #555555; }"
        )
        for menu in (self._menu_cvat, self._menu_quality,
                     self._menu_annotation, self._menu_tools):
            menu.setEnabled(False)
            menu.setStyleSheet(_DISABLED_MENU_STYLE)

        # メニューバー側でも disabled 色を設定
        self.setStyleSheet(
            self.styleSheet()
            + "QMenuBar::item:disabled { color: #555555; }"
        )

    # ------------------------------------------------------------------
    # 距離計測
    # ------------------------------------------------------------------
    def _toggle_measure_mode(self, checked: bool) -> None:
        """距離計測モードのON/OFFを切り替える。"""
        if self._gl is None:
            return
        self._gl._measure_mode = checked
        if not checked:
            # モード終了時に計測値をクリア
            self._gl._measure_pt1 = None
            self._gl._measure_pt2 = None
            self._gl._measure_hover = None
        self._gl.update()
        if checked:
            self._status.showMessage(t("status_measure_on"))
        else:
            self._status.showMessage(t("status_measure_off"))

    def _reset_measure(self) -> None:
        """計測点をリセットする（モードは維持）。"""
        if self._gl is None:
            return
        self._gl._measure_pt1 = None
        self._gl._measure_pt2 = None
        self._gl._measure_hover = None
        self._gl.update()
        self._status.showMessage(t("status_measure_reset"))

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------
    def _build_status_bar(self) -> None:
        
        self._status = self.statusBar()
        self._status.showMessage(t("status_ready"))

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------
    def _show_login_dialog(self) -> None:
        
        if CvatLoginDialog is None:
            return
        config = self._cvat_config or (CvatConfig() if CvatConfig else None)
        if config is None:
            return
        dlg = CvatLoginDialog(self, config=config)
        if dlg.exec_() == QDialog.Accepted:
            self._cvat_client = dlg.client
            self._cvat_config = dlg.config
            self._status.showMessage(
                f"{t('status_connected')} ({dlg.config.host})"
            )
            # 
            self._open_cvat_task_dialog()
        else:
            self._status.showMessage(t("cvat_offline_mode"))


    # ==================================================================
    # 7.2  
    # ==================================================================

    def _open_image(self) -> None:
        
        if DataLoader is None or self._gl is None:
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            t("menu_open_image"),
            "",
            t("file_filter_image"),
        )
        if not filepath:
            return

        self._load_image_file(filepath)
        #  _annotations.xml 
        self._auto_restore_annotations(filepath)

    def _auto_restore_annotations(self, image_path: str) -> None:
        
        if BevCvatConverter is None or self._panel is None or self._gl is None:
            return

        p = Path(image_path)
        xml_path = p.parent / f"{p.stem}_annotations.xml"
        if not xml_path.exists():
            return

        try:
            il = self._gl._bev_image
            converter = BevCvatConverter(il)
            records = converter.cvat_xml_to_lanes(str(xml_path))
            if records:
                self._panel._lanes = records
                self._gl.draw_lane_data = records
                if hasattr(self._panel, '_refresh_table'):
                    self._panel._refresh_table()
                if hasattr(self._panel, '_rebuild_all_poly3d'):
                    self._panel._rebuild_all_poly3d()
                self._gl.update()
                self._status.showMessage(
                    f"{t('status_annotations_loaded')} {xml_path.name}"
                )
        except Exception as e:
            print(f"[MainWindow] Auto-restore failed: {e}")

    def _auto_generate_from_intensity(self, image_path: str) -> None:
        
        if self._panel is None or self._gl is None:
            return

        # 
        if self._panel._lanes:
            return

        img_dir = Path(image_path).parent

        #  1:  cluster.txt 
        cluster_path = self._find_existing_cluster_txt(img_dir)

        #  2: cluster.txt  intensity ｰ
        if cluster_path is None:
            cluster_path = self._generate_cluster_from_intensity(img_dir)

        #  3: intensity  PCD  
        if cluster_path is None:
            intensity_path = self._generate_intensity_from_pcd(img_dir)
            if intensity_path is not None:
                cluster_path = self._generate_cluster_from_intensity(img_dir)

        if cluster_path is None:
            return  # 

        #  cluster.txt 
        try:
            self._status.showMessage("...")
            QApplication.processEvents()

            lane_points = DataLoader.parse_lane_points(cluster_path)
            if len(lane_points) > 0:
                colors = DataLoader.generate_cluster_colors(lane_points[:, 3])
                self._gl.set_lane_points(lane_points[:, :3], colors)

            # 
            self._set_cluster_file_to_panel(cluster_path)

            # 
            self._panel._auto_generate_lanes()

            # 
            self._gl.set_layer_visible("lane_points", False)
            self._gl.update()

            n_lanes = len(self._panel._lanes)
            self._status.showMessage(
                f" {n_lanes}"
            )

        except Exception as e:
            print(f"[MainWindow] Auto-generate failed: {e}")
            import traceback
            traceback.print_exc()
            self._status.showMessage(t("status_auto_gen_fail"))

    def _thin_auto_lanes_for_edit(self) -> None:
        """自動生成ライン（source="sampling_auto"）を5m間隔に間引いて編集しやすくする。

        軌跡全フレームコピーで生成された数百〜数千頂点のポリラインを
        5m間隔に削減する。pkl/txt出力時は20m間隔リサンプルで自動調整されるため
        情報の損失はない。GT JSON出力時はスプライン補間で0.5m間隔に再生成される。
        """
        if self._panel is None:
            return

        try:
            from core.gt_lane_generator import thin_lanes_for_edit

            # source="sampling_auto" または "semi_auto" のレコードを対象にする
            target_sources = {"sampling_auto", "semi_auto"}
            targets = [
                r for r in self._panel._lanes
                if r.get("source") in target_sources
                and r.get("geometry_type", r.get("shape", "")) == "polyline"
            ]
            others = [
                r for r in self._panel._lanes
                if r.get("source") not in target_sources
                or r.get("geometry_type", r.get("shape", "")) != "polyline"
            ]

            if not targets:
                return

            thinned = thin_lanes_for_edit(
                targets,
                interval_m=20.0,    # RDP後に直線が残った場合の最大間隔
                rdp_epsilon_m=0.3,  # 許容誤差0.3m: 直線は2点に削減、R=50mカーブは9点保持
            )

            self._panel._lanes = others + thinned
            self._gl.draw_lane_data = self._panel._lanes
            self._gl.update()
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()

            print(
                f"[MainWindow] 編集用間引き完了: {len(thinned)} 本 "
                f"(5m間隔、出力時は20m間隔pkl / 0.5m間隔GT JSONに自動変換)"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MainWindow] 編集用間引きエラー（スキップ）: {e}")

    def _apply_snap_to_auto_lanes(self) -> None:
        """全自動生成後のSnap工程。

        _auto_generate_lanes() で生成した source="sampling_auto" のレコードに対して
        lane_line_cluster.txt 点群を使い、法線方向（横方向）のみ座標を補正する。

        処理フロー:
          1. lane_line_cluster.txt を探して読み込む
          2. mapping_pose.txt から軌跡データを取得する
          3. snap_lanes_to_cluster() で各ラインの d(s) を補正する
          4. 補正後のレコードで _lanes を更新してGLを再描画する

        ファイルが存在しない・mapping_pose が未ロードの場合はスキップする。
        """
        if self._panel is None or DataLoader is None:
            return

        # ── mapping_pose データを取得 ────────────────────────────────
        traj_path = getattr(self, '_trajectory_path', None)
        if not traj_path or not Path(traj_path).exists():
            print("[MainWindow] Snap: mapping_pose 未ロード → Snap スキップ")
            return

        # ── lane_line_cluster.txt を探して読み込む ───────────────────
        cluster_path = self._find_lane_line_cluster_txt()
        if cluster_path is None:
            print("[MainWindow] Snap: lane_line_cluster.txt が見つからない → Snap スキップ")
            return

        self._status.showMessage(f"Snap点群読み込み中: {Path(cluster_path).name}")
        QApplication.processEvents()

        cluster_points = DataLoader.parse_lane_points(cluster_path)
        if len(cluster_points) == 0:
            print(f"[MainWindow] Snap: 有効点なし ({cluster_path}) → Snap スキップ")
            return

        print(
            f"[MainWindow] Snap工程開始: "
            f"点群 {len(cluster_points)} 点 ({Path(cluster_path).name})"
        )

        # ── 軌跡データを NumPy 配列で取得 ───────────────────────────
        try:
            from core.gt_lane_generator import parse_mapping_pose, snap_lanes_to_cluster
            poses = parse_mapping_pose(traj_path)
            pose_xy_full  = poses[:, :2]
            pose_yaw_full = poses[:, 3]
        except Exception as e:
            print(f"[MainWindow] Snap: mapping_pose 読み込みエラー: {e} → Snap スキップ")
            return

        # ── Snap 対象: source="sampling_auto" のレコードのみ ────────
        snap_targets = [
            r for r in self._panel._lanes
            if r.get("source") == "sampling_auto"
        ]
        other_lanes = [
            r for r in self._panel._lanes
            if r.get("source") != "sampling_auto"
        ]

        if not snap_targets:
            print("[MainWindow] Snap: sampling_auto レコードなし → Snap スキップ")
            return

        # ── Snap 実行 ────────────────────────────────────────────────
        self._status.showMessage(
            f"Snap補正中: {len(snap_targets)} 本..."
        )
        QApplication.processEvents()

        try:
            snapped = snap_lanes_to_cluster(
                records=snap_targets,
                cluster_points=cluster_points,
                pose_xy_full=pose_xy_full,
                pose_yaw_full=pose_yaw_full,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MainWindow] Snap: 補正中にエラー: {e} → プライア維持")
            return

        # ── _lanes を更新して GL を再描画 ────────────────────────────
        self._panel._lanes = other_lanes + snapped
        self._gl.draw_lane_data = self._panel._lanes
        self._gl.update()
        if hasattr(self._panel, '_refresh_table'):
            self._panel._refresh_table()

        print(
            f"[MainWindow] Snap工程完了: "
            f"{len(snapped)} 本補正 ({Path(cluster_path).name})"
        )

        # ── 編集用間引き: Snap後のラインを5m間隔に削減 ──────────────
        # 軌跡全フレームコピーで生成された数百〜数千頂点のポリラインを
        # 5m間隔に間引いて編集しやすい頂点数にする（出力時は高密度で再生成）
        self._thin_auto_lanes_for_edit()

    def _find_lane_line_cluster_txt(self) -> "str | None":
        """Snap工程用の lane_line_cluster.txt を検索して返す。

        探索優先順位:
          1. ルートフォルダ直下の lane_line_cluster.txt
          2. BEV画像と同じディレクトリ
          3. BEV画像の親ディレクトリ
          4. trajectory（mapping_pose.txt）と同じディレクトリ

        Returns:
            見つかったファイルパス（str）。見つからない場合は None。
        """
        candidate_names = ["lane_line_cluster.txt"]

        search_dirs: list = []

        # ① ルートフォルダ
        root = getattr(self, '_root_folder_path', None)
        if root and Path(root).is_dir():
            search_dirs.append(Path(root))

        # ② BEV画像ディレクトリ
        img_path = getattr(self, '_image_path', None)
        if img_path:
            p = Path(img_path)
            search_dirs.append(p.parent)
            search_dirs.append(p.parent.parent)

        # ③ mapping_pose.txt と同じディレクトリ
        traj = getattr(self, '_trajectory_path', None)
        if traj:
            search_dirs.append(Path(traj).parent)

        # 重複排除・存在確認
        seen: set = set()
        for d in search_dirs:
            if d in seen or not d.exists():
                continue
            seen.add(d)
            for name in candidate_names:
                cand = d / name
                if cand.exists():
                    print(f"[MainWindow] lane_line_cluster.txt を発見: {cand}")
                    return str(cand)

        return None

    def _find_existing_cluster_txt(self, img_dir: Path) -> "str | None":
        
        search_dirs = [img_dir]

        #  (mapqr_input result)
        parent = img_dir.parent
        search_dirs.append(parent)

        # img_dir mapqr_input  result 
        if img_dir.name == "mapqr_input":
            search_dirs.append(parent)  # = result 
        else:
            # img_dir  result ｰ
            result_dir = img_dir / "result"
            if result_dir.exists():
                search_dirs.append(result_dir)
            #  result
            parent_result = parent / "result"
            if parent_result.exists():
                search_dirs.append(parent_result)

        for d in search_dirs:
            for pattern in ["lane_points_in_cluster.txt", "*cluster*.txt"]:
                found = list(d.glob(pattern))
                if found:
                    path = str(found[0])
                    print(f"[MainWindow] Found existing cluster.txt: {path}")
                    return path

        return None

    def _generate_cluster_from_intensity(self, img_dir: Path) -> "str | None":
        
        # IBEV_intensity.png 
        intensity_path = None
        for pattern in ["IBEV_intensity.png", "*_intensity.png", "*intensity*.png"]:
            found = list(img_dir.glob(pattern))
            if found:
                intensity_path = found[0]
                break

        if intensity_path is None:
            return None

        # 
        meta_path = intensity_path.with_name(
            intensity_path.stem + "_meta.json"
        )
        if not meta_path.exists():
            for p in img_dir.glob("*_meta.json"):
                meta_path = p
                break
        if not meta_path.exists():
            print("[MainWindow] intensity meta.json not found")
            return None

        try:
            self._status.showMessage("ｰ...")
            QApplication.processEvents()

            import numpy as np
            import cv2
            import json as _json
            import tempfile

            # OpenCV 
            img_data = np.fromfile(str(intensity_path), dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[MainWindow] Failed to read: {intensity_path}")
                return None

            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = _json.load(f)

            from intensity_to_cluster import extract_lane_clusters, write_cluster_txt

            # 
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img)

            points = extract_lane_clusters(
                tmp_path, meta,
                threshold=180, min_area=30,
                max_clusters=500, sample_rate=2,
            )
            os.remove(tmp_path)

            if not points:
                return None

            # cluster.txt 1
            cluster_path = str(img_dir / "lane_points_in_cluster.txt")
            write_cluster_txt(points, cluster_path)
            return cluster_path

        except Exception as e:
            print(f"[MainWindow] Intensity cluster failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_intensity_from_pcd(self, img_dir: Path) -> "Path | None":
        
        # IBEV.pcd 
        pcd_path = None
        for pattern in ["IBEV.pcd", "*BEV.pcd", "*bev.pcd"]:
            found = list(img_dir.glob(pattern))
            if found:
                pcd_path = found[0]
                break

        if pcd_path is None:
            return None

        try:
            self._status.showMessage(f"PCD BEV: {pcd_path.name}...")
            QApplication.processEvents()

            # pcd_to_grid_image 
            import sys
            pcd_tool_dir = str(Path(__file__).resolve().parent.parent.parent /
                               "pointcloud_Viewer")
            if pcd_tool_dir not in sys.path:
                sys.path.insert(0, pcd_tool_dir)
            # 01_code 
            pcd_tool_dir2 = str(Path(__file__).resolve().parent.parent.parent /
                                "pointcloud_Viewer" / "01_code")
            if pcd_tool_dir2 not in sys.path:
                sys.path.insert(0, pcd_tool_dir2)

            from pcd_to_grid_image import load_pcd, pcd_to_grid_image, save_image

            #  
            import tempfile
            import shutil
            tmp_pcd = tempfile.NamedTemporaryFile(suffix='.pcd', delete=False)
            tmp_pcd.close()
            shutil.copy2(str(pcd_path), tmp_pcd.name)

            pcd_data = load_pcd(tmp_pcd.name)
            os.remove(tmp_pcd.name)

            print(f"[info] PCD {len(pcd_data['xyz'])} points")

            # intensity  BEV
            img_rgba, meta = pcd_to_grid_image(
                pcd_data,
                mode="intensity",
                resolution=0.05,
                colormap="gray",
                ground_filter=True,
                scale=1,
                dilate=2,
            )

            # 
            output_path = str(img_dir / "IBEV_intensity.png")
            save_image(img_rgba, output_path, meta, save_meta=True)

            # composite 
            composite_path = str(img_dir / "RGBBEV_composite.png")
            if not (img_dir / "RGBBEV_composite.png").exists():
                try:
                    if pcd_data.get("rgb") is not None:
                        img_comp, meta_comp = pcd_to_grid_image(
                            pcd_data, mode="rgb", resolution=0.05,
                            colormap="viridis", ground_filter=True,
                            scale=1, dilate=2,
                        )
                        save_image(img_comp, composite_path, meta_comp, save_meta=True)
                    else:
                        # RGB  z_gradient composite 
                        img_zg, meta_zg = pcd_to_grid_image(
                            pcd_data, mode="z_gradient", resolution=0.05,
                            colormap="gray_r", ground_filter=True,
                            scale=1, dilate=2,
                        )
                        save_image(img_zg, composite_path, meta_zg, save_meta=True)
                except Exception as e:
                    print(f"[info] composite: {e}")

            self._status.showMessage(
                f"BEV {Path(output_path).name}"
            )
            return Path(output_path)

        except ImportError as e:
            print(f"[MainWindow] pcd_to_grid_image import failed: {e}")
            print(f"[info] pointcloud_Viewer ")
            return None
        except Exception as e:
            print(f"[MainWindow] PCD BEV generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _open_lane(self) -> None:
        
        if DataLoader is None or self._gl is None:
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("menu_open_lane"), "", t("file_filter_lane"),
        )
        if not filepath:
            return
        self._load_lane_file(filepath)

    def _open_trajectory(self) -> None:
        
        if self._gl is None:
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self, t("menu_open_files"), "", t("file_filter_pose"),
        )
        if not filepath:
            return
        self._load_trajectory_file(filepath)

    def _open_ref_pcd(self) -> None:
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            t("menu_open_ref_pcd"),
            "",
            t("file_filter_pcd"),
        )
        if not filepath:
            return
        self._load_ref_pcd_file(filepath)

    def _open_camera_folder(self) -> None:
        
        if ImageWindow is None:
            QMessageBox.warning(
                self, t("dialog_warning"), t("warn_camera_module")
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self, t("warn_select_folder_camera"), ""
        )
        if not folder:
            return

        try:
            self._status.showMessage("...")
            QApplication.processEvents()

            # cam-* 
            cam_dirs: dict[str, str] = {}
            for d in sorted(os.listdir(folder)):
                full = os.path.join(folder, d)
                if os.path.isdir(full) and d.startswith("cam-"):
                    cam_dirs[d] = full

            if not cam_dirs:
                #  cam-* 
                from pathlib import Path as _Path
                parent = str(_Path(folder).parent)
                folder_name = _Path(folder).name
                if folder_name.startswith("cam-"):
                    #  cam-* 
                    for d in sorted(os.listdir(parent)):
                        full = os.path.join(parent, d)
                        if os.path.isdir(full) and d.startswith("cam-"):
                            cam_dirs[d] = full
                    folder = parent  # 

            if not cam_dirs:
                QMessageBox.warning(
                    self, t("dialog_warning"),
                    f"{t('warn_no_cam_folder')}:\n{folder}"
                )
                return

            # 
            cam_suffixes = []
            for cam_id in sorted(cam_dirs.keys(),
                                  key=lambda c: self._cam_sort_key(c)):
                suffix = cam_id[4:]  # "cam-1" "1", "cam-10" "10"
                cam_suffixes.append(suffix)

            # 
            all_images: list[dict] = []
            for cam_id, cam_path in sorted(cam_dirs.items(),
                                            key=lambda x: self._cam_sort_key(x[0])):
                for img_f in sorted(os.listdir(cam_path)):
                    if not img_f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img_path = os.path.join(cam_path, img_f)
                    all_images.append({
                        "camera": cam_id,
                        "image_path": img_path,
                    })

            if not all_images:
                QMessageBox.warning(
                    self, t("dialog_warning"), t("warn_no_images")
                )
                return

            self._camera_images = all_images

            # /
            if self._image_window is not None:
                self._image_window.close()

            self._image_window = ImageWindow(
                cam_suffixes=cam_suffixes,
                parent=self,
            )
            self._image_window.show_images(
                all_images,
                frame_info=f": {Path(folder).name}  "
                           f": {len(cam_suffixes)}",
            )
            self._image_window.show()

            self._status.showMessage(
                f" {len(all_images)}"
                f"({len(cam_suffixes)}: {', '.join(cam_suffixes)})"
            )

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"),
                f"{t('err_camera_load')}{e}",
            )
            self._status.showMessage(t("status_camera_fail"))

    @staticmethod
    def _cam_sort_key(cam_id: str) -> tuple:
        
        name = cam_id[4:] if cam_id.startswith("cam-") else cam_id
        num_str = ""
        rest = ""
        for i, ch in enumerate(name):
            if ch.isdigit():
                num_str += ch
            else:
                rest = name[i:]
                break
        if num_str:
            return (0, int(num_str), rest)
        return (1, 0, name)

    def _save_annotations(self) -> None:
        
        if self._cvat_client is not None and self._cvat_task_id is not None:
            self._cvat_upload()
            return

        #  
        self._save_local_annotations()

    def _save_local_annotations(self) -> None:
        
        if (BevCvatConverter is None or self._panel is None
                or self._gl is None):
            return

        lanes = self._panel._lanes
        if not lanes:
            QMessageBox.information(
                self, t("dialog_info"), t("err_no_annotations")
            )
            return

        il = self._gl._bev_image
        if il is None:
            QMessageBox.warning(
                self, t("dialog_warning"), t("err_no_image_layer")
            )
            return

        try:
            self._status.showMessage(t("status_saving"))
            QApplication.processEvents()

            converter = BevCvatConverter(il)
            # source=="prelabel" のレコードは保存対象から除外する
            # （prelabelは毎回 sampling_lane_global.json から再生成するため保存不要）
            lanes_to_save = [r for r in lanes if r.get("source") != "prelabel"]
            n_excluded = len(lanes) - len(lanes_to_save)
            if n_excluded:
                print(f"[MainWindow] 保存: prelabelレコード {n_excluded} 本を除外")
            xml_str = converter.lanes_to_cvat_xml(
                lanes_to_save,
                image_name=il.name or "unknown.png",
                image_width=il.width_px,
                image_height=il.height_px,
            )

            # : ｧ / {stem}_annotations.xml
            if self._image_path:
                p = Path(self._image_path)
                save_path = p.parent / f"{p.stem}_annotations.xml"
            else:
                save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    t("menu_save_annotations"),
                    "annotations.xml",
                    t("file_filter_xml"),
                )
                if not save_path:
                    self._status.showMessage(t("status_ready"))
                    return
                save_path = Path(save_path)

            save_path.write_text(xml_str, encoding="utf-8")
            self._has_unsaved_changes = False

            self._status.showMessage(
                f"{t('status_annotations_saved')} {save_path.name}"
            )
            QMessageBox.information(
                self,
                t("dialog_save_success"),
                f"{t('dialog_save_success_msg')}\n{save_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                t("dialog_save_fail"),
                f"{t('dialog_save_fail_msg')} {e}",
            )
            self._status.showMessage(t("status_ready"))

    # ==================================================================
    # 7.2b  kl
    # ==================================================================

    def _export_lane_pkl(self) -> None:
        """車線pkl出力（サンプルpklフォーマット準拠）とTXT出力を同時に行う。

        保存先: {ルートフォルダ}/PCD_pkl/{ルートフォルダ名}.pkl/.txt
        ルートフォルダが未設定の場合はダイアログで保存先を選択する。
        """
        if self._panel is None or not self._panel._lanes:
            QMessageBox.warning(
                self, t("menu_export_pkl"),
                t("warn_no_export_data")
            )
            return

        # ── ID未付与チェック（lane_line + boundary 系を含む全ポリライン対象） ──
        # split_point / merge_point / zebra_line / stop_line は除外
        _ID_EXEMPT_CLASSES = {"split_point", "merge_point", "zebra_line", "stop_line"}
        missing_id_lanes = [
            rec for rec in self._panel._lanes
            if rec.get("class") not in _ID_EXEMPT_CLASSES
            and rec.get("geometry_type", rec.get("shape", "")) == "polyline"
            and not rec.get("lane_id_n", "").strip()
        ]
        if missing_id_lanes:
            ids_preview = ", ".join(
                f"{rec.get('class','?')}({rec.get('id','?')})"
                for rec in missing_id_lanes[:5]
            )
            suffix = f"... 他 {len(missing_id_lanes) - 5} 件" if len(missing_id_lanes) > 5 else ""
            QMessageBox.warning(
                self, t("menu_export_pkl"),
                t("warn_missing_n_id").format(ids_preview + suffix)
            )
            return

        # ── 保存先を自動決定 ──────────────────────────────────────────
        save_path: Optional[str] = None
        if self._root_folder_path:
            root = Path(self._root_folder_path)
            pkl_dir = root / "PCD_pkl"
            save_path = str(pkl_dir / f"{root.name}.pkl")
        else:
            # ルートフォルダ未設定時はダイアログ
            if self._image_path:
                base_dir = Path(self._image_path).parent
                stem = Path(self._image_path).stem
                default_pkl = str(base_dir / "results" / f"{stem}.pkl")
            else:
                default_pkl = "output.pkl"
            save_path, _ = QFileDialog.getSaveFileName(
                self, t("menu_export_pkl"),
                default_pkl, "Pickle Files (*.pkl);;All Files (*)"
            )
            if not save_path:
                return

        output_pkl = save_path
        output_txt = str(Path(save_path).with_suffix(".txt"))

        try:
            from core.pkl_exporter import export_lane_pkl
            import json as _json

            # BEVメタ情報を取得(座標変換用)
            bev_meta = getattr(self._panel, '_bev_meta', None) if self._panel else None

            # 軌跡データ
            trajectory = None
            if self._mapping_poses and len(self._mapping_poses) >= 2:
                import numpy as _np
                trajectory = _np.array(
                    [[p['x'], p['y'], p.get('z', 0.0)] for p in self._mapping_poses],
                    dtype=_np.float64)
            elif self._gl is not None and self._gl._trajectory_points is not None:
                trajectory = self._gl._trajectory_points.astype(float)

            result = export_lane_pkl(
                self._panel._lanes,
                output_pkl,
                dims=3,
                bev_meta=bev_meta,
                pcd_z_resolver=self._pcd_z_resolver,
                trajectory=trajectory,
            )

            # ── TXT 出力（JSON形式、float→Pythonネイティブ変換）──────
            import numpy as _np

            def _to_serializable(obj):
                """numpy型をPythonネイティブ型に変換する"""
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (_np.floating, _np.float32, _np.float64)):
                    return float(obj)
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                return obj

            def _convert_dict(d):
                out = {}
                for k, v in d.items():
                    if isinstance(v, list):
                        out[k] = [_convert_entry(e) if isinstance(e, dict) else v for e in v]
                    else:
                        out[k] = _to_serializable(v)
                return out

            def _convert_entry(entry):
                out = {}
                for k, v in entry.items():
                    if isinstance(v, _np.ndarray):
                        out[k] = v.tolist()
                    elif isinstance(v, (_np.floating, _np.float32, _np.float64)):
                        out[k] = float(v)
                    elif isinstance(v, (_np.integer,)):
                        out[k] = int(v)
                    else:
                        out[k] = v
                return out

            txt_data = {}
            for k, entries in result.items():
                txt_data[k] = [_convert_entry(e) for e in entries]

            Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
            with open(output_txt, 'w', encoding='utf-8') as f:
                _json.dump(txt_data, f, ensure_ascii=False, indent=2)

            # 結果サマリー
            total_lanes = sum(len(v) for v in result.values())
            type_summary = ", ".join(
                f"{k}: {len(v)}本" for k, v in sorted(result.items())
            )
            self._status.showMessage(
                f"{t('menu_export_pkl')} 完了: {total_lanes}本 → {output_pkl}"
            )
            QMessageBox.information(
                self, t("menu_export_pkl"),
                f"PKL出力完了: {total_lanes}本\n\n"
                f"種別: {type_summary}\n\n"
                f"pkl: {output_pkl}\n"
                f"txt: {output_txt}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"),
                f"pkl出力に失敗しました:\n{e}"
            )
            import traceback
            traceback.print_exc()


    # ==================================================================
    # 7.3  CVAT 
    # ==================================================================

    def _cvat_connect(self) -> None:
        
        if CvatLoginDialog is None:
            QMessageBox.warning(
                self, t("dialog_warning"), t("status_no_cvat")
            )
            return

        #   
        config = self._cvat_config or (CvatConfig() if CvatConfig else None)
        if config is None:
            return

        login_dlg = CvatLoginDialog(self, config=config)
        if login_dlg.exec_() != QDialog.Accepted:
            return

        self._cvat_client = login_dlg.client
        self._cvat_config = login_dlg.config
        self._status.showMessage(
            f"{t('cvat_login_success')} {login_dlg.config.host}"
        )

        #   (LaneLine ｧ CvatDialog) 
        self._open_cvat_task_dialog()

    def _open_cvat_task_dialog(self) -> None:
        
        if CvatDialog is None or self._cvat_client is None:
            return

        cvat_dlg = CvatDialog(
            self,
            config=self._cvat_config,
            client=self._cvat_client,
            mode=getattr(self, '_cvat_mode', 'annotator'),
        )
        cvat_dlg.task_selected.connect(self._on_task_selected)
        cvat_dlg.exec_()

    def _on_task_selected(
        self, task_id: int, task_name: str, project_name: str
    ) -> None:
        
        # CvatDialog  client 
        sender = self.sender()
        if sender is not None and hasattr(sender, 'client') and sender.client:
            self._cvat_client = sender.client
        if sender is not None and hasattr(sender, 'config'):
            self._cvat_config = sender.config

        if self._cvat_client is None:
            return

        self._cvat_task_id = task_id
        self._cvat_project_name = project_name

        # ID 
        try:
            task_info = self._cvat_client.get_task(task_id)
            self._cvat_project_id = task_info.project_id
            if self._cvat_project_id:
                self._load_project_tasks(self._cvat_project_id)
        except Exception as e:
            print(f"[MainWindow] Task info error: {e}")
            traceback.print_exc()

    def _cvat_register_tasks(self) -> None:
        
        if self._cvat_client is None:
            QMessageBox.warning(
                self, t("dialog_warning"), t("status_no_cvat")
            )
            return
        if register_tasks is None:
            QMessageBox.warning(
                self, t("dialog_warning"), "register_cvat_tasks module not available"
            )
            return

        #   
        dlg = QDialog(self)
        dlg.setWindowTitle(t("dialog_register_title"))
        dlg.setFixedSize(520, 280)
        dlg.setStyleSheet(
            "QDialog { background: #1e1e1e; color: #ccc; }"
            "QLabel { color: #ccc; }"
            "QLineEdit { background: #2a2a2a; color: #ccc; border: 1px solid #555;"
            " border-radius: 3px; padding: 6px; }"
            "QPushButton { background: #2a6ebb; color: white; border: none;"
            " border-radius: 4px; padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #3a7ecc; }"
        )

        layout = QVBoxLayout()
        form = QGridLayout()
        form.setSpacing(8)

        # 
        form.addWidget(QLabel(t("dialog_register_project_name")), 0, 0)
        project_edit = QLineEdit()
        form.addWidget(project_edit, 0, 1, 1, 2)

        # BEV 
        form.addWidget(QLabel(t("dialog_register_image")), 1, 0)
        image_edit = QLineEdit()
        form.addWidget(image_edit, 1, 1)
        btn_image = QPushButton(t("dialog_register_browse"))
        btn_image.clicked.connect(
            lambda: self._browse_file(image_edit, t("file_filter_image"))
        )
        form.addWidget(btn_image, 1, 2)

        # 
        form.addWidget(QLabel(t("dialog_register_lane")), 2, 0)
        lane_edit = QLineEdit()
        form.addWidget(lane_edit, 2, 1)
        btn_lane = QPushButton(t("dialog_register_browse"))
        btn_lane.clicked.connect(
            lambda: self._browse_file(lane_edit, t("file_filter_lane"))
        )
        form.addWidget(btn_lane, 2, 2)

        #  PCD
        form.addWidget(QLabel(t("dialog_register_trajectory")), 3, 0)
        traj_edit = QLineEdit()
        form.addWidget(traj_edit, 3, 1)
        btn_traj = QPushButton(t("dialog_register_browse"))
        btn_traj.clicked.connect(
            lambda: self._browse_file(traj_edit, t("file_filter_pcd"))
        )
        form.addWidget(btn_traj, 3, 2)

        layout.addLayout(form)
        layout.addStretch()

        # 
        btn_layout = QHBoxLayout()
        btn_register = QPushButton(t("btn_register"))
        btn_cancel = QPushButton(t("btn_cancel"))
        btn_cancel.setStyleSheet(
            "QPushButton { background: #333; color: #888; border: 1px solid #555; }"
        )
        btn_layout.addStretch()
        btn_layout.addWidget(btn_register)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        dlg.setLayout(layout)

        btn_cancel.clicked.connect(dlg.reject)

        def _do_register():
            pname = project_edit.text().strip()
            img = image_edit.text().strip()
            lane = lane_edit.text().strip()
            traj = traj_edit.text().strip()
            if not pname or not img or not lane or not traj:
                QMessageBox.warning(
                    dlg, t("dialog_warning"), "All fields are required."
                )
                return
            try:
                self._status.showMessage(t("status_loading"))
                QApplication.processEvents()
                register_tasks(
                    client=self._cvat_client,
                    project_name=pname,
                    image_path=img,
                    lane_path=lane,
                    trajectory_path=traj,
                )
                self._status.showMessage(t("status_task_registered"))
                QMessageBox.information(
                    dlg,
                    t("dialog_register_success"),
                    t("dialog_register_success_msg"),
                )
                dlg.accept()
            except Exception as e:
                QMessageBox.critical(
                    dlg,
                    t("dialog_register_fail"),
                    f"{t('dialog_register_fail_msg')} {e}",
                )
                self._status.showMessage(t("status_ready"))

        btn_register.clicked.connect(_do_register)
        dlg.exec_()

    @staticmethod
    def _browse_file(line_edit: QLineEdit, file_filter: str) -> None:
        
        filepath, _ = QFileDialog.getOpenFileName(
            None, "", "", file_filter
        )
        if filepath:
            line_edit.setText(filepath)

    def _cvat_upload(self) -> None:
        
        if self._cvat_client is None or self._cvat_task_id is None:
            QMessageBox.warning(
                self, t("dialog_warning"), t("status_no_cvat")
            )
            return
        if BevCvatConverter is None or self._panel is None or self._gl is None:
            return

        lanes = self._panel._lanes
        if not lanes:
            QMessageBox.information(
                self, t("dialog_info"), t("err_no_annotations")
            )
            return

        # 
        reply = QMessageBox.question(
            self,
            t("dialog_upload_confirm_title"),
            t("dialog_upload_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._status.showMessage(t("status_uploading"))
            QApplication.processEvents()

            il = self._gl._bev_image
            converter = BevCvatConverter(il)

            #  ID  spec_id 
            label_id_map, attr_spec_map = self._resolve_label_maps(
                self._cvat_task_id
            )

            # source=="prelabel" のレコードはCVATアップロード対象から除外する
            lanes_to_upload = [r for r in lanes if r.get("source") != "prelabel"]
            n_excluded = len(lanes) - len(lanes_to_upload)
            if n_excluded:
                print(f"[MainWindow] CVATアップロード: prelabelレコード {n_excluded} 本を除外")

            cvat_data = converter.lanes_to_cvat(
                lanes_to_upload,
                label_id_map=label_id_map,
                attr_spec_map=attr_spec_map,
            )

            self._cvat_client.put_task_annotations(
                self._cvat_task_id, cvat_data
            )

            self._has_unsaved_changes = False
            self._status.showMessage(t("status_upload_done"))
            QMessageBox.information(
                self, t("dialog_info"), t("status_upload_done")
            )
        except Exception as e:
            self._status.showMessage(f"{t('status_upload_fail')} {e}")
            QMessageBox.critical(
                self,
                t("dialog_error"),
                f"{t('err_cvat_upload')} {e}",
            )

    def _resolve_label_maps(
        self, task_id: int
    ) -> tuple[dict, dict]:
        
        label_id_map: dict[str, int] = {}
        attr_spec_map: dict[tuple[str, str], int] = {}

        if self._cvat_client is None:
            return label_id_map, attr_spec_map

        try:
            # 
            task_info = self._cvat_client.get_task(task_id)
            # ｰ
            project_id = task_info.project_id
            if project_id:
                project = self._cvat_client.get_project(project_id)
                labels = project.labels if hasattr(project, 'labels') else []
            else:
                labels = task_info.labels if hasattr(task_info, 'labels') else []

            for label in labels:
                lname = label.get("name", "") if isinstance(label, dict) else getattr(label, "name", "")
                lid = label.get("id", 0) if isinstance(label, dict) else getattr(label, "id", 0)
                label_id_map[lname] = lid

                attrs = label.get("attributes", []) if isinstance(label, dict) else getattr(label, "attributes", [])
                for attr in attrs:
                    aname = attr.get("name", "") if isinstance(attr, dict) else getattr(attr, "name", "")
                    aid = attr.get("id", 0) if isinstance(attr, dict) else getattr(attr, "id", 0)
                    attr_spec_map[(lname, aname)] = aid

        except Exception as e:
            print(f"[MainWindow] Label resolution error: {e}")
            traceback.print_exc()

        return label_id_map, attr_spec_map

    def _cvat_complete_job(self) -> None:
        
        if self._cvat_client is None or self._cvat_task_id is None:
            QMessageBox.warning(
                self, t("dialog_warning"), t("status_no_cvat")
            )
            return

        # 
        reply = QMessageBox.question(
            self,
            t("dialog_complete_confirm_title"),
            t("dialog_complete_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            # 
            self._cvat_upload_silent()

            # 
            jobs = self._cvat_client.list_jobs(self._cvat_task_id)
            for job in jobs:
                job_id = job.id if hasattr(job, 'id') else job.get('id', 0)
                self._cvat_client.update_job_stage(job_id, "acceptance")
                self._cvat_client.update_job_state(job_id, "completed")

            self._status.showMessage(t("status_job_completed"))
            QMessageBox.information(
                self, t("dialog_info"), t("status_job_completed")
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                t("dialog_error"),
                f"{t('err_cvat_upload')} {e}",
            )

    def _cvat_upload_silent(self) -> None:
        
        if (self._cvat_client is None or self._cvat_task_id is None
                or BevCvatConverter is None or self._panel is None
                or self._gl is None):
            return

        lanes = self._panel._lanes
        if not lanes:
            return

        il = self._gl._bev_image
        converter = BevCvatConverter(il)

        label_id_map, attr_spec_map = self._resolve_label_maps(
            self._cvat_task_id
        )

        cvat_data = converter.lanes_to_cvat(
            lanes,
            label_id_map=label_id_map,
            attr_spec_map=attr_spec_map,
        )

        self._cvat_client.put_task_annotations(
            self._cvat_task_id, cvat_data
        )
        self._has_unsaved_changes = False

    def _load_project_tasks(self, project_id: int) -> None:
        
        if self._cvat_client is None:
            return

        try:
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()

            # 
            project_name = self._cvat_project_name
            if not project_name:
                try:
                    project = self._cvat_client.get_project(project_id)
                    project_name = project.name
                    self._cvat_project_name = project_name
                except Exception:
                    pass

            # (_3d )
            base_name = project_name
            if base_name.endswith("_3d"):
                base_name = base_name[:-3]

            # 2D / 3D 
            tasks_2d = []
            tasks_3d = []
            all_projects = self._cvat_client.list_projects()
            for p in all_projects:
                if p.name == base_name:
                    tasks_2d = self._cvat_client.list_tasks(project_id=p.id)
                    self._cvat_project_id = p.id
                elif p.name == f"{base_name}_3d":
                    tasks_3d = self._cvat_client.list_tasks(project_id=p.id)

            #  (2D + 3D )
            if identify_tasks_dual is not None:
                identified = identify_tasks_dual(tasks_2d, tasks_3d, base_name)
            else:
                # : 
                identified = self._identify_tasks(tasks_2d + tasks_3d, base_name)

            image_task = identified.get("image")
            lane_task = identified.get("lane")
            traj_task = identified.get("trajectory")

            # 
            missing = [k for k, v in identified.items() if v is None]
            if missing:
                self._status.showMessage(
                    f"{t('warn_task_partial')} ({', '.join(missing)})"
                )

            #   
            if image_task is not None:
                task_id = image_task.id if hasattr(image_task, 'id') else image_task
                self._cvat_task_id = task_id
                print(f"[MainWindow] Loading image task ID={task_id}")
                #  bug_tracker  BEV 
                bev_meta = self._get_project_bev_meta(self._cvat_project_id)
                print(f"[MainWindow] BEV meta result: {bev_meta}")
                # ｰ _meta.json 
                if bev_meta is None:
                    bev_meta = self._find_local_meta(task_id)
                self._load_cvat_image(task_id, bev_meta=bev_meta)
                # LaneLinePanel  _ground_z() 
                if self._panel is not None and self._gl is not None:
                    il = self._gl._bev_image
                    if il is not None and hasattr(il, 'ground_z'):
                        #  base_layer  ground_z 
                        from types import SimpleNamespace
                        dummy = SimpleNamespace(
                            xyz=np.array([[0, 0, il.ground_z]], dtype=np.float32),
                            loaded=True,
                            filepath=None,
                        )
                        self._panel._base_layer = dummy
            else:
                print("[MainWindow] No image task found")

            #   
            if lane_task is not None:
                task_id = lane_task.id if hasattr(lane_task, 'id') else lane_task
                self._load_cvat_lane(task_id)

            #   
            if traj_task is not None:
                task_id = traj_task.id if hasattr(traj_task, 'id') else traj_task
                self._load_cvat_trajectory(task_id)

            #   
            if image_task is not None:
                task_id = image_task.id if hasattr(image_task, 'id') else image_task
                self._load_cvat_annotations(task_id)

            self._status.showMessage(
                f"{t('cvat_task_loaded')} {project_name}"
            )

        except Exception as e:
            self._status.showMessage(
                f"{t('cvat_task_load_fail')} {e}"
            )
            traceback.print_exc()

    def _identify_tasks(self, tasks: list, project_name: str) -> dict:
        
        # : 
        result = {"image": None, "lane": None, "trajectory": None}
        suffixes = {"_image": "image", "_lane": "lane", "_trajectory": "trajectory"}
        for task in tasks:
            name = task.name if hasattr(task, 'name') else str(task)
            for suffix, key in suffixes.items():
                if name.endswith(suffix):
                    result[key] = task
                    break
        return result

    def _load_cvat_image(self, task_id: int, bev_meta: dict = None) -> None:
        
        if self._cvat_client is None or self._gl is None or DataLoader is None:
            return

        try:
            data = self._cvat_client.get_task_frame(task_id, 0)
            if data:
                # 
                names = self._cvat_client.get_task_frame_names(task_id)
                name = names[0] if names else "cvat_frame.png"
                image_layer = DataLoader.load_bev_image_from_bytes(
                    data, name, meta=bev_meta
                )
                self._gl.set_bev_image(image_layer)
                self._image_path = None  # CVAT 
        except Exception as e:
            print(f"[MainWindow] Image download error: {e}")
            traceback.print_exc()

    def _get_project_bev_meta(self, project_id: int) -> dict:
        
        if self._cvat_client is None or self._cvat_task_id is None:
            return None
        try:
            import json
            # CvatTask  bug_tracker  API 
            r = self._cvat_client._session.get(
                f"{self._cvat_client.base_url}/api/tasks/{self._cvat_task_id}",
                timeout=10,
            )
            if r.status_code == 200:
                bt = r.json().get("bug_tracker", "") or ""
                if "BEV_META::" in bt:
                    json_str = bt.split("BEV_META::", 1)[1]
                    meta = json.loads(json_str)
                    print(f"[MainWindow] BEV meta from CVAT: {meta}")
                    return meta
        except Exception as e:
            print(f"[MainWindow] BEV meta read error: {e}")
        return None

    def _find_local_meta(self, task_id: int) -> dict:
        
        if self._cvat_client is None:
            return None
        try:
            import json
            names = self._cvat_client.get_task_frame_names(task_id)
            if not names:
                return None
            frame_name = names[0]  # e.g. "RGBBEV_composite.png"
            stem = Path(frame_name).stem  # e.g. "RGBBEV_composite"

            # 
            search_dirs = []
            # 
            if self._image_path:
                search_dirs.append(Path(self._image_path).parent)

            for d in search_dirs:
                meta_path = d / f"{stem}_meta.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    print(f"[MainWindow] Local meta found: {meta_path}")
                    return meta
        except Exception as e:
            print(f"[MainWindow] Local meta search error: {e}")
        return None

    def _load_cvat_lane(self, task_id: int) -> None:
        
        if self._cvat_client is None or self._gl is None or DataLoader is None:
            return

        try:
            data = self._cvat_client.get_task_frame(task_id, 0)
            if not data:
                return

            # PCD VAT 3D 
            # PCD  x y z intensity label 5
            points_full = DataLoader.parse_pcd_full(data)
            if points_full is None or len(points_full) == 0:
                # :  parse  xyz 
                points = DataLoader.parse_pcd(data)
                if len(points) > 0:
                    self._gl.set_lane_points(points, None)
                return

            # xyz + cluster_id  GL 
            xyz = points_full[:, :3]
            cluster_ids = points_full[:, 4] if points_full.shape[1] > 4 else points_full[:, 3]
            colors = DataLoader.generate_cluster_colors(cluster_ids)
            self._gl.set_lane_points(xyz, colors)

            #  cluster.txt 
            self._save_lane_as_cluster_file(points_full, cluster_ids)

        except Exception as e:
            print(f"[MainWindow] Lane download error: {e}")
            traceback.print_exc()

    def _save_lane_as_cluster_file(self, points: "np.ndarray",
                                    cluster_ids: "np.ndarray") -> None:
        pass  # implementation removed due to encoding issue


    def _load_cvat_trajectory(self, task_id: int) -> None:
        """CVAT から軌跡 PCD を読み込んで GL に反映する"""
        if self._cvat_client is None or self._gl is None or DataLoader is None:
            return
        try:
            data = self._cvat_client.get_task_frame(task_id, 0)
            if data:
                points = DataLoader.parse_pcd(data)
                self._gl.set_trajectory_points(points)
        except Exception as e:
            print(f"[MainWindow] Trajectory download error: {e}")
            traceback.print_exc()

    def _load_cvat_annotations(self, task_id: int) -> None:
        """CVAT からアノテーションを読み込んで LaneLinePanel に反映する"""
        if (self._cvat_client is None or BevCvatConverter is None
                or self._panel is None or self._gl is None):
            return
        try:
            annotations = self._cvat_client.get_task_annotations(task_id)
            shapes = annotations.get("shapes", [])
            if not shapes:
                return
            il = self._gl._bev_image
            converter = BevCvatConverter(il)
            label_map, spec_id_map = self._build_reverse_label_maps(task_id)
            records = converter.cvat_to_lanes(
                annotations, label_map=label_map, spec_id_map=spec_id_map)
            if records:
                self._panel._lanes = records
                self._gl.draw_lane_data = records
                if hasattr(self._panel, '_refresh_table'):
                    self._panel._refresh_table()
                if hasattr(self._panel, '_rebuild_all_poly3d'):
                    self._panel._rebuild_all_poly3d()
                self._gl.update()
                self._status.showMessage(
                    f"{t('cvat_ann_loaded')} {len(records)} {t('cvat_shapes')}")
        except Exception as e:
            print(f"[MainWindow] Annotation download error: {e}")
            traceback.print_exc()

    def _build_reverse_label_maps(self, task_id: int) -> tuple:
        """CVAT の label/attr ID を内部名にマップする逆引き辞書を構築する"""
        label_map: dict = {}
        spec_id_map: dict = {}
        if self._cvat_client is None:
            return label_map, spec_id_map
        try:
            task_info = self._cvat_client.get_task(task_id)
            project_id = task_info.project_id
            if project_id:
                project = self._cvat_client.get_project(project_id)
                labels = project.labels if hasattr(project, 'labels') else []
            else:
                labels = task_info.labels if hasattr(task_info, 'labels') else []
            for label in labels:
                lname = label.get("name", "") if isinstance(label, dict) else getattr(label, "name", "")
                lid = label.get("id", 0) if isinstance(label, dict) else getattr(label, "id", 0)
                label_map[lid] = lname
                attrs = label.get("attributes", []) if isinstance(label, dict) else getattr(label, "attributes", [])
                for attr in attrs:
                    aname = attr.get("name", "") if isinstance(attr, dict) else getattr(attr, "name", "")
                    aid = attr.get("id", 0) if isinstance(attr, dict) else getattr(attr, "id", 0)
                    spec_id_map[aid] = aname
        except Exception as e:
            print(f"[MainWindow] Reverse label map error: {e}")
            traceback.print_exc()
        return label_map, spec_id_map

    # ==================================================================
    # 品質チェック
    # ==================================================================

    def _run_quality_check(self) -> None:
        """品質チェックを実行して QualityPanel と GL に結果を反映する"""
        if QualityChecker is None or AttributeAssigner is None or PriorityClassifier is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_quality_module"))
            return
        if self._panel is None or self._gl is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_panel_gl"))
            return
        lanes = self._panel._lanes
        if not lanes:
            QMessageBox.information(self, t("dialog_info"), t("info_no_annotations"))
            return
        trajectory = None
        if self._gl._trajectory_points is not None and len(self._gl._trajectory_points) >= 2:
            trajectory = self._gl._trajectory_points
        image_layer = self._gl._bev_image
        try:
            self._status.showMessage(t("status_quality_running"))
            QApplication.processEvents()
            progress = __import__('PyQt5.QtWidgets', fromlist=['QProgressBar']).QProgressBar()
            progress.setMaximum(100)
            self._status.addWidget(progress)
            QApplication.processEvents()
            def progress_callback(current, total):
                pct = int(current / max(total, 1) * 100)
                progress.setValue(pct)
                QApplication.processEvents()
            config = QualityConfig.load()
            checker = QualityChecker(config)
            check_results_map = checker.run_all_checks(lanes, image_layer, trajectory, progress_callback)
            assigner = AttributeAssigner()
            auto_attributes_map = assigner.assign_all(lanes, trajectory)
            classifier = PriorityClassifier(config.priority_thresholds)
            quality_reports = classifier.classify_all(check_results_map, auto_attributes_map)
            self._status.removeWidget(progress)
            progress.deleteLater()
            self._quality_reports = quality_reports
            if self._quality_panel is not None:
                self._quality_panel.set_reports(quality_reports)
                self._quality_panel.show()
            if hasattr(self._gl, 'set_priority_highlights'):
                self._gl.set_priority_highlights(quality_reports)
            self._status.showMessage(t("status_quality_done"))
            self._connect_quality_signals()
        except Exception as e:
            try:
                self._status.removeWidget(progress)
                progress.deleteLater()
            except Exception:
                pass
            QMessageBox.critical(self, t("dialog_error"), t("err_quality_check") + str(e))
            self._status.showMessage(t("status_quality_fail"))

    def _connect_quality_signals(self) -> None:
        """QualityPanel のシグナルを MainWindow/GLWidget に接続する"""
        if self._quality_panel is None:
            return
        try:
            self._quality_panel.lane_selected.disconnect(self._on_quality_lane_selected)
        except (TypeError, RuntimeError):
            pass
        try:
            self._quality_panel.issue_selected.disconnect(self._on_quality_issue_selected)
        except (TypeError, RuntimeError):
            pass
        self._quality_panel.lane_selected.connect(self._on_quality_lane_selected)
        self._quality_panel.issue_selected.connect(self._on_quality_issue_selected)

    def _on_quality_lane_selected(self, lane_id: str) -> None:
        """QualityPanel で lane が選択されたときの処理"""
        if self._panel is None or self._gl is None:
            return
        for i, rec in enumerate(self._panel._lanes):
            if rec.get("id") == lane_id:
                if hasattr(self._panel, '_tbl'):
                    self._panel._tbl.selectRow(i)
                poly = rec.get("poly3d")
                if poly is not None:
                    self._gl.draw_lane_highlight = poly
                    self._gl.update()
                break

    def _on_quality_issue_selected(self, lane_id: str, check_result_index: int) -> None:
        """QualityPanel で issue が選択されたときの処理"""
        if self._gl is None:
            return
        if hasattr(self._gl, 'set_issue_markers'):
            self._gl.set_issue_markers(lane_id, check_result_index)

    def _show_quality_settings(self) -> None:
        """品質チェック設定ダイアログを表示する"""
        QMessageBox.information(
            self, t("dlg_quality_settings_title"), t("dlg_quality_settings_msg"))

    def _export_quality_report(self) -> None:
        """品質チェック結果を JSON ファイルにエクスポートする"""
        if not hasattr(self, '_quality_reports') or not self._quality_reports:
            QMessageBox.information(self, t("dialog_info"), t("info_no_quality_result"))
            return
        default_name = "quality_report.json"
        if self._image_path:
            stem = Path(self._image_path).stem
            default_name = f"{stem}_quality_report.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self, t("dlg_report_export_title"), default_name, t("file_filter_json"))
        if not filepath:
            return
        try:
            import json
            from datetime import datetime
            reports_data = []
            for lane_id, report in self._quality_reports.items():
                report_dict = {
                    "lane_id": report.lane_id, "class": "",
                    "review_priority": report.review_priority,
                    "confidence_class": report.confidence_class,
                    "check_results": [
                        {"check_type": cr.check_type, "severity": cr.severity,
                         "message": cr.message, "details": cr.details}
                        for cr in report.check_results],
                    "auto_attributes": report.auto_attributes,
                }
                if self._panel is not None:
                    for rec in self._panel._lanes:
                        if rec.get("id") == lane_id:
                            report_dict["class"] = rec.get("class", "")
                            break
                reports_data.append(report_dict)
            summary = self._quality_panel.get_summary() if self._quality_panel else {}
            export_data = {
                "summary": {
                    "total_annotations": summary.get("total", 0),
                    "priority_counts": {"A": summary.get("A", 0), "B": summary.get("B", 0),
                                        "C": summary.get("C", 0), "D": summary.get("D", 0)},
                    "check_datetime": datetime.now().isoformat(),
                },
                "reports": reports_data,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self._status.showMessage(t("status_report_exported") + filepath)
            QMessageBox.information(self, t("dlg_report_success_title"),
                                    t("dlg_report_success_msg") + filepath)
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_report_export") + str(e))

    # ==================================================================
    # ID 自動付与
    # ==================================================================

    def _get_lane_id_assigner(self) -> "LaneIdAssigner":
        """LaneIdAssigner を生成して返す。mapping_pose.txt 優先で軌跡を渡す"""
        if LaneIdAssigner is None:
            return None
        trajectory = None
        if self._mapping_poses:
            trajectory = np.array(
                [[p['x'], p['y'], p['z']] for p in self._mapping_poses],
                dtype=np.float32)
        elif self._gl is not None and hasattr(self._gl, '_trajectory_points'):
            traj = self._gl._trajectory_points
            if traj is not None and len(traj) >= 2:
                trajectory = traj
        return LaneIdAssigner(trajectory=trajectory)

    def _auto_assign_all(self) -> None:
        """N / Y / boundary / split-merge の ID を一括自動付与する"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        if not lanes:
            QMessageBox.information(self, t("dialog_info"), t("info_no_annotations"))
            return
        if self._gl is None or (not self._mapping_poses and self._gl._trajectory_points is None):
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_trajectory"))
            return
        self._status.showMessage(t("status_id_assigning"))
        QApplication.processEvents()
        current_step = ""
        try:
            current_step = "N 自動付与（全クラス共通通し番号）"
            stop_lines = [r for r in lanes if r.get("class") == "stop_line"]
            zebra_lines = [r for r in lanes if r.get("class") == "zebra_line"]
            # 全クラス共通のN通し番号を一括付与（lane_line + boundary系）
            assigner.assign_all_n_values(lanes, stop_lines, zebra_lines)
            n_count = sum(1 for r in lanes if r.get("lane_id_n"))
            current_step = "Y 自動付与"
            split_merge_points = [r for r in lanes
                                   if r.get("class") in ("split_point", "merge_point")]
            assigner.assign_y_values(lanes, split_merge_points)
            current_step = "分合流点候補"
            boundary_classes = ["curb_boundary", "fence_boundary",
                                 "TrafficCone_boundary", "WaterSafety_boundary", "boundary"]
            boundary_count = sum(1 for r in lanes
                                  if r.get("class") in boundary_classes and r.get("boundary_id"))
            split_merge_count = 0
            for smp in split_merge_points:
                vertices = smp.get("vertices", [])
                if not vertices:
                    continue
                candidates = assigner.find_nearby_lane_endpoints(vertices[0], lanes)
                if candidates["start_line_ids"]:
                    smp["start_line_ids"] = ",".join(candidates["start_line_ids"])
                    split_merge_count += 1
                if candidates["end_line_ids"]:
                    smp["end_line_ids"] = ",".join(candidates["end_line_ids"])
                    split_merge_count += 1
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if self._gl is not None:
                self._gl.update()
            self._has_unsaved_changes = True
            self._status.showMessage(
                t("status_id_assign_done").format(n_count, boundary_count, split_merge_count))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"),
                                  t("err_auto_assign_n") + f" (step:{current_step}) " + str(e))
            self._status.showMessage(t("status_id_assign_fail"))

    def _renumber_ids(self) -> None:
        """全アノテーションの ID を振り直す"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        if not lanes:
            QMessageBox.information(self, t("dialog_info"), t("info_no_annotations"))
            return
        if self._gl is None or (not self._mapping_poses and self._gl._trajectory_points is None):
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_trajectory"))
            return
        reply = QMessageBox.question(
            self, t("dlg_renumber_confirm_title"), t("dlg_renumber_confirm_msg"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            self._status.showMessage(t("status_renumber_done") + "...")
            QApplication.processEvents()
            assigner.renumber_all(lanes)
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if self._gl is not None:
                self._gl.update()
            self._has_unsaved_changes = True
            self._status.showMessage(t("status_renumber_done"))
            QMessageBox.information(self, t("dlg_renumber_done_title"), t("dlg_renumber_done_msg"))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_renumber") + str(e))
            self._status.showMessage(t("status_renumber_fail"))

    def _auto_assign_n(self) -> None:
        """ID(N) を自動付与する"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        if not lanes:
            QMessageBox.information(self, t("dialog_info"), t("info_no_annotations"))
            return
        if self._gl is None or (not self._mapping_poses and self._gl._trajectory_points is None):
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_trajectory_n"))
            return
        try:
            self._status.showMessage(t("status_id_n_assigning"))
            QApplication.processEvents()
            stop_lines = [r for r in lanes if r.get("class") == "stop_line"]
            zebra_lines = [r for r in lanes if r.get("class") == "zebra_line"]
            # 全クラス共通のN通し番号を一括付与（lane_line + boundary系）
            assigner.assign_all_n_values(lanes, stop_lines, zebra_lines)
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if self._gl is not None:
                self._gl.update()
            self._has_unsaved_changes = True
            assigned_ll = sum(1 for r in lanes
                              if r.get("class") == "lane_line" and r.get("lane_id_n"))
            boundary_classes = ["curb_boundary", "fence_boundary",
                                 "TrafficCone_boundary", "WaterSafety_boundary", "boundary"]
            assigned_bnd = sum(1 for r in lanes
                               if r.get("class") in boundary_classes and r.get("lane_id_n"))
            self._status.showMessage(
                t("status_id_n_done_with_boundary").format(assigned_ll, assigned_bnd)
            )
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_auto_assign_n") + str(e))
            self._status.showMessage(t("status_auto_assign_n_fail"))

    def _auto_assign_x(self) -> None:
        """X(横) を自動付与する"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        if not lanes:
            return
        if self._gl is None or (not self._mapping_poses and self._gl._trajectory_points is None):
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_trajectory_n"))
            return
        try:
            assigner.assign_x_values(lanes)
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if self._gl is not None:
                self._gl.update()
            self._has_unsaved_changes = True
            self._status.showMessage(t("status_x_done"))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_auto_assign_n") + str(e))

    def _auto_assign_y(self) -> None:
        """Y(縦) を自動付与する"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        if not lanes:
            return
        if self._gl is None or (not self._mapping_poses and self._gl._trajectory_points is None):
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_trajectory_n"))
            return
        try:
            split_merge_points = [r for r in lanes
                                   if r.get("class") in ("split_point", "merge_point")]
            assigner.assign_y_values(lanes, split_merge_points)
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if self._gl is not None:
                self._gl.update()
            self._has_unsaved_changes = True
            self._status.showMessage(t("status_y_done"))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_auto_assign_n") + str(e))

    def _auto_assign_boundary_ids(self) -> None:
        """境界クラスの boundary_id を自動付与する"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        if not lanes:
            return
        try:
            assigner.assign_boundary_ids(lanes)
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            self._has_unsaved_changes = True
            boundary_classes = ["curb_boundary", "fence_boundary",
                                 "TrafficCone_boundary", "WaterSafety_boundary", "boundary"]
            assigned_count = sum(1 for r in lanes
                                  if r.get("class") in boundary_classes and r.get("boundary_id"))
            self._status.showMessage(t("status_boundary_done").format(assigned_count))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_auto_assign_n") + str(e))

    def _suggest_split_merge_ids(self) -> None:
        """分合流点候補を検出して start/end_line_ids を提案する"""
        assigner = self._get_lane_id_assigner()
        if assigner is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_assigner"))
            return
        if self._panel is None:
            return
        lanes = self._panel._lanes
        split_merge_records = [r for r in lanes
                                if r.get("class") in ("split_point", "merge_point")]
        if not split_merge_records:
            QMessageBox.information(self, t("dialog_info"), t("info_no_annotations"))
            return
        if self._gl is None or (not self._mapping_poses and self._gl._trajectory_points is None):
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_trajectory"))
            return
        try:
            updated_count = 0
            for smp in split_merge_records:
                vertices = smp.get("vertices", [])
                if not vertices:
                    continue
                candidates = assigner.find_nearby_lane_endpoints(vertices[0], lanes)
                if candidates["start_line_ids"] and not smp.get("start_line_ids"):
                    smp["start_line_ids"] = ",".join(candidates["start_line_ids"])
                    updated_count += 1
                if candidates["end_line_ids"] and not smp.get("end_line_ids"):
                    smp["end_line_ids"] = ",".join(candidates["end_line_ids"])
                    updated_count += 1
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if updated_count > 0:
                self._has_unsaved_changes = True
            self._status.showMessage(t("status_split_merge_done").format(updated_count))
            QMessageBox.information(
                self, t("dlg_renumber_done_title"),
                t("dlg_split_merge_done_msg").format(updated_count))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), t("err_auto_assign_n") + str(e))

    # ==================================================================
    # プロジェクト管理
    # ==================================================================

    def _current_project_data(self):
        """現在の作業状態を ProjectData として返す"""
        from core.project_manager import ProjectData
        anno_path = None
        if self._image_path:
            p = Path(self._image_path)
            candidate = p.parent / f"{p.stem}_annotations.xml"
            if candidate.exists():
                anno_path = str(candidate)
        return ProjectData(
            image=self._image_path,
            lane=getattr(self, "_lane_path", None),
            trajectory=getattr(self, "_trajectory_path", None),
            ref_pcd=(self._pcd_z_resolver.pcd_path
                     if self._pcd_z_resolver and self._pcd_z_resolver.is_loaded else None),
            annotation=anno_path,
        )

    def _project_new(self) -> None:
        """プロジェクトをクリアして新規状態にする"""
        if self._has_unsaved_changes:
            reply = QMessageBox.question(
                self, t("dialog_confirm"), t("dialog_exit_msg_local"),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self._save_annotations()
        if self._gl is not None:
            self._gl.set_bev_image(None)
            self._gl.set_lane_points(None, None)
            self._gl.set_trajectory_points(None)
            self._gl.update()
        if self._panel is not None:
            self._panel._lanes = []
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
        self._image_path = None
        self._lane_path = None
        self._trajectory_path = None
        self._pcd_z_resolver = None
        self._has_unsaved_changes = False
        self._status.showMessage(t("status_ready"))

    def _project_save(self) -> None:
        """現在のプロジェクトをファイルに保存する。

        保存先: {ルートフォルダ}/{ルートフォルダ名}.blap
        ルートフォルダが未設定の場合はダイアログで保存先を選択する。
        アノテーションデータも同時に XML として保存する。
        """
        from core.project_manager import save_project

        # ── ① アノテーションを先に保存（XML）──────────────────────
        # XML を保存してから ProjectData に含める
        if (self._panel is not None and self._panel._lanes
                and self._gl is not None and self._gl._bev_image is not None):
            try:
                self._save_local_annotations()
            except Exception as e:
                print(f"[ProjectSave] アノテーション保存失敗: {e}")

        project = self._current_project_data()
        if project.is_empty():
            QMessageBox.information(self, t("menu_project_save"), t("info_no_annotations"))
            return

        # ── ② 保存先を自動決定 ────────────────────────────────────
        if self._root_folder_path:
            root = Path(self._root_folder_path)
            filepath = str(root / f"{root.name}.blap")
        else:
            # ルートフォルダ未設定時はダイアログ
            default_dir = ""
            if self._image_path:
                p = Path(self._image_path)
                default_dir = str(p.parent / p.stem)
            filepath, _ = QFileDialog.getSaveFileName(
                self, t("menu_project_save"),
                default_dir + ".blap" if default_dir else "",
                t("file_filter_project"))
            if not filepath:
                return

        try:
            save_project(project, filepath)
            self._status.showMessage(f"{t('status_project_saved')} {Path(filepath).name}")
            self._rebuild_recent_menu()
            QMessageBox.information(
                self, t("menu_project_save"),
                t("info_project_saved_with_anno").format(filepath, project.annotation)
                if project.annotation
                else t("info_project_saved").format(filepath)
            )
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_project_save')} {e}")

    def _project_open(self, filepath: str = "") -> None:
        """プロジェクトファイルを開いてデータを読み込む"""
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(
                self, t("menu_project_open"), "", t("file_filter_project"))
        if not filepath:
            return
        from core.project_manager import load_project
        try:
            project = load_project(filepath)
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_project_load')} {e}")
            return
        from core.project_manager import KEY_LABELS_JA
        missing = []
        for key, path in project.loaded_files().items():
            if path and not Path(path).exists():
                label = KEY_LABELS_JA.get(key, key)
                missing.append(f"  {label}: {path}")
        if missing:
            msg = t("dialog_warning") + "\n\n" + "\n".join(missing) + "\n\n不足ファイルはスキップします。"
            QMessageBox.warning(self, t("menu_project_open"), msg)
        if project.image and Path(project.image).exists():
            self._load_image_file(project.image)
        if project.lane and Path(project.lane).exists():
            self._load_lane_file(project.lane)
        if project.trajectory and Path(project.trajectory).exists():
            self._load_trajectory_file(project.trajectory)
        if project.ref_pcd and Path(project.ref_pcd).exists():
            self._load_ref_pcd_file(project.ref_pcd)
        if project.annotation and Path(project.annotation).exists():
            self._load_annotation_file(project.annotation)
        self._status.showMessage(f"{t('status_project_loaded')} {Path(filepath).name}")
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        """最近使ったプロジェクトのメニューを再構築する"""
        from core.project_manager import get_recent_projects, clear_recent_projects
        self._menu_recent.clear()
        recent = get_recent_projects()
        if not recent:
            act = QAction(t("menu_project_recent_empty"), self)
            act.setEnabled(False)
            self._menu_recent.addAction(act)
        else:
            for path in recent:
                label = Path(path).name
                act = QAction(label, self)
                act.setToolTip(path)
                act.triggered.connect(lambda checked, p=path: self._project_open(p))
                self._menu_recent.addAction(act)
            self._menu_recent.addSeparator()
            act_clear = QAction(t("menu_project_recent_clear"), self)
            act_clear.triggered.connect(lambda: (
                clear_recent_projects(), self._rebuild_recent_menu()))
            self._menu_recent.addAction(act_clear)

    # ==================================================================
    # ルートフォルダを開く
    # ==================================================================

    def _open_root_folder(self, folder_path: str = "") -> None:
        """プロジェクトフォルダ（ルート）を開き、配下のファイルを自動検索して読み込む。

        初回読み込み対象（即時ロード）:
          - BEV画像 (RGBBEV_rgb.png) + Z勾配オーバーレイ
          - 軌跡 (mapping_pose.txt)

        遅延ロード（自動生成ボタン押下時に読み込み）:
          - レーン点群 (*cluster*.txt) → _pending_lane_path に保存
          - サンプリングレーン (sampling_lane_global.*) → _sampling_lane_hint に保存

        ref_pcd は手動メニューから読み込む。
        アノテーションデータは自動読み込みしない（手作業データのため）。
        """
        if not folder_path:
            folder_path = QFileDialog.getExistingDirectory(
                self,
                t("menu_open_root_folder"),
                "",
            )
        if not folder_path:
            return

        from core.project_manager import open_project_folder, ProjectFolder
        try:
            # フォルダ内のファイルを検索
            folder = ProjectFolder(folder_path)
            if not folder.is_valid():
                QMessageBox.critical(
                    self, t("dialog_error"),
                    f"フォルダが見つかりません:\n{folder_path}"
                )
                return

            desc = folder.describe()
            # ルートフォルダパスを保存（出力先自動決定に使用）
            self._root_folder_path = str(Path(folder_path).resolve())

            # 未保存確認
            if self._has_unsaved_changes:
                save_reply = QMessageBox.question(
                    self, t("dialog_confirm"), t("dialog_exit_msg_local"),
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                )
                if save_reply == QMessageBox.Cancel:
                    return
                if save_reply == QMessageBox.Yes:
                    self._save_annotations()

            # Projection calibration overrides are project-local; source camera JSON is immutable.
            self._load_calibration_overrides_for_root()

            # ── 即時読み込み: BEV画像 + 軌跡 + 参照PCD ────────────────
            image_path = desc.get("image")
            if image_path and Path(image_path).exists():
                self._load_image_file(image_path)

            traj_path = desc.get("trajectory")
            if traj_path and Path(traj_path).exists():
                self._load_trajectory_file(traj_path)

            # 参照PCD（Z補完用）を自動読み込み
            ref_pcd_path = desc.get("ref_pcd")
            if ref_pcd_path and Path(ref_pcd_path).exists():
                self._load_ref_pcd_file(ref_pcd_path)
                if self._layer_panel is not None:
                    self._layer_panel.set_loaded("trajectory", True)
            else:
                # 見つからなかった場合はリセット（前回の残骸をクリア）
                self._pcd_z_resolver = None

            # ── 遅延ロード用パスを保存（自動生成ボタン押下時に使う） ──
            lane_path = desc.get("lane")
            self._pending_lane_path = lane_path if lane_path and Path(lane_path).exists() else None

            sampling_path = desc.get("sampling_lane")
            self._sampling_lane_hint = sampling_path if sampling_path and Path(sampling_path).exists() else None

            # レイヤーパネルのsampling_laneを「未読み込み」のまま維持
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("lane_points", False)

            status_msg = f"{t('status_folder_opened')} {Path(folder_path).name}"
            if self._pending_lane_path or self._sampling_lane_hint:
                status_msg += f"  ※{t('status_folder_opened_deferred')}"
            self._status.showMessage(status_msg)

            # フォルダ履歴に追加してからメニューを再構築
            from core.project_manager import _add_recent_folder
            _add_recent_folder(str(Path(folder_path).resolve()))
            self._rebuild_recent_folders_menu()

            # カメラパネルにデバッグ保存先を設定（ルートフォルダ配下）
            if self._camera_strip_panel is not None:
                debug_dir = str(Path(folder_path) / "projection_debug")
                self._camera_strip_panel._debug_save_dir = debug_dir
                print(f"[MainWindow] デバッグ保存先: {debug_dir}")

            # ── カメラ画像フォルダを自動検索して下段パネルに表示 ──────
            self._auto_load_camera_images(folder_path)

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"),
                f"フォルダを開く際にエラーが発生しました:\n{e}"
            )

    def _auto_load_camera_images(self, root_folder: str) -> None:
        """ルートフォルダからカメラ画像を自動検索して下段パネルに表示する。

        pcd_timestamp.txt が見つかれば FrameSyncTable を構築して
        フレームスライダーを有効化する。
        見つからなければレガシーモード（単純インデックス対応）で動作する。

        検索対象: cam-FW, cam-B, Fisheye-cam-L, Fisheye-cam-R
        """
        if self._camera_strip_panel is None:
            return
        if find_camera_dirs is None or build_camera_image_list is None:
            return

        try:
            self._status.showMessage("カメラ画像フォルダを検索中...")
            QApplication.processEvents()

            cam_dirs = find_camera_dirs(root_folder)
            if not cam_dirs:
                self._camera_strip_panel.clear()
                self._reset_frame_slider()
                self._status.showMessage(
                    f"カメラフォルダが見つかりませんでした: {Path(root_folder).name}"
                )
                return

            image_list = build_camera_image_list(cam_dirs)
            if not image_list:
                self._camera_strip_panel.clear()
                self._reset_frame_slider()
                return

            # ── pcd_timestamp.txt を探して FrameSyncTable を構築 ──
            pcd_ts_path = None
            if find_pcd_timestamp_path is not None:
                pcd_ts_path = find_pcd_timestamp_path(cam_dirs)

            if (pcd_ts_path and FrameSyncTable is not None
                    and self._mapping_poses):
                try:
                    table = FrameSyncTable.build(
                        pcd_timestamp_path=pcd_ts_path,
                        mapping_poses=self._mapping_poses,
                        camera_image_list=image_list,
                    )
                    self._frame_sync_table = table
                    self._camera_strip_panel.load_from_sync_table(table)
                    self._setup_frame_slider(len(table))
                    n_frames = len(table)
                    cam_names = list(cam_dirs.keys())
                    self._status.showMessage(
                        f"カメラ画像＋時刻同期: {', '.join(cam_names)}"
                        f"  {n_frames}フレーム / pcd_timestamp 使用"
                    )
                    return
                except Exception as e_sync:
                    print(f"[WARNING] FrameSyncTable 構築失敗、レガシーモードで継続: {e_sync}")

            # ── レガシーモード（pcd_timestamp なし or pose なし）──
            self._frame_sync_table = None
            self._camera_strip_panel.load_images(image_list)
            total = self._camera_strip_panel._total_frames
            self._setup_frame_slider(total)
            cam_names = list(cam_dirs.keys())
            self._status.showMessage(
                f"カメラ画像: {', '.join(cam_names)}  {total}フレーム"
            )

        except Exception as e:
            print(f"[WARNING] カメラ画像自動読み込み失敗: {e}")
            if self._camera_strip_panel is not None:
                self._camera_strip_panel.clear()
            self._reset_frame_slider()

    def _nearest_pose_frame_to_xy(
        self,
        x_world: float,
        y_world: float,
    ) -> Optional[int]:
        """指定Global XYに最も近いposeを持つフレームを返す。

        FrameSyncTableがある場合は、各画像/PCD同期フレームのego poseを使用する。
        レガシーモードではmapping_poseのインデックスをフレームとして使用する。
        poseを持たないフレームは候補から除外する。
        """
        frame_indices = []
        xy_points = []

        if self._frame_sync_table is not None:
            try:
                n = len(self._frame_sync_table)
            except Exception:
                n = 0

            if hasattr(self, "_frame_slider"):
                n = min(
                    n,
                    self._frame_slider.maximum() + 1,
                )

            for i in range(n):
                try:
                    info = self._frame_sync_table[i]
                    if not info.has_pose:
                        continue
                    x = float(info.ego_x)
                    y = float(info.ego_y)
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    frame_indices.append(i)
                    xy_points.append((x, y))
                except Exception:
                    continue

        elif self._mapping_poses:
            n = len(self._mapping_poses)
            if hasattr(self, "_frame_slider"):
                n = min(
                    n,
                    self._frame_slider.maximum() + 1,
                )

            for i in range(n):
                try:
                    pose = self._mapping_poses[i]
                    x = float(pose.get("x", 0.0))
                    y = float(pose.get("y", 0.0))
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    frame_indices.append(i)
                    xy_points.append((x, y))
                except Exception:
                    continue

        if not xy_points:
            return None

        xy = np.asarray(xy_points, dtype=np.float64)
        dx = xy[:, 0] - float(x_world)
        dy = xy[:, 1] - float(y_world)
        nearest_local_idx = int(np.argmin(dx * dx + dy * dy))
        return int(frame_indices[nearest_local_idx])

    def _on_ego_marker_drag(
        self,
        x_world: float,
        y_world: float,
        phase: str,
    ) -> None:
        """2Dビューの黄色い車両矩形ドラッグをフレーム移動へ変換する。

        ドラッグ先Global XYに最も近いmapping_pose/FrameSyncフレームへスナップし、
        マスターフレームスライダーを更新する。
        これにより下段カメラ画像・車両pose・画像投影が一括更新される。
        """
        if (
            not hasattr(self, "_frame_slider")
            or not self._frame_slider.isEnabled()
        ):
            return

        nearest = self._nearest_pose_frame_to_xy(
            x_world,
            y_world,
        )
        if nearest is None:
            return

        nearest = max(
            0,
            min(
                int(nearest),
                self._frame_slider.maximum(),
            ),
        )

        # ドラッグ開始時に同一フレームなら再投影しない。
        # move中も最寄りフレームが変わった時だけ重い画像/投影更新を行う。
        if phase == "start":
            self._ego_drag_last_frame = self._frame_slider.value()
            return

        if (
            nearest != self._frame_slider.value()
            and nearest != self._ego_drag_last_frame
        ):
            self._ego_drag_last_frame = nearest
            self._frame_slider.setValue(nearest)

        if phase == "end":
            self._ego_drag_last_frame = None

            # 車両追従モードの場合はドラッグ終了後だけ再センタリングする。
            # move中は BevGLWidget.set_ego_pose() が再センタリングを抑止するため、
            # 黄色い矩形を軌跡上で直接ドラッグできる。
            if (
                self._gl is not None
                and getattr(self._gl, "_follow_mode", "follow") == "follow"
            ):
                (
                    ego_x,
                    ego_y,
                    _ego_z,
                    ego_yaw,
                    _ego_roll,
                    _ego_pitch,
                ) = self._get_ego_pose_for_frame(
                    self._frame_slider.value()
                )
                if ego_x is not None and ego_y is not None:
                    self._gl.set_ego_pose(
                        float(ego_x),
                        float(ego_y),
                        float(ego_yaw) if ego_yaw is not None else 0.0,
                    )

    def _on_draw_position_follow_changed(self, state: int) -> None:
        """新規描画1点目・一覧選択時の追従ON/OFF。"""
        self._draw_position_follow_enabled = (
            state == Qt.Checked
        )

    def _is_draw_follow_enabled(self) -> bool:
        """2D中心追従チェック状態を返す。"""
        checkbox = getattr(
            self,
            "_chk_draw_position_follow",
            None,
        )
        if checkbox is not None:
            return checkbox.isChecked()
        return bool(
            getattr(
                self,
                "_draw_position_follow_enabled",
                True,
            )
        )

    def _set_frame_preserving_2d_view(self, frame_idx: int) -> None:
        """2Dビュー範囲を一切変えずにフレームだけ変更する。"""
        if (
            not hasattr(self, "_frame_slider")
            or not self._frame_slider.isEnabled()
        ):
            return

        frame_idx = max(
            0,
            min(
                int(frame_idx),
                self._frame_slider.maximum(),
            ),
        )
        if frame_idx == self._frame_slider.value():
            return

        gl = self._gl
        if gl is None:
            self._frame_slider.setValue(frame_idx)
            return

        center_copy = None
        try:
            center_copy = gl._center.copy()
        except Exception:
            center_copy = None

        view_state = (
            float(gl.pan_x),
            float(gl.pan_y),
            float(gl.zoom),
            float(gl.rot_x),
            float(gl.rot_z),
        )

        gl.set_follow_recenter_suppressed(True)
        try:
            self._frame_slider.setValue(frame_idx)
        finally:
            gl.set_follow_recenter_suppressed(False)
            (
                gl.pan_x,
                gl.pan_y,
                gl.zoom,
                gl.rot_x,
                gl.rot_z,
            ) = view_state
            if center_copy is not None:
                gl._center = center_copy
            gl.update()

    def _on_current_location_clicked(self) -> None:
        """「現在地」ボタン: 2Dビュー中心に最も近いフレームへ明示的に移動する。

        追従チェックON/OFFに関係なく実行する。
        2Dビューの表示範囲・Zoom・回転は維持し、
        黄色い自車矩形・下段カメラ画像・投影だけ更新する。
        """
        if self._gl is None or self._gl._view_mode != "2d":
            return

        center = self._gl.get_view_center_world()
        if center is None:
            return

        nearest = self._nearest_pose_frame_to_xy(
            float(center[0]),
            float(center[1]),
        )
        if nearest is None:
            return

        self._set_frame_preserving_2d_view(nearest)

    def _sync_frame_to_view_center(self) -> None:
        """追従ON時、現在の2Dビュー中心に最も近いフレームへ同期する。

        自動呼び出しはアノテーション一覧選択時のみ。
        2Dビューのパン/ズーム/Fit/Resetでは呼ばない。
        """
        if not self._is_draw_follow_enabled():
            return
        if self._gl is None or self._gl._view_mode != "2d":
            return

        center = self._gl.get_view_center_world()
        if center is None:
            return

        nearest = self._nearest_pose_frame_to_xy(
            float(center[0]),
            float(center[1]),
        )
        if nearest is None:
            return

        self._set_frame_preserving_2d_view(nearest)

    def _on_annotation_selected_first_point(
        self,
        x_world: float,
        y_world: float,
    ) -> None:
        """一覧選択時、先頭点を2Dビュー中心へ移動する。

        Zoom/回転は維持する。
        追従ONなら、その新しい2D中心に最も近いフレームへ
        黄色い矩形と下段カメラ画像も同期する。
        """
        if self._gl is None:
            return

        self._gl.center_view_on_world(
            float(x_world),
            float(y_world),
            notify=False,
        )

        if self._is_draw_follow_enabled():
            self._sync_frame_to_view_center()

    def _on_annotation_draw_position(
        self,
        x_world: float,
        y_world: float,
        geometry_type: str = "",
    ) -> None:
        """新規geometryの1点目だけ、描画位置に最も近いフレームへ移動する。

        LaneLinePanel側が1点目のときだけこのメソッドを呼ぶ。
        2Dビュー範囲は固定し、黄色い矩形・下段画像・投影だけ更新する。
        """
        if not self._is_draw_follow_enabled():
            return

        nearest = self._nearest_pose_frame_to_xy(
            float(x_world),
            float(y_world),
        )
        if nearest is None:
            return

        self._set_frame_preserving_2d_view(nearest)

    def _frame_skip(self, delta: int) -> None:
        """指定フレーム数だけ前後へスキップする。"""
        if (
            not hasattr(self, "_frame_slider")
            or not self._frame_slider.isEnabled()
        ):
            return

        new_value = self._frame_slider.value() + int(delta)
        new_value = max(
            0,
            min(
                new_value,
                self._frame_slider.maximum(),
            ),
        )
        self._frame_slider.setValue(new_value)

    # ------------------------------------------------------------------
    # フレームスライダー管理
    # ------------------------------------------------------------------
    def _setup_frame_slider(self, total_frames: int) -> None:
        """フレームスライダーを有効化して範囲を設定する。"""
        if not hasattr(self, "_frame_slider"):
            return
        self._frame_slider.blockSignals(True)
        self._frame_spin.blockSignals(True)
        self._frame_slider.setMaximum(max(total_frames - 1, 0))
        self._frame_spin.setMaximum(max(total_frames - 1, 0))
        self._frame_slider.setValue(0)
        self._frame_spin.setValue(0)
        self._frame_slider.setEnabled(total_frames > 0)
        self._frame_spin.setEnabled(total_frames > 0)
        self._frame_slider.blockSignals(False)
        self._frame_spin.blockSignals(False)
        self._lbl_frame_info.setText(f"1 / {total_frames}")
        self._current_frame_idx = 0
        self._ego_drag_last_frame = None

    def _reset_frame_slider(self) -> None:
        """フレームスライダーを無効化する。"""
        if not hasattr(self, "_frame_slider"):
            return
        self._frame_slider.blockSignals(True)
        self._frame_spin.blockSignals(True)
        self._frame_slider.setMaximum(0)
        self._frame_spin.setMaximum(0)
        self._frame_slider.setValue(0)
        self._frame_spin.setValue(0)
        self._frame_slider.setEnabled(False)
        self._frame_spin.setEnabled(False)
        self._frame_slider.blockSignals(False)
        self._frame_spin.blockSignals(False)
        self._lbl_frame_info.setText("— / —")
        self._ego_drag_last_frame = None

    def _on_frame_slider_changed(self, value: int) -> None:
        """マスタースライダー変更 → 全コンポーネントを一括更新する。"""
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(value)
        self._frame_spin.blockSignals(False)
        self._apply_frame(value)

    def _on_frame_spin_changed(self, value: int) -> None:
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(value)
        self._frame_slider.blockSignals(False)
        self._apply_frame(value)

    def _frame_prev(self) -> None:
        """前のフレームへ。"""
        if not hasattr(self, "_frame_slider"):
            return
        v = max(self._frame_slider.value() - 1, 0)
        self._frame_slider.setValue(v)

    def _frame_next(self) -> None:
        """次のフレームへ。"""
        if not hasattr(self, "_frame_slider"):
            return
        v = min(self._frame_slider.value() + 1, self._frame_slider.maximum())
        self._frame_slider.setValue(v)

    def _apply_frame(self, frame_idx: int) -> None:
        """フレームインデックスを全コンポーネントに適用する。

        適用順:
          1. フレーム情報ラベル更新
          2. カメラストリップパネル → 画像表示更新
          3. 2D BEV ビュー → 車両位置・姿勢更新
          4. アノテーション投影 → カメラ画像にオーバーレイ
        """
        self._current_frame_idx = frame_idx
        total = self._frame_slider.maximum() + 1
        self._lbl_frame_info.setText(f"{frame_idx + 1} / {total}")

        # ── カメラ画像更新 ─────────────────────────────────────
        if self._camera_strip_panel is not None:
            self._camera_strip_panel.set_frame(frame_idx)

        # ── ego pose 取得 ──────────────────────────────────────
        ego_x = ego_y = ego_z = ego_yaw = None
        ego_roll = ego_pitch = 0.0
        if self._frame_sync_table is not None:
            try:
                info = self._frame_sync_table[frame_idx]
                if info.has_pose:
                    ego_x, ego_y, ego_z = info.ego_x, info.ego_y, info.ego_z
                    ego_yaw   = info.ego_yaw
                    ego_roll  = getattr(info, "ego_roll",  0.0)
                    ego_pitch = getattr(info, "ego_pitch", 0.0)
            except (IndexError, Exception):
                pass
        elif self._mapping_poses:
            if frame_idx < len(self._mapping_poses):
                p = self._mapping_poses[frame_idx]
                ego_x     = float(p.get("x",     0.0))
                ego_y     = float(p.get("y",     0.0))
                ego_z     = float(p.get("z",     0.0))
                ego_yaw   = float(p.get("yaw",   0.0))
                ego_roll  = float(p.get("roll",  0.0))
                ego_pitch = float(p.get("pitch", 0.0))

        # ── 2D BEV ビュー: 車両位置更新 ───────────────────────
        if ego_x is not None and self._gl is not None:
            self._gl.set_ego_pose(ego_x, ego_y, ego_yaw)
            self._gl.set_pose_angles(ego_roll, ego_pitch,
                                     ego_yaw if ego_yaw is not None else 0.0)

        # ── SLAM match_score 表示更新 ──────────────────────────
        if self._gl is not None:
            match_score = None
            if self._frame_sync_table is not None:
                try:
                    info = self._frame_sync_table[frame_idx]
                    if info.pose is not None:
                        match_score = info.pose.get("match_score")
                except Exception:
                    pass
            elif self._mapping_poses and frame_idx < len(self._mapping_poses):
                match_score = self._mapping_poses[frame_idx].get("match_score")
            self._gl.set_match_score(match_score)

        # ── アノテーション投影 ─────────────────────────────────
        self._apply_annotation_projection(ego_x, ego_y, ego_z, ego_yaw,
                                          ego_roll, ego_pitch)

    def _apply_annotation_projection(
        self,
        ego_x: Optional[float],
        ego_y: Optional[float],
        ego_z: Optional[float],
        ego_yaw: Optional[float],
        ego_roll: float = 0.0,
        ego_pitch: float = 0.0,
    ) -> None:
        """アノテーションをカメラ画像に投影してオーバーレイ表示する。

        ego_z が None の場合は BEV画像の ground_z を代わりに使う。
        ego_roll, ego_pitch は道路傾斜補正に使用する。
        """
        if self._camera_strip_panel is None:
            print("[Projection] ❌ _camera_strip_panel is None → スキップ")
            return
        if self._annotation_projector is None:
            print("[Projection] ❌ _annotation_projector is None → スキップ")
            return
        if hasattr(self, "_chk_project_anno") and not self._chk_project_anno.isChecked():
            print("[Projection] ℹ️ 投影チェックボックス OFF → スキップ")
            self._camera_strip_panel.clear_annotation_overlays()
            return
        if ego_x is None or ego_y is None:
            print(f"[Projection] ❌ ego_x={ego_x} ego_y={ego_y} → スキップ")
            self._camera_strip_panel.clear_annotation_overlays()
            return

        annotations = []
        if self._gl is not None:
            annotations = getattr(self._gl, "draw_lane_data", [])

        print(f"\n[Projection] --- _apply_annotation_projection ---")
        print(f"[Projection] ego: x={ego_x:.3f} y={ego_y:.3f} z={ego_z} yaw={ego_yaw}")
        print(f"[Projection] annotations件数: {len(annotations)}")
        if annotations:
            for i, rec in enumerate(annotations[:3]):
                poly3d = rec.get("poly3d")
                verts  = rec.get("vertices", [])
                print(f"  anno[{i}] class={rec.get('class','?')} "
                      f"poly3d={'あり shape='+str(np.array(poly3d).shape) if poly3d is not None else 'None'} "
                      f"vertices数={len(verts)}")
                if poly3d is not None and len(poly3d) > 0:
                    arr = np.array(poly3d)
                    print(f"    poly3d[0]={arr[0]}  z範囲=[{arr[:,2].min():.3f},{arr[:,2].max():.3f}]")

        if not annotations:
            print("[Projection] ❌ annotations が空 → オーバーレイクリア")
            self._camera_strip_panel.clear_annotation_overlays()
            return

        # ego_z フォールバック
        if ego_z is None:
            ego_z = 0.0
            if self._gl is not None:
                bev = getattr(self._gl, "_bev_image", None)
                if bev is not None:
                    ego_z = float(getattr(bev, "ground_z", 0.0))
            print(f"[Projection] ego_z=None → BEV ground_z={ego_z:.3f} を使用")

        # カメラパネルの状態確認
        print(f"[Projection] camera_strip_panel: "
              f"total_frames={self._camera_strip_panel._total_frames} "
              f"current_frame={self._camera_strip_panel._current_frame} "
              f"overlay_enabled={self._camera_strip_panel._overlay_enabled}")

        # ── XYZデータをJSONに保存（デバッグ用）──────────────────
        debug_dir = getattr(self._camera_strip_panel, "_debug_save_dir", None)
        if debug_dir:
            try:
                import json as _json
                import os as _os
                _os.makedirs(debug_dir, exist_ok=True)
                debug_data = {
                    "frame_idx": self._current_frame_idx,
                    "ego": {
                        "x": float(ego_x), "y": float(ego_y),
                        "z": float(ego_z) if ego_z is not None else None,
                        "yaw_rad": float(ego_yaw) if ego_yaw is not None else None,
                        "yaw_deg": float(np.degrees(ego_yaw)) if ego_yaw is not None else None,
                    },
                    "annotations": [],
                }
                for rec in annotations:
                    poly3d = rec.get("poly3d")
                    verts  = rec.get("vertices", [])
                    if poly3d is not None and len(poly3d) > 0:
                        pts = np.array(poly3d)[:, :3].tolist()
                    else:
                        pts = [[v[0], v[1], v[2] if len(v) > 2 else 0.0] for v in verts]
                    debug_data["annotations"].append({
                        "class": rec.get("class", ""),
                        "geometry_type": rec.get("geometry_type", rec.get("shape", "polyline")),
                        "num_pts": len(pts),
                        "first_pt": pts[0] if pts else None,
                        "z_range": [float(min(p[2] for p in pts)), float(max(p[2] for p in pts))] if pts else None,
                        "pts": pts,
                    })
                xyz_path = _os.path.join(
                    debug_dir,
                    f"projection_xyz_f{self._current_frame_idx:04d}.json"
                )
                with open(xyz_path, "w", encoding="utf-8") as _f:
                    _json.dump(debug_data, _f, indent=2, ensure_ascii=False)
                print(f"[Projection] 💾 XYZデータ保存: {xyz_path}")
            except Exception as _je:
                print(f"[Projection] ⚠️ XYZデータ保存失敗: {_je}")

        try:
            from ui.camera_strip_panel import _CAMERA_ORDER
            # 実画像サイズを取得してProjectorに渡す（JSON height 誤記への対策）
            actual_sizes = (self._camera_strip_panel.get_actual_image_sizes()
                            if self._camera_strip_panel is not None else {})
            overlays = self._annotation_projector.project(
                annotations=annotations,
                ego_x=float(ego_x),
                ego_y=float(ego_y),
                ego_z=float(ego_z),
                ego_yaw=float(ego_yaw) if ego_yaw is not None else 0.0,
                ego_roll=float(ego_roll),
                ego_pitch=float(ego_pitch),
                cam_names=_CAMERA_ORDER,
                actual_image_sizes=actual_sizes,
            )
            # オーバーレイが1つでも得られたか確認
            any_overlay = any(len(v) > 0 for v in overlays.values())
            print(f"[Projection] overlays: {' '.join(k+'='+str(len(v)) for k,v in overlays.items())} "
                  f"{'✅' if any_overlay else '❌ 全カメラ投影なし'}")
            # force_frame でカメラパネルのフレームを MainWindow 側と同期させる
            self._camera_strip_panel.set_annotation_overlays(
                overlays,
                force_frame=self._current_frame_idx,
            )

            # _CameraCell に overlay_pixmap が設定されたか確認
            if hasattr(self._camera_strip_panel, '_cells'):
                for cam, cell in self._camera_strip_panel._cells.items():
                    has_ov = cell._overlay_pixmap is not None
                    has_base = cell._pixmap is not None
                    print(f"  cell[{cam}]: base_pixmap={'✅' if has_base else '❌'} "
                          f"overlay_pixmap={'✅' if has_ov else '❌'}")
        except Exception as e:
            import traceback
            print(f"[Projection] ❌ 例外: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Full-screen Projection Validation / Calibration Viewer
    # ------------------------------------------------------------------
    def _calibration_override_path(self) -> Optional[Path]:
        if not self._root_folder_path:
            return None
        return Path(self._root_folder_path) / "calibration_override.json"

    def _load_calibration_overrides_for_root(self) -> None:
        if self._annotation_projector is None:
            return
        self._annotation_projector.reset_all_calibration_overrides()
        path = self._calibration_override_path()
        if path is None or not path.exists():
            return
        try:
            if self._annotation_projector.load_calibration_overrides(str(path)):
                print(f"[Calibration] override loaded: {path}")
        except Exception as exc:
            print(f"[Calibration] override load failed: {exc}")

    def _on_calibration_viewer_frame_requested(self, frame_idx: int) -> None:
        if not hasattr(self, "_frame_slider") or not self._frame_slider.isEnabled():
            return
        frame_idx = max(0, min(int(frame_idx), self._frame_slider.maximum()))
        if self._frame_slider.value() == frame_idx:
            self._apply_frame(frame_idx)
        else:
            self._frame_slider.setValue(frame_idx)

    def _get_ego_pose_for_frame(self, frame_idx: int):
        ego_x = ego_y = ego_z = ego_yaw = None
        ego_roll = ego_pitch = 0.0
        if self._frame_sync_table is not None:
            try:
                info = self._frame_sync_table[frame_idx]
                if info.has_pose:
                    ego_x, ego_y, ego_z = info.ego_x, info.ego_y, info.ego_z
                    ego_roll, ego_pitch, ego_yaw = info.ego_roll, info.ego_pitch, info.ego_yaw
            except Exception:
                pass
        elif self._mapping_poses and 0 <= frame_idx < len(self._mapping_poses):
            pose = self._mapping_poses[frame_idx]
            ego_x = float(pose.get("x", 0.0)); ego_y = float(pose.get("y", 0.0)); ego_z = float(pose.get("z", 0.0))
            ego_roll = float(pose.get("roll", 0.0)); ego_pitch = float(pose.get("pitch", 0.0)); ego_yaw = float(pose.get("yaw", 0.0))
        return ego_x, ego_y, ego_z, ego_yaw, ego_roll, ego_pitch

    def _reproject_calibration_camera(self, cam_name: str) -> None:
        if self._annotation_projector is None or self._camera_strip_panel is None:
            return
        annotations = getattr(self._gl, "draw_lane_data", []) if self._gl is not None else []
        if not annotations:
            return
        frame_idx = int(getattr(self, "_current_frame_idx", 0))
        ego_x, ego_y, ego_z, ego_yaw, ego_roll, ego_pitch = self._get_ego_pose_for_frame(frame_idx)
        if ego_x is None or ego_y is None:
            return
        if ego_z is None:
            ego_z = 0.0
            if self._gl is not None:
                bev = getattr(self._gl, "_bev_image", None)
                if bev is not None:
                    ego_z = float(getattr(bev, "ground_z", 0.0))
        try:
            updated = self._annotation_projector.project(
                annotations=annotations, ego_x=float(ego_x), ego_y=float(ego_y), ego_z=float(ego_z),
                ego_yaw=float(ego_yaw) if ego_yaw is not None else 0.0,
                ego_roll=float(ego_roll), ego_pitch=float(ego_pitch),
                cam_names=[cam_name], use_calibration_override=True,
                actual_image_sizes=(self._camera_strip_panel.get_actual_image_sizes()
                                    if self._camera_strip_panel is not None else {}),
            )
            merged = dict(getattr(self._camera_strip_panel, "_current_overlays", {}))
            merged.update(updated)
            self._camera_strip_panel.set_annotation_overlays(merged, force_frame=frame_idx)
        except Exception as exc:
            print(f"[Calibration] reproject failed ({cam_name}): {exc}")
            traceback.print_exc()

    def _on_calibration_viewer_changed(self, cam_name: str, values: dict) -> None:
        if self._annotation_projector is None:
            return
        self._annotation_projector.set_calibration_override(cam_name, values)
        self._reproject_calibration_camera(cam_name)

    def _on_calibration_viewer_reset(self, cam_name: str) -> None:
        if self._annotation_projector is None:
            return
        self._annotation_projector.reset_calibration_override(cam_name)
        self._reproject_calibration_camera(cam_name)

    def _on_calibration_viewer_save(self) -> None:
        if self._annotation_projector is None:
            return
        path = self._calibration_override_path()
        if path is None:
            QMessageBox.warning(self, "Calibration", "ルートフォルダを開いてから補正値を保存してください。")
            return
        try:
            self._annotation_projector.save_calibration_overrides(str(path))
            self._status.showMessage(f"Calibration override saved: {path}")
            print(f"[Calibration] override saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Calibration", f"補正値の保存に失敗しました:\n{exc}")

    def _on_follow_mode_changed(self, state: int) -> None:
        """「車両追従」チェック変更。"""
        if state and hasattr(self, "_chk_free"):
            self._chk_free.blockSignals(True)
            self._chk_free.setChecked(False)
            self._chk_free.blockSignals(False)
        if self._gl is not None:
            self._gl.set_follow_mode("follow" if state else "free")

    def _on_free_mode_changed(self, state: int) -> None:
        """「自由表示」チェック変更。"""
        if state and hasattr(self, "_chk_follow"):
            self._chk_follow.blockSignals(True)
            self._chk_follow.setChecked(False)
            self._chk_follow.blockSignals(False)
        if self._gl is not None:
            self._gl.set_follow_mode("free" if state else "follow")

    def _on_project_anno_changed(self, state: int) -> None:
        """「投影」チェック変更：ON なら再投影、OFF ならオーバーレイクリア。"""
        if self._camera_strip_panel is None:
            return
        if state:
            ego_x = ego_y = ego_z = ego_yaw = None
            ego_roll = ego_pitch = 0.0
            if self._frame_sync_table is not None:
                try:
                    info = self._frame_sync_table[self._current_frame_idx]
                    if info.has_pose:
                        ego_x, ego_y, ego_z = info.ego_x, info.ego_y, info.ego_z
                        ego_yaw   = info.ego_yaw
                        ego_roll  = getattr(info, "ego_roll",  0.0)
                        ego_pitch = getattr(info, "ego_pitch", 0.0)
                except Exception:
                    pass
            elif self._mapping_poses:
                idx = self._current_frame_idx
                if idx < len(self._mapping_poses):
                    p = self._mapping_poses[idx]
                    ego_x     = float(p.get("x",     0.0))
                    ego_y     = float(p.get("y",     0.0))
                    ego_z     = float(p.get("z",     0.0))
                    ego_yaw   = float(p.get("yaw",   0.0))
                    ego_roll  = float(p.get("roll",  0.0))
                    ego_pitch = float(p.get("pitch", 0.0))
            self._apply_annotation_projection(ego_x, ego_y, ego_z, ego_yaw,
                                              ego_roll, ego_pitch)
            self._camera_strip_panel.set_overlay_enabled(True)
        else:
            self._camera_strip_panel.set_overlay_enabled(False)

    def _on_annotations_changed(self) -> None:
        """アノテーション (draw_lane_data) が更新されたときに再投影する。"""
        if self._camera_strip_panel is None:
            return
        if not (hasattr(self, "_chk_project_anno") and self._chk_project_anno.isChecked()):
            return

        ego_x = ego_y = ego_z = ego_yaw = None
        ego_roll = ego_pitch = 0.0
        if self._frame_sync_table is not None:
            try:
                info = self._frame_sync_table[self._current_frame_idx]
                if info.has_pose:
                    ego_x, ego_y, ego_z = info.ego_x, info.ego_y, info.ego_z
                    ego_yaw   = info.ego_yaw
                    ego_roll  = getattr(info, "ego_roll",  0.0)
                    ego_pitch = getattr(info, "ego_pitch", 0.0)
            except Exception:
                pass
        elif self._mapping_poses:
            idx = self._current_frame_idx
            if idx < len(self._mapping_poses):
                p = self._mapping_poses[idx]
                ego_x     = float(p.get("x",     0.0))
                ego_y     = float(p.get("y",     0.0))
                ego_z     = float(p.get("z",     0.0))
                ego_yaw   = float(p.get("yaw",   0.0))
                ego_roll  = float(p.get("roll",  0.0))
                ego_pitch = float(p.get("pitch", 0.0))

        self._apply_annotation_projection(ego_x, ego_y, ego_z, ego_yaw,
                                          ego_roll, ego_pitch)

    def _rebuild_recent_folders_menu(self) -> None:
        """最近開いたフォルダのメニューを再構築する"""
        from core.project_manager import get_recent_folders, clear_recent_folders
        self._menu_recent_folders.clear()
        recent = get_recent_folders()
        if not recent:
            act = QAction(t("menu_project_recent_empty"), self)
            act.setEnabled(False)
            self._menu_recent_folders.addAction(act)
        else:
            for path in recent:
                # work_dir の親フォルダ名を含むラベルにする
                # 例: .../work_dir/bevld → 親フォルダ/bevld
                p = Path(path)
                parent_name = p.parent.name   # 例: "work_dir" または日付フォルダ
                folder_name = p.name          # 例: "bevld"
                # work_dir 以外の意味ある親名を探す（work_dir を跳ばして上の名前）
                if parent_name.lower() in ("work_dir", "workdir", "bevld", "."):
                    # さらに上の階層も含める
                    label = f"{p.parent.parent.name}/{parent_name}/{folder_name}"
                else:
                    label = f"{parent_name}/{folder_name}"
                act = QAction(label, self)
                act.setToolTip(path)
                act.triggered.connect(
                    lambda checked, p=path: self._open_root_folder(p)
                )
                self._menu_recent_folders.addAction(act)
            self._menu_recent_folders.addSeparator()
            act_clear = QAction(t("menu_project_recent_clear"), self)
            act_clear.triggered.connect(lambda: (
                clear_recent_folders(), self._rebuild_recent_folders_menu()
            ))
            self._menu_recent_folders.addAction(act_clear)

    # ==================================================================
    # サンプルデータ読み込み
    # ==================================================================

    # サンプルファイルの固定パス
    _ANNO_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "annoSamples"
    _SAMPLE_PCD_PATH  = _ANNO_SAMPLES_DIR / "RGBBEV.pcd"
    _SAMPLE_POSE_PATH = _ANNO_SAMPLES_DIR / "mapping_pose.txt"
    _SAMPLE_JSON_PATH = _ANNO_SAMPLES_DIR / "GT" / "2025-10-14-09-04-01.json"
    _SAMPLE_PKL_PATH  = _ANNO_SAMPLES_DIR / "PCD-pkl" / "20251014090606_Sunny_City_Day_000010.pkl"

    def _open_sample_pcd(self) -> None:
        """PCDファイルをダイアログで選択して点群として表示する。

        選択したPCDと同じフォルダに mapping_pose.txt があれば自動で軌跡として読み込む。
        ASCII PCD形式（大容量ファイル対応）: 1000点に1点の割合でサンプリング表示。
        """
        if self._gl is None:
            return

        # ── ファイル選択ダイアログ ────────────────────────────────────
        # annoSamples フォルダを初期ディレクトリとして提案する
        default_dir = str(self._ANNO_SAMPLES_DIR) if self._ANNO_SAMPLES_DIR.exists() else ""
        pcd_path_str, _ = QFileDialog.getOpenFileName(
            self,
            t("menu_open_sample_pcd"),
            default_dir,
            t("file_filter_pcd"),
        )
        if not pcd_path_str:
            return

        pcd_path = Path(pcd_path_str)

        # 同フォルダの mapping_pose.txt を自動検索
        pose_path = pcd_path.parent / "mapping_pose.txt"
        if not pose_path.exists():
            pose_path = None

        try:
            self._toolbar_progress.setValue(0)
            self._toolbar_progress.setVisible(True)
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()

            # ── Step1: mapping_pose.txt を先に読んでカメラ範囲を確定 ──
            # 軌跡の XY 範囲 = PCD の表示範囲として _center / _scale を設定する
            self._toolbar_progress.setValue(10)
            QApplication.processEvents()

            if pose_path is not None:
                poses = self._parse_mapping_pose(str(pose_path))
                if poses:
                    xyz = np.array(
                        [[p['x'], p['y'], p['z']] for p in poses],
                        dtype=np.float32)
                    # 軌跡の重心と XY スパンからカメラパラメータを先に設定
                    center = xyz[:, :3].mean(axis=0)
                    self._gl._center = center
                    xy_min = xyz[:, :2].min(axis=0)
                    xy_max = xyz[:, :2].max(axis=0)
                    span = float(np.max(xy_max - xy_min))
                    if span > 1e-6:
                        self._gl._scale = 1.0 / (span + 1e-9)
                    # GL に軌跡をセット（mapping_posesも更新）
                    self._gl.set_trajectory_points(xyz)
                    self._trajectory_path = str(pose_path)
                    self._mapping_poses = poses
                    if self._layer_panel is not None:
                        self._layer_panel.set_loaded("trajectory", True)
                    # 軌跡進行方向に合わせて基準向きを設定
                    self._reset_to_home_view()
                    print(f"[MainWindow] 軌跡から表示範囲を設定: "
                          f"center=({center[0]:.1f},{center[1]:.1f}), span={span:.1f}m")

            self._toolbar_progress.setValue(30)
            QApplication.processEvents()

            # ── Step2: ASCII PCD 読み込み（ヘッダをスキップ） ────────────────
            # バイナリPCDの場合はヘッダのDATA行を確認してスキップ
            points_xyz = []

            with open(str(pcd_path), 'r', encoding='utf-8', errors='replace') as f:
                in_data = False
                data_format = "ascii"
                step = 0
                for line in f:
                    stripped = line.strip()
                    upper = stripped.upper()
                    # DATA行でフォーマットを確認
                    if upper.startswith("DATA"):
                        parts = upper.split()
                        data_format = parts[1] if len(parts) > 1 else "ascii"
                        in_data = True
                        if data_format != "ASCII":
                            # バイナリ形式は非対応
                            QMessageBox.warning(
                                self, t("dialog_warning"),
                                f"バイナリPCDは表示非対応です。\n"
                                f"フォーマット: {data_format}\n"
                                f"ASCII形式のPCDを選択してください。")
                            return
                        continue
                    if not in_data:
                        continue
                    # 1000点に1点をサンプリング（大容量ファイル対策）
                    step += 1
                    if step % 1000 != 0:
                        continue
                    cols = stripped.split()
                    if len(cols) < 3:
                        continue
                    try:
                        x, y, z = float(cols[0]), float(cols[1]), float(cols[2])
                        points_xyz.append([x, y, z])
                    except ValueError:
                        continue

            self._toolbar_progress.setValue(80)
            QApplication.processEvents()

            if not points_xyz:
                QMessageBox.warning(self, t("dialog_warning"),
                                    "PCD から有効な点を取得できませんでした。\n"
                                    "ASCII形式のPCDであることを確認してください。")
                return

            pts = np.array(points_xyz, dtype=np.float32)
            # 白色系の単色で点群表示
            colors = np.ones((len(pts), 3), dtype=np.float32) * 0.8
            self._gl.set_lane_points(pts, colors)
            # 点群全体が画面に収まるようにビューをフィット
            self._gl.fit_to_all()
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("lane_points", True)

            self._toolbar_progress.setValue(100)
            QApplication.processEvents()

            pose_info = f" + {pose_path.name}" if pose_path else ""
            self._status.showMessage(
                f"{t('status_sample_pcd_loaded')} {pcd_path.name}{pose_info} ({len(pts)} pts)")

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"),
                f"PCD読み込みエラー:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self._toolbar_progress.setVisible(False)

    def _open_sample_json(self) -> None:
        """サンプルGT JSON を読み込んで2Dビュー + アノテーション一覧に表示する。

        GT JSONの各エントリをポリラインレコードとして _lanes に追加する。
        draw_lane_data 経由で描画するため、クラス別/ID別色分けボタンが自動的に効く。
        source="imported" として区別する。
        """
        if self._gl is None:
            return

        json_path = self._SAMPLE_JSON_PATH
        if not json_path.exists():
            QMessageBox.warning(
                self, t("dialog_warning"),
                f"サンプルJSONが見つかりません:\n{json_path}")
            return

        try:
            import json as _json
            import re as _re
            with open(str(json_path), 'r', encoding='utf-8') as f:
                gt_list = _json.load(f)

            def _label_to_class_attrs(label: str) -> tuple:
                """GT JSONの label 文字列からクラス名と属性を返す"""
                extras: dict = {}
                if "dashed_line" in label or "Dashed" in label:
                    extras["line_type"] = "dash"
                    return "lane_line", extras
                if "solid_line" in label or "Solid" in label:
                    extras["line_type"] = "solid"
                    return "lane_line", extras
                if "WideDash" in label or "wide_dash" in label:
                    extras["line_type"] = "wide_dash"
                    return "lane_line", extras
                if "lane_normal" in label or "lane_line" in label:
                    return "lane_line", extras
                if "stop_line" in label:
                    return "stop_line", extras
                if "zebra" in label:
                    return "zebra_line", extras
                if "curb" in label:
                    return "curb_boundary", extras
                if "fence" in label:
                    return "fence_boundary", extras
                if "split_point" in label:
                    return "split_point", extras
                if "merge_point" in label:
                    return "merge_point", extras
                return "lane_line", extras

            # クラス対応表示色（CLASS_DISPLAY_COLORSに準拠）
            _CLASS_COLORS = {
                "lane_line":            (0.9, 0.9, 0.9),
                "stop_line":            (1.0, 0.3, 0.3),
                "curb_boundary":        (0.5, 0.7, 1.0),
                "fence_boundary":       (0.6, 0.4, 0.2),
                "TrafficCone_boundary": (1.0, 0.8, 0.0),
                "WaterSafety_boundary": (0.3, 0.6, 1.0),
                "boundary":             (0.3, 0.8, 0.3),
                "zebra_line":           (1.0, 0.6, 0.2),
                "split_point":          (1.0, 0.0, 1.0),
                "merge_point":          (0.8, 0.2, 0.8),
            }

            new_records = []
            if self._panel is not None:
                for obj in gt_list:
                    label = obj.get("label", "")
                    cls_name, extra_attrs = _label_to_class_attrs(label)

                    if cls_name in ("split_point", "merge_point"):
                        bp = obj.get("branch_point", None)
                        if bp is None:
                            continue
                        verts = [[float(bp[0]), float(bp[1]),
                                  float(bp[2]) if len(bp) > 2 else 0.0]]
                        geom = "point"
                    else:
                        lanes_coords = obj.get("lanes", [])
                        if len(lanes_coords) < 2:
                            continue
                        verts = [[float(v[0]), float(v[1]),
                                  float(v[2]) if len(v) > 2 else 0.0]
                                 for v in lanes_coords]
                        geom = "polyline"

                    pts = np.array(verts, dtype=np.float32)

                    # ── number フィールドから N_X-Y を分解 ──────────────
                    # None や空文字の場合は空文字として扱う
                    raw_number = obj.get("number")
                    lane_num_str = str(raw_number) if raw_number is not None else ""
                    lane_id_n, lane_id_x, lane_id_y = "", "", ""
                    if lane_num_str:
                        m = _re.match(r'^(\d+)_(\d+)-(\d+)$', lane_num_str)
                        if m:
                            lane_id_n = m.group(1)
                            lane_id_x = m.group(2)
                            lane_id_y = m.group(3)
                        else:
                            # "N" のみや別フォーマットはNに格納
                            lane_id_n = lane_num_str

                    # line_color の正規化
                    raw_color = obj.get("color") or "white"
                    line_color = raw_color if raw_color in ("white", "yellow") else "white"

                    # 色: ID別モード時は _generate_random_color で個別色、
                    # クラス別モード時は CLASS_DISPLAY_COLORS を使う。
                    # _lane_color() が _color → class の優先順で使うため、
                    # _color を None にしてクラス色にフォールバックさせる。
                    # ID別モードでは _lane_color が _color を返すので個別色を設定する。
                    from lane_line_panel import _generate_random_color
                    id_color = _generate_random_color()

                    rec = {
                        "id": f"lane_{self._panel._next_id:04d}",
                        "class": cls_name,
                        "geometry_type": geom,
                        "shape": geom,
                        "vertices": verts,
                        "poly3d": pts,
                        "_is_spline": False,
                        "_color": id_color,          # ID別色分け用（ランダム色）
                        "_class_color": _CLASS_COLORS.get(cls_name, (0.7, 0.7, 0.7)),
                        "source": "imported",
                        "coordinate_system": "map",
                        "visibility": "visible",
                        "quality": "ok",
                        "is_interpolated": "false",
                        "interpolation_reason": "",
                        "note": f"GT JSON: {label}",
                        "line_type": extra_attrs.get("line_type", "solid"),
                        "line_color": line_color,
                        "line_count": "single",
                        "lane_number": lane_num_str,
                        "lane_id_n": lane_id_n,
                        "lane_id_x": lane_id_x,
                        "lane_id_y": lane_id_y,
                        "lane_id_manual": True,
                        "boundary_id": "",
                        "start_line_ids": "",
                        "end_line_ids": "",
                        "buffer": 0.10,
                    }
                    self._panel._next_id += 1
                    new_records.append(rec)

                # アノテーション一覧に追加してテーブルを更新
                self._panel._lanes.extend(new_records)
                self._gl.draw_lane_data = self._panel._lanes
                if hasattr(self._panel, '_refresh_table'):
                    self._panel._refresh_table()
                if self._layer_panel is not None:
                    self._layer_panel.set_loaded("annotations", True)
                    self._layer_panel.set_loaded("sample_json", True)
                self._has_unsaved_changes = True

            # GL表示用の _sample_json_lanes も更新（レイヤー表示制御用）
            # ただし draw_lane_data 経由で描画するため、ここでは空にする
            self._gl._sample_json_lanes = []
            self._gl.update()

            self._status.showMessage(
                t("status_sample_json_loaded").format(len(new_records)))

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"),
                f"サンプルJSON読み込みエラー:\n{e}")
            import traceback
            traceback.print_exc()

    def _open_sample_pkl(self) -> None:
        """サンプルPKL を読み込んで2Dビュー + アノテーション一覧に表示する。

        PKL座標はエゴ相対座標系のため、ファイル名末尾のフレーム番号から
        mapping_pose.txt の対応姿勢を取得してグローバル座標に変換して表示する。

        変換式: gx = cos(yaw)*ex - sin(yaw)*ey + pose_x
                gy = sin(yaw)*ex + cos(yaw)*ey + pose_y
        """
        if self._gl is None:
            return

        pkl_path = self._SAMPLE_PKL_PATH
        if not pkl_path.exists():
            QMessageBox.warning(
                self, t("dialog_warning"),
                f"サンプルPKLが見つかりません:\n{pkl_path}")
            return

        try:
            import pickle as _pickle
            import re as _re
            import math as _math

            with open(str(pkl_path), 'rb') as f:
                data = _pickle.load(f)

            # ── フレーム番号を PKL ファイル名から取得 ──────────────────
            # 例: "20251014090606_Sunny_City_Day_000010.pkl" → frame=10
            frame_id: int | None = None
            m = _re.search(r'_(\d+)\.pkl$', pkl_path.name)
            if m:
                frame_id = int(m.group(1))
            print(f"[MainWindow] PKL frame_id={frame_id}")

            # ── mapping_pose.txt から対応姿勢を取得 ────────────────────
            # 既に _mapping_poses がある場合はそれを使う、なければ試みる
            pose_x, pose_y, pose_z, yaw = None, None, None, None

            mapping_poses = getattr(self, '_mapping_poses', None) or []
            if not mapping_poses:
                # 同フォルダの mapping_pose.txt を試みる
                candidate = pkl_path.parent.parent / "mapping_pose.txt"
                if candidate.exists():
                    mapping_poses = self._parse_mapping_pose(str(candidate))

            if frame_id is not None and mapping_poses:
                # frame フィールドで一致するものを探す
                for pose in mapping_poses:
                    if pose.get('frame') == frame_id:
                        pose_x = pose['x']
                        pose_y = pose['y']
                        pose_z = pose['z']
                        yaw    = pose['yaw']
                        break
                if pose_x is None:
                    # frame番号でなくインデックスで参照（0始まり）
                    if frame_id < len(mapping_poses):
                        p = mapping_poses[frame_id]
                        pose_x = p['x']; pose_y = p['y']
                        pose_z = p['z']; yaw = p['yaw']

            if pose_x is not None:
                print(f"[MainWindow] PKL pose: ({pose_x:.3f},{pose_y:.3f},{pose_z:.3f})"
                      f"  yaw={_math.degrees(yaw):.2f}deg")
            else:
                print("[MainWindow] PKL 姿勢が見つからないためエゴ座標系のまま表示")

            # エゴ→グローバル変換ヘルパー
            def _ego_to_global(pts_ego: np.ndarray) -> list:
                """エゴ相対座標をグローバル座標に変換する。"""
                if pose_x is None:
                    return [[float(p[0]), float(p[1]),
                             float(p[2]) if len(p) > 2 else 0.0]
                            for p in pts_ego]
                cos_y = _math.cos(yaw)
                sin_y = _math.sin(yaw)
                result = []
                for p in pts_ego:
                    ex, ey = float(p[0]), float(p[1])
                    ez = float(p[2]) if len(p) > 2 else 0.0
                    gx = cos_y * ex - sin_y * ey + pose_x
                    gy = sin_y * ex + cos_y * ey + pose_y
                    gz = ez + pose_z
                    result.append([gx, gy, gz])
                return result

            # PKLキー → クラス名マッピング
            _PKL_KEY_TO_CLASS = {
                "dashed_line":    "lane_line",
                "solid_line":     "lane_line",
                "WideDash_line":  "lane_line",
                "lane_normal":    "lane_line",
                "curb_boundary":  "curb_boundary",
                "fence_boundary": "fence_boundary",
                "stop_line":      "stop_line",
                "zebra_line":     "zebra_line",
            }
            _PKL_KEY_TO_LINE_TYPE = {
                "dashed_line":    "dash",
                "solid_line":     "solid",
                "WideDash_line":  "wide_dash",
                "lane_normal":    "solid",
            }

            records = []
            if self._panel is not None:
                all_entries: list = []
                if isinstance(data, dict):
                    for key, entries in data.items():
                        if isinstance(entries, list):
                            for entry in entries:
                                entry["_pkl_key"] = key
                            all_entries.extend(entries)
                elif isinstance(data, list):
                    all_entries = data

                for entry in all_entries:
                    pts_raw = entry.get("points_20", None)
                    if pts_raw is None:
                        continue
                    try:
                        pts_ego = np.array(pts_raw, dtype=np.float32)
                    except Exception:
                        continue
                    if pts_ego.ndim != 2 or pts_ego.shape[1] < 2:
                        continue

                    # エゴ → グローバル変換
                    verts = _ego_to_global(pts_ego)
                    if len(verts) < 2:
                        continue

                    pkl_key = entry.get("_pkl_key", "")
                    cls_name = _PKL_KEY_TO_CLASS.get(pkl_key, "lane_line")
                    line_type = _PKL_KEY_TO_LINE_TYPE.get(pkl_key, "solid")

                    # lane_num パース: 文字列 "N_X-Y" or 整数
                    raw_lane_num = entry.get("lane_num", "")
                    lane_num_str = ""
                    lane_id_n, lane_id_x, lane_id_y = "", "", ""
                    if raw_lane_num is not None and raw_lane_num != 0 and raw_lane_num != "":
                        lane_num_str = str(raw_lane_num)
                        m2 = _re.match(r'^(\d+)_(\d+)-(\d+)$', lane_num_str)
                        if m2:
                            lane_id_n = m2.group(1)
                            lane_id_x = m2.group(2)
                            lane_id_y = m2.group(3)
                        else:
                            lane_id_n = lane_num_str

                    poly_arr = np.array(verts, dtype=np.float32)
                    rec = {
                        "id": f"lane_{self._panel._next_id:04d}",
                        "class": cls_name,
                        "geometry_type": "polyline",
                        "shape": "polyline",
                        "vertices": verts,
                        "poly3d": poly_arr,
                        "_is_spline": False,
                        "_color": (1.0, 0.6, 0.1),  # 橙色（PKL識別用）
                        "source": "imported_pkl",
                        "coordinate_system": "map" if pose_x is not None else "ego_vehicle",
                        "visibility": "visible",
                        "quality": "ok",
                        "is_interpolated": "false",
                        "interpolation_reason": "",
                        "note": (f"PKL: {pkl_key} idx={entry.get('idx','')}"
                                 f" frame={frame_id}"),
                        "line_type": line_type,
                        "line_color": "white",
                        "line_count": "single",
                        "lane_number": lane_num_str,
                        "lane_id_n": lane_id_n,
                        "lane_id_x": lane_id_x,
                        "lane_id_y": lane_id_y,
                        "lane_id_manual": True,
                        "boundary_id": "",
                        "start_line_ids": "",
                        "end_line_ids": "",
                        "buffer": 0.10,
                    }
                    self._panel._next_id += 1
                    records.append(rec)

                # アノテーション一覧に追加してテーブルを更新
                self._panel._lanes.extend(records)
                self._gl.draw_lane_data = self._panel._lanes
                if hasattr(self._panel, '_refresh_table'):
                    self._panel._refresh_table()
                if self._layer_panel is not None:
                    self._layer_panel.set_loaded("annotations", True)
                    self._layer_panel.set_loaded("sample_pkl", True)
                self._has_unsaved_changes = True

            # GL の _sample_pkl_lanes は draw_lane_data 経由で描画するため空に
            self._gl._sample_pkl_lanes = []
            self._gl._vis_sample_pkl = True
            self._gl.update()

            coord_info = "グローバル座標" if pose_x is not None else "エゴ座標(変換不可)"
            self._status.showMessage(
                f"{t('status_sample_pkl_loaded').format(len(records))} [{coord_info}]")

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"),
                f"サンプルPKL読み込みエラー:\n{e}")
            import traceback
            traceback.print_exc()

    # ==================================================================
    # サンプリングレーン自動インポート
    # ==================================================================

    def _auto_import_sampling_lane(self, filepath: str) -> None:
        """sampling_lane_global.json/txt をプリアノテーションとして自動インポートする。

        ルートフォルダを開いたときに自動呼び出しされる。
        既存のアノテーションデータが存在する場合はインポートをスキップする。
        """
        if self._panel is None or self._gl is None:
            return

        # 既存のアノテーションデータがある場合はスキップ（手動データを壊さない）
        if self._panel._lanes:
            print(f"[MainWindow] サンプリングレーン自動インポートスキップ: "
                  f"既存レコード {len(self._panel._lanes)} 本あり")
            return

        from core.sampling_lane_importer import load_sampling_lane_json

        new_records = load_sampling_lane_json(filepath)
        if not new_records:
            print(f"[MainWindow] サンプリングレーンデータなし: {filepath}")
            return

        # poly3d を計算して ID を振り直す
        for rec in new_records:
            verts = rec["vertices"]
            pts = np.array(
                [(v[0], v[1], v[2]) for v in verts],
                dtype=np.float32)
            rec["poly3d"] = pts
            rec["_is_spline"] = False
            rec["id"] = f"lane_{self._panel._next_id:04d}"
            self._panel._next_id += 1

        self._panel._lanes.extend(new_records)
        self._gl.draw_lane_data = self._panel._lanes
        if hasattr(self._panel, '_refresh_table'):
            self._panel._refresh_table()
        self._gl.update()
        if self._layer_panel is not None:
            self._layer_panel.set_loaded("annotations", True)
        self._has_unsaved_changes = True

        print(f"[MainWindow] サンプリングレーン自動インポート完了: "
              f"{len(new_records)} 本 ({Path(filepath).name})")

    # ==================================================================
    # GT JSON / アノテーション出力
    # ==================================================================

    def _export_gt_json(self) -> None:
        """GT JSON 形式でアノテーションを出力する。

        保存先: {ルートフォルダ}/GT/{ルートフォルダ名}.json
        ルートフォルダが未設定の場合はダイアログで保存先を選択する。
        """
        if self._panel is None or not self._panel._lanes:
            QMessageBox.warning(self, t("menu_export_gt_json"), t("warn_no_export_data"))
            return

        # ── ID未付与チェック（lane_line + boundary 系を含む全ポリライン対象） ──
        # split_point / merge_point / zebra_line / stop_line は除外
        _ID_EXEMPT_CLASSES = {"split_point", "merge_point", "zebra_line", "stop_line"}
        missing_id_lanes = [
            rec for rec in self._panel._lanes
            if rec.get("class") not in _ID_EXEMPT_CLASSES
            and rec.get("geometry_type", rec.get("shape", "")) == "polyline"
            and not rec.get("lane_id_n", "").strip()
        ]
        if missing_id_lanes:
            ids_preview = ", ".join(
                f"{rec.get('class','?')}({rec.get('id','?')})"
                for rec in missing_id_lanes[:5]
            )
            suffix = f"... 他 {len(missing_id_lanes) - 5} 件" if len(missing_id_lanes) > 5 else ""
            QMessageBox.warning(
                self, t("menu_export_gt_json"),
                t("warn_missing_n_id").format(ids_preview + suffix)
            )
            return

        # ── 保存先を自動決定 ──────────────────────────────────────────
        filepath: Optional[str] = None
        if self._root_folder_path:
            root = Path(self._root_folder_path)
            gt_dir = root / "GT"
            filepath = str(gt_dir / f"{root.name}.json")
        else:
            # ルートフォルダ未設定時はダイアログ
            default_dir = ""
            if self._image_path:
                p = Path(self._image_path)
                default_dir = str(p.parent / "GT")
            filepath, _ = QFileDialog.getSaveFileName(
                self, t("menu_export_gt_json"),
                str(Path(default_dir) / "gt.json") if default_dir else "gt.json",
                "JSON Files (*.json)")
            if not filepath:
                return
        try:
            import json
            import uuid
            bev_meta = getattr(self._panel, '_bev_meta', None)
            from core.pkl_exporter import (pixel_to_lidar, needs_coordinate_conversion)
            gt_list = []
            counter = 1
            for rec in self._panel._lanes:
                verts = rec.get("vertices", [])
                if len(verts) < 2:
                    continue
                if bev_meta and needs_coordinate_conversion(verts, bev_meta):
                    verts = pixel_to_lidar(verts, bev_meta)
                if (self._pcd_z_resolver is not None and self._pcd_z_resolver.is_loaded):
                    verts = self._pcd_z_resolver.resolve_z_for_vertices(verts)
                cls = rec.get("class", "lane_line")
                if cls in ("split_point", "merge_point"):
                    branch_pt = verts[0] if verts else [0.0, 0.0, 0.0]
                    branch_xyz = [round(float(branch_pt[0]), 6),
                                  round(float(branch_pt[1]), 6),
                                  round(float(branch_pt[2]) if len(branch_pt) > 2 else 0.0, 6)]
                    incoming = [s.strip() for s in
                                rec.get("start_line_ids", "").split(",") if s.strip()]
                    outgoing = [s.strip() for s in
                                rec.get("end_line_ids", "").split(",") if s.strip()]
                    obj = {"label": f"{counter}_{cls}", "kognic_uuid": str(uuid.uuid4()),
                           "branch_point": branch_xyz, "incoming_lanes": incoming,
                           "outgoing_lanes": outgoing}
                    gt_list.append(obj)
                    counter += 1
                    continue
                line_type = rec.get("line_type", "solid")
                LINE_TYPE_TO_SUFFIX = {
                    "solid": "solid_line", "dash": "dashed_line",
                    "wide_dash": "WideDash_line", "fish_bone": "WideDash_line",
                    "dash_solid": "dashed_line", "virtual": "dashed_line",
                    "unknown": "solid_line", "Solid_line": "solid_line",
                    "Dashed_line": "dashed_line", "WideDash_line": "WideDash_line",
                    "Virtual_lane": "dashed_line",
                }
                cls_to_suffix = {
                    "lane_line": LINE_TYPE_TO_SUFFIX.get(line_type, "solid_line"),
                    "stop_line": "stop_line", "zebra_line": "zebra_line",
                    "curb_boundary": "curb_boundary", "fence_boundary": "fence_boundary",
                    "TrafficCone_boundary": "curb_boundary",
                    "WaterSafety_boundary": "other_boundary", "boundary": "other_boundary",
                }
                suffix = cls_to_suffix.get(cls, LINE_TYPE_TO_SUFFIX.get(line_type, "solid_line"))

                # boundary系・stop_line・zebra_lineは color/number なし
                is_line_type = cls in ("lane_line",)
                is_boundary = cls in ("curb_boundary", "fence_boundary",
                                      "TrafficCone_boundary", "WaterSafety_boundary", "boundary",
                                      "stop_line", "zebra_line")

                # GT JSON は Global座標の頂点列をそのまま出力する（サンプル準拠）
                # アノテーターが編集した頂点をありのまま保存する
                lanes_coords = [[round(float(v[0]), 6), round(float(v[1]), 6),
                                 round(float(v[2]) if len(v) > 2 else 0.0, 6)]
                                for v in verts]

                if is_line_type:
                    # lane_line: number は lane_number フィールドを直接使う
                    number = rec.get("lane_number", "")
                    if not number:
                        # フォールバック: N_X-Y を組み立てる
                        n_id = rec.get("lane_id_n", rec.get("lane_number", ""))
                        x_id = rec.get("lane_id_x", "0")
                        y_id = rec.get("lane_id_y", "0")
                        number = f"{n_id}_{x_id}-{y_id}" if n_id else ""
                    color_raw = rec.get("line_color", "white")
                    color_map = {"white": "white", "yellow": "yellow", "orange": "orange",
                                 "blue": "white", "other": "white"}
                    color = color_map.get(color_raw, "white")
                    obj = {
                        "label": f"{counter}_{suffix}",
                        "kognic_uuid": str(uuid.uuid4()),
                        "lanes": lanes_coords,
                        "color": color,
                        "number": number,
                    }
                else:
                    # boundary / stop_line / zebra_line: lanes のみ
                    obj = {
                        "label": f"{counter}_{suffix}",
                        "kognic_uuid": str(uuid.uuid4()),
                        "lanes": lanes_coords,
                    }
                gt_list.append(obj)
                counter += 1
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(gt_list, f, ensure_ascii=False, indent=2)
            self._status.showMessage(
                f"{t('status_gt_json_exported')} {Path(filepath).name} ({len(gt_list)} obj)")
            QMessageBox.information(
                self, t("menu_export_gt_json"),
                t("info_gt_json_done").format(len(gt_list), filepath))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_gt_json_export')} {e}")
            traceback.print_exc()

    def _open_annotation(self) -> None:
        """アノテーションファイル (CVAT XML) を開く"""
        # 初期ディレクトリを決定する（アノテーションXMLの保存場所を優先）
        # 優先順: ①画像と同フォルダ → ②ルート/mapqr_input/ → ③ルート直下 → ④空
        init_dir = ""
        if self._image_path and Path(self._image_path).parent.is_dir():
            init_dir = str(Path(self._image_path).parent)
        elif self._root_folder_path:
            mapqr_dir = Path(self._root_folder_path) / "mapqr_input"
            if mapqr_dir.is_dir():
                init_dir = str(mapqr_dir)
            else:
                init_dir = self._root_folder_path

        filepath, _ = QFileDialog.getOpenFileName(
            self, t("menu_open_annotation"), init_dir, t("file_filter_xml"))
        if not filepath:
            return
        self._load_annotation_file(filepath)

    # ==================================================================
    # ファイル読み込み内部実装
    # ==================================================================

    def _load_image_file(self, filepath: str) -> None:
        """BEV 画像ファイルを読み込む。隣に z_gradient があれば自動でオーバーレイ読み込み"""
        if DataLoader is None or self._gl is None:
            return
        try:
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()
            image_layer = DataLoader.load_bev_image(filepath)
            self._gl.set_bev_image(image_layer)
            self._image_path = filepath
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("bev_image", True)
            self._status.showMessage(f"{t('status_image_loaded')} {Path(filepath).name}")
            p = Path(filepath)
            grad_stem = p.stem.replace("_rgb", "_z_gradient")
            grad_path = p.parent / f"{grad_stem}.png"
            if not grad_path.exists():
                candidates = list(p.parent.glob("*_z_gradient.png"))
                grad_path = candidates[0] if candidates else None
            if grad_path is not None and grad_path.exists():
                try:
                    self._bev_overlay_path = str(grad_path)
                    init_threshold = 208
                    self._gl.set_bev_overlay_with_threshold(self._bev_overlay_path, init_threshold)
                    if self._layer_panel is not None:
                        self._layer_panel.set_loaded("bev_overlay", True)
                        overlay_row = self._layer_panel._rows.get("bev_overlay")
                        if overlay_row and overlay_row._spinbox is not None:
                            overlay_row._spinbox.setValue(init_threshold)
                    print(f"[MainWindow] z_gradient 読み込み: {grad_path.name}")
                except Exception as e:
                    print(f"[MainWindow] z_gradient 失敗: {e}")
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('dialog_load_fail_msg')} {e}")

    def _load_lane_file(self, filepath: str) -> None:
        """レーン点群ファイルを読み込む"""
        if DataLoader is None or self._gl is None:
            return
        try:
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()
            points = DataLoader.parse_lane_points(filepath)
            if len(points) > 0:
                colors = DataLoader.generate_cluster_colors(points[:, 3])
                self._gl.set_lane_points(points[:, :3], colors)
            else:
                self._gl.set_lane_points(
                    points[:, :3] if points.shape[1] >= 3 else points, None)
            self._set_cluster_file_to_panel(filepath)
            self._lane_path = filepath
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("lane_points", True)
            self._status.showMessage(f"{t('status_lane_loaded')} {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('dialog_load_fail_msg')} {e}")

    def _load_trajectory_file(self, filepath: str) -> None:
        """軌跡ファイル (mapping_pose.txt) を読み込む"""
        if self._gl is None:
            return
        try:
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()
            poses = self._parse_mapping_pose(filepath)
            if not poses:
                raise ValueError("有効な姿勢データが見つかりません")
            xyz = np.array([[p['x'], p['y'], p['z']] for p in poses], dtype=np.float32)
            self._gl.set_trajectory_points(xyz)
            self._trajectory_path = filepath
            self._mapping_poses = poses
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("trajectory", True)
            self._status.showMessage(
                f"{t('status_pose_loaded')} {Path(filepath).name} ({len(poses)} poses)")
            # 軌跡読み込み後、自動的に基本位置（進行方向が画面上向き）に回転する
            self._reset_to_home_view()
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('dialog_load_fail_msg')} {e}")

    @staticmethod
    def _parse_mapping_pose(filepath: str) -> list:
        """mapping_pose.txt を解析して pose リストを返す"""
        poses = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        start = 0
        if lines and not lines[0].strip()[0].isdigit():
            start = 1
        for line in lines[start:]:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                poses.append({
                    'frame':       int(parts[0]),
                    'x':           float(parts[1]),
                    'y':           float(parts[2]),
                    'z':           float(parts[3]),
                    'roll':        float(parts[4]),
                    'pitch':       float(parts[5]),
                    'yaw':         float(parts[6]),
                    'match_score': float(parts[7]) if len(parts) > 7 else None,
                    'timestamp':   parts[8] if len(parts) > 8 else '',
                })
            except (ValueError, IndexError):
                continue
        return poses

    def _load_ref_pcd_file(self, filepath: str) -> None:
        """参照 PCD ファイルを読み込む (Z 補完用)。プログレスバーで進捗を表示する。"""
        try:
            from core.pcd_z_resolver import PcdZResolver

            # プログレスバー表示開始
            self._toolbar_progress.setValue(0)
            self._toolbar_progress.setVisible(True)
            self._status.showMessage(
                f"{t('status_ref_pcd_loading')} {Path(filepath).name}"
            )
            QApplication.processEvents()

            resolver = PcdZResolver()

            def _on_progress(value: int) -> None:
                self._toolbar_progress.setValue(value)
                QApplication.processEvents()

            resolver.load(filepath, progress_cb=_on_progress)
            self._pcd_z_resolver = resolver

            self._toolbar_progress.setValue(100)
            QApplication.processEvents()
            self._status.showMessage(
                f"{t('status_ref_pcd_loaded')} {Path(filepath).name}"
            )

        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_ref_pcd_load')} {e}")
            self._pcd_z_resolver = None
        finally:
            self._toolbar_progress.setVisible(False)

    def _load_annotation_file(self, filepath: str) -> None:
        """CVAT XML アノテーションファイルを読み込んで GL に反映する"""
        if BevCvatConverter is None or self._panel is None or self._gl is None:
            return
        try:
            il = self._gl._bev_image if self._gl else None
            converter = BevCvatConverter(il)
            records = converter.cvat_xml_to_lanes(filepath)
            if records:
                self._panel._lanes = records
                self._gl.draw_lane_data = records   # ← 2D描画に必須
                if hasattr(self._panel, '_refresh_table'):
                    self._panel._refresh_table()
                if hasattr(self._panel, '_rebuild_all_poly3d'):
                    self._panel._rebuild_all_poly3d()
                else:
                    # _rebuild_all_poly3d がない場合は各レコードの poly3d を個別に再構築する
                    # lane_line_panel の _rebuild_poly3d_cur 相当の処理をパネル経由で呼ぶ
                    if hasattr(self._panel, '_rebuild_poly3d_cur'):
                        for rec in records:
                            self._panel._rebuild_poly3d_cur(rec)
                    elif hasattr(self._panel, '_use_spline'):
                        # lane_line_panel の _rebuild_poly3d をモジュールから直接呼ぶ
                        try:
                            from lane_line_panel import _rebuild_poly3d
                            use_spline = getattr(self._panel, '_use_spline', False)
                            for rec in records:
                                _rebuild_poly3d(rec, use_spline=use_spline)
                        except ImportError:
                            pass
                self._gl.update()
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("annotations", True)
            self._status.showMessage(f"{t('status_annotation_loaded')} {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('dialog_load_fail_msg')} {e}")

    # ==================================================================
    # プリアノテーション読み込み
    # ==================================================================

    def _import_prelabel_json(self) -> None:
        """sampling_lane_global.json を読み込んでプリアノテーションとして登録する。

        処理概要:
          1. ファイルダイアログでJSONファイルを選択する
             （BEV画像と同じセッションに sampling_lane_global.json があれば自動提案）
          2. 既存のprelabelレコード（source=="prelabel"）をすべて削除する
          3. JSONのキーごとに1本のポリラインを生成し、_lanes に追加する
             ノードの座標・順序はそのまま使用し、加工は一切しない
        """
        if self._panel is None or self._gl is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_panel_gl"))
            return

        # ── ファイルパスの決定 ───────────────────────────────────────────
        import os as _os
        auto_filepath = None
        default_dir = ""
        bev_path = getattr(self, '_image_path', None)

        # ルートフォルダ開時にセットされたsampling_lane候補を優先する
        sampling_hint = getattr(self, '_sampling_lane_hint', None)
        if sampling_hint and _os.path.isfile(sampling_hint):
            auto_filepath = sampling_hint
            default_dir = _os.path.dirname(sampling_hint)
        elif bev_path:
            # BEV画像パス例: .../result/mapqr_input/RGBBEV_rgb.png
            # JSON候補パス:  .../result/sampling_lane/sampling_lane_global.json
            result_dir = _os.path.dirname(_os.path.dirname(bev_path))
            for fname in ("sampling_lane_global.json", "sampling_lane_global.txt"):
                candidate = _os.path.join(result_dir, "sampling_lane", fname)
                if _os.path.isfile(candidate):
                    auto_filepath = candidate
                    default_dir = _os.path.dirname(candidate)
                    break
            if not auto_filepath:
                default_dir = result_dir

        if auto_filepath:
            reply = QMessageBox.question(
                self,
                t("menu_import_prelabel"),
                f"同じセッションのファイルが見つかりました:\n\n"
                f"{auto_filepath}\n\n"
                f"このファイルを読み込みますか？\n"
                f"（「いいえ」で別ファイルを選択できます）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            filepath = auto_filepath if reply == QMessageBox.Yes else None
            if filepath is None:
                filepath, _ = QFileDialog.getOpenFileName(
                    self, t("menu_import_prelabel"), default_dir,
                    t("file_filter_json"))
        else:
            filepath, _ = QFileDialog.getOpenFileName(
                self, t("menu_import_prelabel"), default_dir,
                t("file_filter_json"))

        if not filepath:
            return

        try:
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()

            from core.sampling_lane_importer import load_sampling_lane_json

            # JSONを読み込む（ノードをそのまま頂点として使用）
            new_records = load_sampling_lane_json(filepath)

            if not new_records:
                QMessageBox.information(self, t("dialog_info"),
                                        t("warn_prelabel_no_data"))
                return

            # ── 既存のprelabelレコードを削除（重複防止） ──────────────
            before_count = len(self._panel._lanes)
            self._panel._lanes = [
                r for r in self._panel._lanes
                if r.get("source") != "prelabel"
            ]
            removed = before_count - len(self._panel._lanes)
            if removed:
                print(f"[MainWindow] 既存のprelabelレコード {removed} 本を削除")

            # ── poly3d を計算してIDを振り直す ─────────────────────────
            for rec in new_records:
                verts = rec["vertices"]
                pts = np.array(
                    [(v[0], v[1], v[2]) for v in verts],
                    dtype=np.float32)
                rec["poly3d"] = pts
                rec["_is_spline"] = False
                rec["id"] = f"lane_{self._panel._next_id:04d}"
                self._panel._next_id += 1

            # ── パネルに追加して描画を更新 ──────────────────────────
            self._panel._lanes.extend(new_records)
            self._gl.draw_lane_data = self._panel._lanes
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            self._gl.update()
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("annotations", True)
            self._has_unsaved_changes = True

            n = len(new_records)
            self._status.showMessage(
                t("status_prelabel_imported").format(n, Path(filepath).name))
            QMessageBox.information(
                self, t("menu_import_prelabel"),
                t("info_prelabel_done").format(n, filepath))

        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()
            print(f"[MainWindow] prelabel import error:\n{tb_str}")
            QMessageBox.critical(
                self, t("dialog_error"),
                f"{t('err_prelabel_import')} {e}\n\n{tb_str[-500:]}")
            self._status.showMessage(t("status_ready"))

    # ==================================================================
    # Z オーバーレイ
    # ==================================================================

    def _import_lane_csv(self) -> None:
        """lane.csv + timestamps_info.txt からマップ座標系の車線ポリラインを生成して追加する"""
        if self._panel is None or self._gl is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_panel_gl"))
            return

        # mapping_pose.txt が未読み込みなら警告して中断
        mapping_poses = getattr(self, '_mapping_poses', None) or []
        if not mapping_poses:
            QMessageBox.warning(
                self, t("dialog_warning"), t("warn_lane_csv_no_trajectory"))
            return

        # lane.csv を選択
        lane_csv_path, _ = QFileDialog.getOpenFileName(
            self, t("dlg_select_lane_csv"), "",
            "CSV Files (*.csv);;All Files (*)")
        if not lane_csv_path:
            return

        # timestamps_info.txt を選択
        # lane.csv と同じフォルダに timestamps_info.txt があれば自動検出
        default_ts = str(Path(lane_csv_path).parent / "timestamps_info.txt")
        if Path(default_ts).exists():
            ts_path = default_ts
        else:
            ts_path, _ = QFileDialog.getOpenFileName(
                self, t("dlg_select_timestamps"), str(Path(lane_csv_path).parent),
                "Text Files (*.txt);;All Files (*)")
            if not ts_path:
                return

        try:
            self._status.showMessage(t("status_loading"))
            QApplication.processEvents()

            from core.lane_csv_importer import load_lane_csv_as_polylines

            new_records = load_lane_csv_as_polylines(
                lane_csv_path=lane_csv_path,
                timestamps_path=ts_path,
                mapping_poses=mapping_poses,
            )

            if not new_records:
                QMessageBox.information(
                    self, t("dialog_info"), t("warn_prelabel_no_data"))
                return

            # poly3d を事前計算（直線接続 - スプライン補間しない）
            for rec in new_records:
                verts = rec.get("vertices", [])
                if len(verts) >= 2:
                    pts = np.array(
                        [(float(v[0]), float(v[1]),
                          float(v[2]) if len(v) > 2 else 0.0)
                         for v in verts], dtype=np.float32)
                    rec["poly3d"] = pts
                    rec["_is_spline"] = False
                else:
                    rec["poly3d"] = None

            # ID 付番して追加
            for rec in new_records:
                rec["id"] = f"lane_{self._panel._next_id:04d}"
                self._panel._next_id += 1

            self._panel._lanes.extend(new_records)
            self._gl.draw_lane_data = self._panel._lanes
            if hasattr(self._panel, '_refresh_table'):
                self._panel._refresh_table()
            if self._gl:
                self._gl.update()
            if self._layer_panel is not None:
                self._layer_panel.set_loaded("annotations", True)
            self._has_unsaved_changes = True

            n = len(new_records)
            self._status.showMessage(
                t("status_lane_csv_imported").format(n, Path(lane_csv_path).name))
            QMessageBox.information(
                self, t("menu_import_lane_csv"),
                t("info_lane_csv_done").format(n, lane_csv_path))

        except Exception as e:
            QMessageBox.critical(
                self, t("dialog_error"), f"{t('err_lane_csv_import')} {e}")
            import traceback
            traceback.print_exc()
            self._status.showMessage(t("status_ready"))

    def _on_overlay_threshold_changed(self, value: int) -> None:
        """Z 勾配オーバーレイの輝度閾値が変更されたときに適用する (Enter 確定式)"""
        if self._gl is None or not self._bev_overlay_path:
            return
        if not Path(self._bev_overlay_path).exists():
            return
        self._toolbar_progress.setValue(0)
        self._toolbar_progress.setVisible(True)
        QApplication.processEvents()
        try:
            self._toolbar_progress.setValue(30)
            QApplication.processEvents()
            self._gl.set_bev_overlay_with_threshold(self._bev_overlay_path, value)
            self._toolbar_progress.setValue(100)
            QApplication.processEvents()
        except Exception as e:
            print(f"[MainWindow] オーバーレイ閾値変更失敗: {e}")
        finally:
            self._toolbar_progress.setVisible(False)

    # ==================================================================
    # 色自動調整
    # ==================================================================

    def _auto_color_adjust(self) -> None:
        """BEV 画像の色を 95 パーセンタイル基準でガンマストレッチ調整する"""
        if self._gl is None or self._image_path is None:
            QMessageBox.warning(self, t("dialog_warning"), t("warn_no_bev_pixels"))
            return
        self._toolbar_progress.setValue(0)
        self._toolbar_progress.setVisible(True)
        QApplication.processEvents()
        try:
            import cv2
            gamma = self._gamma_spin.value()
            self._toolbar_progress.setValue(10)
            QApplication.processEvents()
            img_data = np.fromfile(self._image_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("画像の読み込みに失敗しました")
            img = cv2.flip(img, 0)  # FLIP_TOP_BOTTOM
            self._toolbar_progress.setValue(40)
            QApplication.processEvents()
            img_f = img.astype(np.float32)
            p95 = np.percentile(img_f[img_f > 0], 95) if np.any(img_f > 0) else 255.0
            if p95 > 0:
                img_f = img_f / p95
            img_f = np.clip(img_f, 0.0, 1.0)
            img_f = np.power(img_f, 1.0 / gamma)
            img_out = (img_f * 255.0).astype(np.uint8)
            self._toolbar_progress.setValue(80)
            QApplication.processEvents()
            from core.data_loader import DataLoader as _DL
            if _DL is not None:
                il = self._gl._bev_image
                if il is not None:
                    import cv2 as _cv2
                    img_rgba = _cv2.cvtColor(img_out, _cv2.COLOR_BGR2RGBA)
                    il._img_data = img_rgba
                    self._gl.makeCurrent()
                    il.upload_texture()
                    self._gl.doneCurrent()
                    self._gl.update()
            self._toolbar_progress.setValue(100)
            QApplication.processEvents()
            self._status.showMessage(t("status_auto_color_done"))
        except Exception as e:
            QMessageBox.critical(self, t("dialog_error"), f"{t('err_auto_color')} {e}")
            self._status.showMessage(t("status_ready"))
        finally:
            self._toolbar_progress.setVisible(False)

    # ==================================================================
    # 言語切り替え / UI 再翻訳
    # ==================================================================

    def _switch_language(self, lang: str) -> None:
        """言語を切り替えて UI を更新する"""
        set_language(lang)
        try:
            lang_file = _LOCALE_DIR / "lang.txt"
            lang_file.write_text(lang, encoding="utf-8")
        except Exception as e:
            print(f"[MainWindow] lang.txt save error: {e}")
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        """言語変更時に全 UI テキストを再翻訳する"""
        self.setWindowTitle(t("app_title"))
        # ファイルメニュー
        self._menu_file.setTitle(t("menu_file"))
        self._submenu_project.setTitle(t("menu_file_project"))
        self._act_project_new.setText(t("menu_project_new"))
        self._act_project_open.setText(t("menu_project_open"))
        self._act_project_save.setText(t("menu_project_save"))
        self._menu_recent.setTitle(t("menu_project_recent"))
        self._submenu_root.setTitle(t("menu_submenu_root"))
        self._act_open_root_folder.setText(t("menu_open_root_folder"))
        self._menu_recent_folders.setTitle(t("menu_recent_folders"))
        self._act_open_sample_pcd.setText(t("menu_open_sample_pcd"))
        self._submenu_camera.setTitle(t("menu_submenu_camera"))
        self._act_open_camera.setText(t("menu_open_camera"))
        self._submenu_anno.setTitle(t("menu_file_annotation_group"))
        self._submenu_anno_work.setTitle(t("menu_anno_workdata"))
        self._act_open_annotation.setText(t("menu_open_annotation"))
        self._act_save.setText(t("menu_save_annotations"))
        self._act_open_sample_json.setText(t("menu_open_sample_json"))
        self._act_open_sample_pkl.setText(t("menu_open_sample_pkl"))
        self._submenu_anno_output.setTitle(t("menu_anno_final_output"))
        self._act_export_gt_json.setText(t("menu_export_gt_json"))
        self._act_export_pkl.setText(t("menu_export_pkl"))
        self._act_exit.setText(t("menu_exit"))
        # ボタン
        self._btn_home_view.setText(t("btn_home_view"))
        self._btn_home_view.setToolTip(t("tooltip_home_view"))
        if hasattr(self, '_btn_fit_all'):
            self._btn_fit_all.setText(t("btn_fit_all"))
            self._btn_fit_all.setToolTip(t("tooltip_fit_all"))
        self._btn_auto_color.setText(t("btn_auto_color"))
        self._btn_auto_color.setToolTip(t("tooltip_auto_color"))
        self._gamma_spin.setToolTip(t("tooltip_gamma"))
        if hasattr(self, "_btn_lanegen_generate"):
            self._btn_lanegen_generate.setText(
                self._lanegen_ui_text("初期生成", "Generate")
            )
            self._btn_lanegen_generate.setToolTip(
                self._lanegen_ui_text(
                    "IBEV LaneGen Production V6で初期アノテーションを生成",
                    "Generate initial annotations with IBEV LaneGen Production V6",
                )
            )
        if hasattr(self, "_btn_lanegen_cancel"):
            self._btn_lanegen_cancel.setText(
                self._lanegen_ui_text("キャンセル", "Cancel")
            )
        # CVAT メニュー
        self._menu_cvat.setTitle(t("menu_cvat"))
        self._act_cvat_connect.setText(t("cvat_connect"))
        self._act_cvat_register.setText(t("cvat_register_tasks"))
        self._act_cvat_upload.setText(t("cvat_upload"))
        self._act_cvat_complete.setText(t("cvat_complete_job"))
        # 言語メニュー
        self._menu_lang.setTitle(t("menu_language"))
        self._act_lang_ja.setText(t("menu_lang_ja"))
        self._act_lang_en.setText(t("menu_lang_en"))
        # 品質チェックメニュー
        self._menu_quality.setTitle(t("menu_quality"))
        self._act_quality_run.setText(t("menu_quality_run"))
        self._act_quality_settings.setText(t("menu_quality_settings"))
        self._act_quality_export.setText(t("menu_quality_export"))
        # アノテーション支援メニュー
        self._menu_annotation.setTitle(t("menu_annotation"))
        self._act_auto_assign_all.setText(t("menu_auto_assign_all"))
        self._act_renumber_ids.setText(t("menu_renumber_ids"))
        self._act_auto_assign_n.setText(t("menu_auto_assign_n"))
        self._act_auto_assign_x.setText(t("menu_auto_assign_x"))
        self._act_auto_assign_y.setText(t("menu_auto_assign_y"))
        self._act_auto_boundary_ids.setText(t("menu_auto_boundary_ids"))
        self._act_suggest_split_merge.setText(t("menu_suggest_split_merge"))
        # ステータス
        self._status.showMessage(t("status_ready"))
        # LaneLinePanel と LayerPanel の再翻訳
        if self._panel is not None:
            self._panel._retranslate_ui()
        if self._layer_panel is not None and hasattr(self._layer_panel, 'retranslate'):
            self._layer_panel.retranslate()

    def closeEvent(self, event) -> None:
        """右上の×で終了するとき、生成中ジョブと作業データを安全に扱う。

        Yes:
            現在の作業アノテーションを保存してから終了する。
            geometry整合性エラーやローカル保存失敗時は終了を中止する。
        No:
            保存せず終了する。

        従来のCVAT/ローカル別の終了確認は一本化し、
        必ずこの確認ダイアログを1回だけ表示する。
        """
        runner = getattr(self, "_lanegen_runner", None)
        if runner is not None and runner.is_running:
            answer = QMessageBox.question(
                self,
                self._lanegen_ui_text("生成実行中", "Generation running"),
                self._lanegen_ui_text(
                    "LaneGenを安全にキャンセルしてから終了します。よろしいですか？",
                    "Cancel LaneGen safely before closing?",
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            runner.request_cancel()
            self._status.showMessage(
                self._lanegen_ui_text(
                    "LaneGen停止待ちです。停止後にもう一度閉じてください。",
                    "Waiting for LaneGen to stop. Close again after cancellation.",
                )
            )
            event.ignore()
            return

        reply = QMessageBox.question(
            self,
            "終了確認",
            "作業アノテーションデータを保存して終了しますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.No:
            event.accept()
            return

        # ── Yes: 保存前の最終geometry整合性チェック ──────────────
        panel = getattr(self, "_panel", None)
        normalize = getattr(
            panel,
            "normalize_all_geometries",
            None,
        )
        if callable(normalize):
            try:
                if not normalize(show_error=True):
                    # 不正geometryがある場合はデータ保護のため閉じない。
                    event.ignore()
                    return
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "保存エラー",
                    "アノテーションの整合性確認に失敗したため、"
                    "終了を中止します。\n"
                    f"{exc}",
                )
                event.ignore()
                return

        # ローカル保存では、成功すると _has_unsaved_changes=False になる。
        is_local_save = not (
            self._cvat_client is not None
            and self._cvat_task_id is not None
        )

        try:
            self._save_annotations()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "保存エラー",
                "作業アノテーションの保存に失敗したため、"
                "終了を中止します。\n"
                f"{exc}",
            )
            event.ignore()
            return

        # ローカル保存で未保存フラグが残っている場合は、
        # 保存ダイアログのキャンセル・保存失敗などと判断し、終了しない。
        if is_local_save and bool(
            getattr(self, "_has_unsaved_changes", False)
        ):
            self._status.showMessage(
                "作業アノテーションが保存されていないため、終了を中止しました。"
            )
            event.ignore()
            return

        event.accept()
