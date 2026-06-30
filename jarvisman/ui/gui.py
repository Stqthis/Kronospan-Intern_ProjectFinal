from __future__ import annotations

from jarvisman.ui.theme_manager import ThemeManager

import html
import sys
from typing import Optional

from PyQt6.QtGui import QClipboard

from PyQt6.QtCore import Qt, QThreadPool, QTimer, QUrl
from PyQt6.QtGui import QImage, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from jarvisman import config as cfg
from jarvisman.agent import Agent
from jarvisman.llm.ollama_client import OllamaClient
from jarvisman.retrieval.rag import RAGPipeline
from jarvisman.retrieval.vector_store import VectorStore
from jarvisman.ui.workers import Worker
from jarvisman.ui.response_handler import ResponseDataExtractor
from jarvisman.ui.chart_selector import ChartTypeSelector, DataSummaryPanel

# --------------------------------------------------------------------------- #
# Theme: one palette drives the whole window via a Qt style sheet.            #
# --------------------------------------------------------------------------- #
THEME = {
    "bg": "#0f1419", 
    "panel": "#171c24", 
    "panel2": "#1f262f",
    "border": "#2a323d", 
    "text": "#e6edf3", 
    "muted": "#8b97a5",
    "accent": "#4c9aff", 
    "accent_dim": "#2d5a88", 
    "bot": "#1f262f",
    "ok": "#3fb950", 
    "err": "#f85149", 
    "warn": "#d29922", 
    "code_bg": "#11151a",
    "grid": "#2a323d",
    "cycle": [                          # ← ADD THIS WHOLE SECTION
        "#4c9aff", "#3fb950", "#d29922", "#f85149", "#a371f7",
        "#39c5cf", "#db61a2", "#e3b341", "#7ee787", "#ffa657"
    ]
}

# back-compat aliases (other modules/tests import these)
USER_COLOR = THEME["accent"]
BOT_COLOR = THEME["ok"]
MUTED_COLOR = THEME["muted"]
ERR_COLOR = THEME["err"]


_UI = {
    "en": {
        "how": "&#9656; How this was computed", "choose": "Choose one:",
        "title": "Offline Servant", "models": "Models",
        "chat_model": "General / chat model",
        "code_model": "Code model (plans & queries)",
        "understanding_model": "Understanding model (reasoning)",
        "embed_model": "Embedding model (PDF search)",
        "auto": "(auto: use general model)", "refresh": "Refresh model list",
        "documents": "Documents", "add": "Add PDF / Excel \u2026",
        "build": "Build / Update Index", "load": "Load Saved Index",
        "clear": "Clear", "new_chat": "New chat", "index_empty": "Index: empty", "send": "Send",
        "placeholder": "Ask about your data, or request a chart \u2026",
        "welcome_title": "Welcome",
        "welcome_body": ("Add PDF or Excel files on the left, build the index, "
                         "then ask questions in plain language. Everything runs "
                         "locally through Ollama \u2014 nothing leaves this machine."),
        "ready": "Ready.", "settings": "Toggle settings",
        "you": "You", "assistant": "Assistant",
        "advanced": "Advanced",
        "embed_warn": "Not an embedding model \u2014 PDF search will be poor.",
        "badge_no_index": "No data loaded",
        "badge_index": "{tables} table(s), {chunks} chunk(s)",
        "hero_no_index": "Add a PDF or Excel on the left, then build the index.",
        "greeting": "I am {name}.",
        "greeting_sub_named": "How may I help you, {who}?",
        "greeting_sub_plain": "How may I help you?",
        "suggest_title": "Try asking",
        "suggestions": [
            "Total funds in EUR by company",
            "Breakdown of amounts per currency",
            "Sort interest rates, highest first",
            "Directorships as at 31/12/2024",
        ],
        "visualize": "\U0001F4CA  Visualize / breakdown",
        "thinking": "{name} is thinking \u2026",
    },
    "el": {
        "how": "&#9656; \u03a0\u03ce\u03c2 \u03c5\u03c0\u03bf\u03bb\u03bf\u03b3\u03af\u03c3\u03c4\u03b7\u03ba\u03b5",
        "choose": "\u0394\u03b9\u03ac\u03bb\u03b5\u03be\u03b5 \u03ad\u03bd\u03b1:",
        "title": "Offline Servant", "models": "\u039c\u03bf\u03bd\u03c4\u03ad\u03bb\u03b1",
        "chat_model": "\u0393\u03b5\u03bd\u03b9\u03ba\u03cc \u03bc\u03bf\u03bd\u03c4\u03ad\u03bb\u03bf",
        "code_model": "\u039c\u03bf\u03bd\u03c4\u03ad\u03bb\u03bf \u03ba\u03ce\u03b4\u03b9\u03ba\u03b1",
        "understanding_model": "\u039c\u03bf\u03bd\u03c4\u03ad\u03bb\u03bf \u03ba\u03b1\u03c4\u03b1\u03bd\u03cc\u03b7\u03c3\u03b7\u03c2",
        "embed_model": "\u039c\u03bf\u03bd\u03c4\u03ad\u03bb\u03bf \u03b1\u03bd\u03b1\u03b6\u03ae\u03c4\u03b7\u03c3\u03b7\u03c2 PDF",
        "auto": "(\u03b1\u03c5\u03c4\u03cc\u03bc\u03b1\u03c4\u03bf)", "refresh": "\u0391\u03bd\u03b1\u03bd\u03ad\u03c9\u03c3\u03b7",
        "documents": "\u0388\u03b3\u03b3\u03c1\u03b1\u03c6\u03b1", "add": "\u03a0\u03c1\u03bf\u03c3\u03b8\u03ae\u03ba\u03b7 \u2026",
        "build": "\u0394\u03b7\u03bc\u03b9\u03bf\u03c5\u03c1\u03b3\u03af\u03b1 \u0395\u03c5\u03c1\u03b5\u03c4\u03b7\u03c1\u03af\u03bf\u03c5",
        "load": "\u03a6\u03cc\u03c1\u03c4\u03c9\u03c3\u03b7", "clear": "\u039a\u03b1\u03b8\u03b1\u03c1\u03b9\u03c3\u03bc\u03cc\u03c2", "new_chat": "\u039d\u03ad\u03b1 \u03c3\u03c5\u03b6\u03ae\u03c4\u03b7\u03c3\u03b7",
        "index_empty": "\u0395\u03c5\u03c1\u03b5\u03c4\u03ae\u03c1\u03b9\u03bf: \u03ba\u03b5\u03bd\u03cc", "send": "\u0391\u03c0\u03bf\u03c3\u03c4\u03bf\u03bb\u03ae",
        "placeholder": "\u03a1\u03c9\u03c4\u03ae\u03c3\u03c4\u03b5 \u03b3\u03b9\u03b1 \u03c4\u03b1 \u03b4\u03b5\u03b4\u03bf\u03bc\u03ad\u03bd\u03b1 \u03c3\u03b1\u03c2 \u2026",
        "welcome_title": "\u039a\u03b1\u03bb\u03ce\u03c2 \u03ae\u03c1\u03b8\u03b1\u03c4\u03b5",
        "welcome_body": ("\u03a0\u03c1\u03bf\u03c3\u03b8\u03ad\u03c3\u03c4\u03b5 \u03b1\u03c1\u03c7\u03b5\u03af\u03b1, \u03c6\u03c4\u03b9\u03ac\u03be\u03c4\u03b5 \u03c4\u03bf \u03b5\u03c5\u03c1\u03b5\u03c4\u03ae\u03c1\u03b9\u03bf "
                         "\u03ba\u03b1\u03b9 \u03c1\u03c9\u03c4\u03ae\u03c3\u03c4\u03b5. \u038c\u03bb\u03b1 \u03c4\u03c1\u03ad\u03c7\u03bf\u03c5\u03bd \u03c4\u03bf\u03c0\u03b9\u03ba\u03ac."),
        "ready": "\u0388\u03c4\u03bf\u03b9\u03bc\u03bf.", "settings": "\u03a1\u03c5\u03b8\u03bc\u03af\u03c3\u03b5\u03b9\u03c2",
        "you": "\u0395\u03c3\u03b5\u03af\u03c2", "assistant": "\u0392\u03bf\u03b7\u03b8\u03cc\u03c2",
        "advanced": "\u0393\u03b9\u03b1 \u03c0\u03c1\u03bf\u03c7\u03c9\u03c1\u03b7\u03bc\u03ad\u03bd\u03bf\u03c5\u03c2",
        "embed_warn": "\u0394\u03b5\u03bd \u03b5\u03af\u03bd\u03b1\u03b9 embedding \u03bc\u03bf\u03bd\u03c4\u03ad\u03bb\u03bf \u2014 \u03b7 \u03b1\u03bd\u03b1\u03b6\u03ae\u03c4\u03b7\u03c3\u03b7 PDF \u03b8\u03b1 \u03b5\u03af\u03bd\u03b1\u03b9 \u03c6\u03c4\u03c9\u03c7\u03ae.",
        "badge_no_index": "\u03a7\u03c9\u03c1\u03af\u03c2 \u03b4\u03b5\u03b4\u03bf\u03bc\u03ad\u03bd\u03b1",
        "badge_index": "{tables} \u03c0\u03af\u03bd\u03b1\u03ba\u03b5\u03c2, {chunks} chunk(s)",
        "hero_no_index": "\u03a0\u03c1\u03bf\u03c3\u03b8\u03ad\u03c3\u03c4\u03b5 PDF \u03ae Excel \u03b1\u03c1\u03b9\u03c3\u03c4\u03b5\u03c1\u03ac \u03ba\u03b1\u03b9 \u03c6\u03c4\u03b9\u03ac\u03be\u03c4\u03b5 \u03c4\u03bf \u03b5\u03c5\u03c1\u03b5\u03c4\u03ae\u03c1\u03b9\u03bf.",
        "greeting": "\u0395\u03af\u03bc\u03b1\u03b9 \u03bf {name}.",
        "greeting_sub_named": "\u03a0\u03ce\u03c2 \u03bc\u03c0\u03bf\u03c1\u03ce \u03bd\u03b1 \u03c3\u03b1\u03c2 \u03b2\u03bf\u03b7\u03b8\u03ae\u03c3\u03c9, {who};",
        "greeting_sub_plain": "\u03a0\u03ce\u03c2 \u03bc\u03c0\u03bf\u03c1\u03ce \u03bd\u03b1 \u03c3\u03b1\u03c2 \u03b2\u03bf\u03b7\u03b8\u03ae\u03c3\u03c9;",
        "suggest_title": "\u0394\u03bf\u03ba\u03b9\u03bc\u03ac\u03c3\u03c4\u03b5",
        "suggestions": [
            "\u03a3\u03cd\u03bd\u03bf\u03bb\u03bf \u03ba\u03b5\u03c6\u03b1\u03bb\u03b1\u03af\u03c9\u03bd \u03c3\u03b5 EUR \u03b1\u03bd\u03ac \u03b5\u03c4\u03b1\u03b9\u03c1\u03b5\u03af\u03b1",
            "\u0391\u03bd\u03ac\u03bb\u03c5\u03c3\u03b7 \u03c0\u03bf\u03c3\u03ce\u03bd \u03b1\u03bd\u03ac \u03bd\u03cc\u03bc\u03b9\u03c3\u03bc\u03b1",
            "\u03a4\u03b1\u03be\u03b9\u03bd\u03cc\u03bc\u03b7\u03c3\u03b7 \u03b5\u03c0\u03b9\u03c4\u03bf\u03ba\u03af\u03c9\u03bd, \u03c5\u03c8\u03b7\u03bb\u03cc\u03c4\u03b5\u03c1\u03bf \u03c0\u03c1\u03ce\u03c4\u03bf",
            "\u0394\u03b9\u03b5\u03c5\u03b8\u03c5\u03bd\u03c4\u03ad\u03c2 \u03c9\u03c2 31/12/2024",
        ],
        "visualize": "\U0001F4CA  \u0393\u03c1\u03ac\u03c6\u03b7\u03bc\u03b1 / \u03b1\u03bd\u03ac\u03bb\u03c5\u03c3\u03b7",
        "thinking": "\u039f {name} \u03c3\u03ba\u03ad\u03c6\u03c4\u03b5\u03c4\u03b1\u03b9 \u2026",
    },
}


def _t(key: str) -> str:
    lang = _UI.get(cfg.UI_LANG, _UI["en"])
    return lang.get(key, _UI["en"].get(key, key))


def _qss() -> str:
    T = THEME
    return f"""
    QMainWindow, QWidget {{ background: {T['bg']}; color: {T['text']};
        font-size: 14px; }}
    QLabel {{ color: {T['text']}; }}
    QLabel#muted {{ color: {T['muted']}; font-size: 12px; }}
    QLabel#heading {{ color: {T['text']}; font-size: 12px; font-weight: 700;
        letter-spacing: 1px; }}
    QFrame#card {{ background: {T['panel']}; border: 1px solid {T['border']};
        border-radius: 12px; }}
    QFrame#sep {{ background: {T['border']}; max-height: 1px; border: none; }}
    QComboBox, QLineEdit {{ background: {T['panel2']}; color: {T['text']};
        border: 1px solid {T['border']}; border-radius: 8px;
        padding: 7px 10px; selection-background-color: {T['accent_dim']}; }}
    QComboBox:focus, QLineEdit:focus {{ border: 1px solid {T['accent']}; }}
    QComboBox QAbstractItemView {{ background: {T['panel2']};
        color: {T['text']}; selection-background-color: {T['accent_dim']};
        border: 1px solid {T['border']}; outline: none; }}
    QPushButton {{ background: {T['panel2']}; color: {T['text']};
        border: 1px solid {T['border']}; border-radius: 8px;
        padding: 8px 14px; }}
    QPushButton:hover {{ border: 1px solid {T['accent']}; }}
    QPushButton:disabled {{ color: {T['muted']}; }}
    QPushButton#primary {{ background: {T['accent']}; color: #06121f;
        border: none; font-weight: 700; }}
    QPushButton#primary:hover {{ background: #5aa6ff; }}
    QPushButton#primary:disabled {{ background: {T['panel2']};
        color: {T['muted']}; }}
    QToolButton {{ background: transparent; color: {T['muted']};
        border: none; font-size: 18px; padding: 2px 6px; }}
    QToolButton:hover {{ color: {T['accent']}; }}
    QListWidget {{ background: {T['panel2']}; color: {T['text']};
        border: 1px solid {T['border']}; border-radius: 8px; padding: 4px; }}
    QListWidget::item {{ padding: 5px 6px; border-radius: 6px; }}
    QListWidget::item:selected {{ background: {T['accent_dim']}; }}
    QTextBrowser {{ background: {T['bg']}; color: {T['text']};
        border: none; font-size: 14px; }}
    QProgressBar {{ background: {T['panel2']}; border: 1px solid {T['border']};
        border-radius: 6px; height: 6px; }}
    QProgressBar::chunk {{ background: {T['accent']}; border-radius: 6px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {T['border']};
        border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {T['muted']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QStatusBar {{ background: {T['panel']}; color: {T['muted']}; }}
    QSplitter::handle {{ background: {T['border']}; }}
    QPushButton#chip {{ background: {T['panel2']}; color: {T['text']};
        border: 1px solid {T['border']}; border-radius: 16px;
        padding: 9px 16px; text-align: center; }}
    QPushButton#chip:hover {{ border: 1px solid {T['accent']};
        color: {T['accent']}; }}
    QToolButton#advanced {{ color: {T['muted']}; font-size: 12px;
        padding: 2px 0; }}
    QToolButton#advanced:hover {{ color: {T['accent']}; }}
    QLabel#badge {{ color: {T['muted']}; font-size: 11px;
        background: {T['panel']}; border: 1px solid {T['border']};
        border-radius: 10px; padding: 3px 10px; }}
    QLabel#warn {{ color: {T['warn']}; font-size: 11px; }}
    QLabel#hero {{ color: {T['text']}; font-size: 25px; font-weight: 800; }}
    QLabel#herosub {{ color: {T['muted']}; font-size: 16px; }}
    QLabel#heromark {{ color: {T['accent']}; font-size: 30px; }}
    QLabel#suggest {{ color: {T['muted']}; font-size: 12px; }}
    QLabel#typing {{ color: {T['accent']}; font-size: 12px; }}
    """


class MainWindow(QMainWindow):
    _prov_seq = 0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{cfg.ASSISTANT_NAME} \u2014 " + _t("title"))
        self.resize(1180, 760)

        self.pool = QThreadPool()
        self.busy = False
        self._img_counter = 0
        self.pending_files: list[str] = []
        self._pending_options: list[str] = []
        self._prov_store: dict[str, str] = {}
        self._chart_store: dict[str, tuple] = {}   # key -> (title, headers, rows)
        self._chart_windows: list = []             # keep dialog refs alive
        self._chart_seq = 0
        self._buf = None              # when a list, _append_html buffers into it
        self._sidebar_visible = True
        self.is_dark_theme = True
        self.theme_manager = ThemeManager()
        self.current_theme = self.theme_manager.get_theme(self.is_dark_theme)

        self.ollama = OllamaClient(cfg.OLLAMA_HOST)
        self.store = VectorStore()
        self.rag = RAGPipeline(self.ollama, self.store, cfg.DEFAULT_CHAT_MODEL,
                               cfg.DEFAULT_EMBED_MODEL)
        self.agent = Agent(self.ollama, self.rag, cfg.DEFAULT_CHAT_MODEL)

        self.setStyleSheet(self._get_stylesheet())
        self._build_ui()
        self._startup_checks()
        self.conversation_history = []  # Store all messages

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sidebar = self._build_left_panel()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self._build_chat_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([340, 840])
        self.splitter.setHandleWidth(1)
        self.setCentralWidget(self.splitter)
        self.status = self.statusBar()
        self.status.showMessage(_t("ready"))

    def _card(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        return card, lay

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(14, 14, 10, 14)
        outer.setSpacing(12)

        brand = QHBoxLayout()
        mark = QLabel("\u25c6")
        mark.setStyleSheet(f"color:{THEME['accent']}; font-size:18px;")
        names = QVBoxLayout()
        names.setSpacing(0)
        title = QLabel(cfg.ASSISTANT_NAME)
        title.setStyleSheet("font-size:16px; font-weight:800;")
        subtitle = QLabel(_t("title"))
        subtitle.setObjectName("muted")
        names.addWidget(title)
        names.addWidget(subtitle)
        self.health_dot = QLabel("\u25cf")
        self.health_dot.setStyleSheet(f"color:{THEME['muted']}; font-size:12px;")
        self.health_dot.setToolTip("Ollama status")
        brand.addWidget(mark)
        brand.addLayout(names)
        brand.addStretch(1)
        brand.addWidget(self.health_dot)
        outer.addLayout(brand)

        card, lay = self._card()
        lay.addWidget(self._heading(_t("models")))
        lay.addWidget(self._field_label(_t("chat_model")))
        self.chat_combo = self._combo(cfg.DEFAULT_CHAT_MODEL,
                                      self._on_chat_model_changed)
        lay.addWidget(self.chat_combo)

        self.advanced_btn = QToolButton()
        self.advanced_btn.setObjectName("advanced")
        self.advanced_btn.setText("\u25b8 " + _t("advanced"))
        self.advanced_btn.clicked.connect(self._toggle_advanced)
        lay.addWidget(self.advanced_btn)

        self.advanced_box = QWidget()
        adv = QVBoxLayout(self.advanced_box)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(8)
        adv.addWidget(self._field_label(_t("code_model")))
        self.code_combo = self._combo(cfg.CODE_MODEL or "",
                                      self._on_code_model_changed, True)
        adv.addWidget(self.code_combo)
        adv.addWidget(self._field_label(_t("understanding_model")))
        self.understanding_combo = self._combo(
            cfg.UNDERSTANDING_MODEL or "",
            self._on_understanding_model_changed, True)
        adv.addWidget(self.understanding_combo)
        adv.addWidget(self._field_label(_t("embed_model")))
        self.embed_combo = self._combo(cfg.DEFAULT_EMBED_MODEL,
                                       self._on_embed_model_changed)
        adv.addWidget(self.embed_combo)
        self.embed_warn = QLabel(_t("embed_warn"))
        self.embed_warn.setObjectName("warn")
        self.embed_warn.setWordWrap(True)
        self.embed_warn.setVisible(False)
        adv.addWidget(self.embed_warn)
        self.advanced_box.setVisible(False)
        lay.addWidget(self.advanced_box)

        self.refresh_btn = QPushButton(_t("refresh"))
        self.refresh_btn.clicked.connect(self._load_models)
        lay.addWidget(self.refresh_btn)
        outer.addWidget(card)
        self._update_embed_warning(self.embed_combo.currentText())

        card2, lay2 = self._card()
        lay2.addWidget(self._heading(_t("documents")))
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(120)
        lay2.addWidget(self.file_list, stretch=1)
        self.add_btn = QPushButton(_t("add"))
        self.add_btn.clicked.connect(self._choose_files)
        lay2.addWidget(self.add_btn)
        self.build_btn = QPushButton(_t("build"))
        self.build_btn.setObjectName("primary")
        self.build_btn.clicked.connect(self._build_index)
        lay2.addWidget(self.build_btn)
        load_clear = QHBoxLayout()
        self.load_btn = QPushButton(_t("load"))
        self.load_btn.clicked.connect(self._load_index)
        self.clear_btn = QPushButton(_t("clear"))
        self.clear_btn.clicked.connect(self._clear_index)
        load_clear.addWidget(self.load_btn)
        load_clear.addWidget(self.clear_btn)
        lay2.addLayout(load_clear)
        self.index_label = QLabel(_t("index_empty"))
        self.index_label.setObjectName("muted")
        lay2.addWidget(self.index_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        lay2.addWidget(self.progress)
        outer.addWidget(card2, stretch=1)
        return panel

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        
        header = QHBoxLayout()
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("\u2630")
        self.toggle_btn.setToolTip(_t("settings"))
        self.toggle_btn.clicked.connect(self._toggle_sidebar)
        header.addWidget(self.toggle_btn)
        
        # ADD THEME TOGGLE BUTTON
        self.theme_btn = QToolButton()
        self.theme_btn.setText("🌙")
        self.theme_btn.setToolTip("Toggle theme")
        self.theme_btn.clicked.connect(self._toggle_theme)
        header.addWidget(self.theme_btn)
        
        header.addStretch(1)
        self.typing_label = QLabel("")
        self.typing_label.setObjectName("typing")
        header.addWidget(self.typing_label)
        self.model_badge = QLabel("")
        self.model_badge.setObjectName("badge")
        header.addWidget(self.model_badge)
        self.index_badge = QLabel("")
        self.index_badge.setObjectName("badge")
        header.addWidget(self.index_badge)
        layout.addLayout(header)
        
        self.welcome = self._build_welcome_widget()
        layout.addWidget(self.welcome, stretch=1)
        
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setOpenLinks(False)
        self.chat.anchorClicked.connect(self._on_anchor)
        self.chat.setVisible(False)
        
        # Chart visualization widgets (NEW)
        self.data_summary = DataSummaryPanel(theme=self.current_theme)
        self.data_summary.setVisible(False)
        
        self.chart_selector = ChartTypeSelector(parent=self, theme=self.current_theme)
        self.chart_selector.visualization_requested.connect(self._on_chart_visualization_requested)
        self.chart_selector.setVisible(False)
        
        # Create layout with chat and chart widgets
        chat_layout = QVBoxLayout()
        chat_layout.addWidget(self.chat, stretch=1)
        chat_layout.addWidget(self.data_summary)
        chat_layout.addWidget(self.chart_selector)
        layout.addLayout(chat_layout)
        
        row = QHBoxLayout()
        row.setSpacing(8)
        self.input = QLineEdit()
        self.input.setPlaceholderText(_t("placeholder"))
        self.input.setMinimumHeight(42)
        self.input.returnPressed.connect(self._send)
        self.send_btn = QPushButton(_t("send"))
        self.send_btn.setObjectName("primary")
        self.send_btn.setMinimumHeight(42)
        self.send_btn.setMinimumWidth(90)
        self.send_btn.clicked.connect(self._send)
        self.newchat_btn = QPushButton(_t("new_chat"))
        self.newchat_btn.setMinimumHeight(42)
        self.newchat_btn.clicked.connect(self._new_chat)
        row.addWidget(self.input, stretch=1)
        row.addWidget(self.send_btn)
        row.addWidget(self.newchat_btn)
        layout.addLayout(row)
        
        self._update_welcome()
        self._update_badges()
        return panel


    def _build_welcome_widget(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(20, 40, 20, 20)
        outer.addStretch(1)

        mark = QLabel("\u25c6")
        mark.setObjectName("heromark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(mark)

        self.hero = QLabel(_t("greeting").format(name=cfg.ASSISTANT_NAME))
        self.hero.setObjectName("hero")
        self.hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.hero)

        self.hero_sub = QLabel("")
        self.hero_sub.setObjectName("herosub")
        self.hero_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_sub.setWordWrap(True)
        outer.addWidget(self.hero_sub)
        outer.addSpacing(18)

        self.suggest_label = QLabel(_t("suggest_title"))
        self.suggest_label.setObjectName("suggest")
        self.suggest_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.suggest_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.chip_buttons = []
        suggestions = _t("suggestions")
        for i, text in enumerate(suggestions):
            btn = QPushButton(text)
            btn.setObjectName("chip")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, t=text: self._use_suggestion(t))
            self.chip_buttons.append(btn)
            grid.addWidget(btn, i // 2, i % 2)
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addLayout(grid)
        holder.addStretch(1)
        outer.addLayout(holder)
        outer.addStretch(2)
        return w

    @staticmethod
    def _heading(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("heading")
        return lbl

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("muted")
        return lbl

    def _combo(self, initial: str, slot, allow_auto: bool = False) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        if allow_auto:
            combo.addItem(_t("auto"))
        if initial:
            combo.addItem(initial)
            combo.setCurrentText(initial)
        elif allow_auto:
            combo.setCurrentText(_t("auto"))
        combo.currentTextChanged.connect(slot)
        return combo

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("sep")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _toggle_sidebar(self) -> None:
        self._sidebar_visible = not self._sidebar_visible
        self.sidebar.setVisible(self._sidebar_visible)

    def _toggle_advanced(self) -> None:
        show = not self.advanced_box.isVisible()
        self.advanced_box.setVisible(show)
        arrow = "\u25be" if show else "\u25b8"
        self.advanced_btn.setText(f"{arrow} " + _t("advanced"))
    
    def _toggle_theme(self) -> None:
        """Toggle between dark and light theme."""
        self.is_dark_theme = not self.is_dark_theme
        self.current_theme = self.theme_manager.get_theme(self.is_dark_theme)
        
        T = self.current_theme
        
        # Update button
        self.theme_btn.setText("☀️" if self.is_dark_theme else "🌙")
        
        # Update main stylesheet
        self.setStyleSheet(self._get_stylesheet())
        
        # Get current HTML
        html_content = self.chat.toHtml()
        
        if html_content:
            # Replace theme colors in the HTML
            old_theme = self.theme_manager.get_theme(not self.is_dark_theme)
            
            # Replace old colors with new colors
            html_content = html_content.replace(old_theme['bg'], T['bg'])
            html_content = html_content.replace(old_theme['text'], T['text'])
            html_content = html_content.replace(old_theme['panel'], T['panel'])
            html_content = html_content.replace(old_theme['accent'], T['accent'])
            html_content = html_content.replace(old_theme['accent_dim'], T['accent_dim'])
            html_content = html_content.replace(old_theme['panel2'], T['panel2'])
            html_content = html_content.replace(old_theme['border'], T['border'])
            
            # Re-apply the updated HTML
            self.chat.setHtml(html_content)
        
        # Update chat widget COLORS WITHOUT clearing text
        self.chat.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: {T['bg']};
                color: {T['text']};
                border: none;
                font-size: 14px;
            }}
            """
        )
        
        # Update input box
        self.input.setStyleSheet(
            f"""
            QLineEdit {{
                background: {T['panel2']};
                color: {T['text']};
                border: 1px solid {T['border']};
                border-radius: 8px;
                padding: 7px 10px;
            }}
            """
        )
        
        # Update sidebar
        self.sidebar.setStyleSheet(f"QWidget {{ background: {T['bg']}; }}")
        
        # Force repaint
        self.repaint()
        self.update()



    @staticmethod
    def _looks_like_embedder(name: str) -> bool:
        n = (name or "").lower()
        return any(h in n for h in ("embed", "nomic", "bge", "gte", "e5",
                                    "minilm", "mxbai", "arctic"))

    def _update_embed_warning(self, name: str) -> None:
        if not hasattr(self, "embed_warn"):
            return
        bad = bool(name) and name != _t("auto") \
            and not self._looks_like_embedder(name)
        self.embed_warn.setVisible(bad)

    def _has_data(self) -> bool:
        return bool(self.agent.dataframes) or self.store.count > 0

    def _update_welcome(self) -> None:
        honor = (cfg.USER_HONORIFIC or "").strip()
        if self._has_data():
            sub = (_t("greeting_sub_named").format(who=honor) if honor
                   else _t("greeting_sub_plain"))
        else:
            sub = _t("hero_no_index")
        self.hero_sub.setText(sub)
        for btn in getattr(self, "chip_buttons", []):
            btn.setVisible(self._has_data())
        if hasattr(self, "suggest_label"):
            self.suggest_label.setVisible(self._has_data())

    def _update_badges(self) -> None:
        if not hasattr(self, "model_badge"):
            return
        model = self.agent.chat_model or cfg.DEFAULT_CHAT_MODEL
        self.model_badge.setText("\u25c6 " + model)
        chunks, tables = self.store.count, len(self.agent.dataframes)
        if chunks == 0 and tables == 0:
            self.index_badge.setText(_t("badge_no_index"))
        else:
            self.index_badge.setText(
                _t("badge_index").format(tables=tables, chunks=chunks))

    def _use_suggestion(self, text: str) -> None:
        if self.busy:
            return
        self.input.setText(text)
        self._send()

    def _show_chat(self) -> None:
        if self.welcome.isVisible():
            self.welcome.setVisible(False)
            self.chat.setVisible(True)

    # typing indicator
    def _start_typing(self) -> None:
        from time import time
        self._typing_dots = 0
        self._typing_start_time = time()  # ← ADD THIS
        if not hasattr(self, "_typing_timer"):
            self._typing_timer = QTimer(self)
            self._typing_timer.timeout.connect(self._tick_typing)
        self._typing_timer.start(400)
        self._tick_typing()

    def _tick_typing(self) -> None:
        self._typing_dots = (self._typing_dots + 1) % 4
        
        # Calculate elapsed time if timer started
        elapsed = ""
        if hasattr(self, '_typing_start_time'):
            from time import time
            elapsed_sec = int(time() - self._typing_start_time)
            if elapsed_sec > 0:
                elapsed = f" ({elapsed_sec}s)"
        
        base = _t("thinking").format(name=cfg.ASSISTANT_NAME).rstrip(" .\u2026")
        self.typing_label.setText(base + " " + "\u00b7" * self._typing_dots + elapsed)

    def _stop_typing(self) -> None:
        if hasattr(self, "_typing_timer"):
            self._typing_timer.stop()
        self.typing_label.setText("")

    # ------------------------------------------------------------------ #
    def _startup_checks(self) -> None:
        worker = Worker(self.ollama.is_alive)
        worker.signals.result.connect(self._on_health)
        self.pool.start(worker)

    def _on_health(self, alive: bool) -> None:
        if alive:
            self.health_dot.setStyleSheet(f"color:{THEME['ok']}; font-size:12px;")
            self.health_dot.setToolTip("Connected to Ollama")
            self._system_line(_t("ready"), THEME["ok"])
            self._load_models()
        else:
            self.health_dot.setStyleSheet(f"color:{THEME['err']}; font-size:12px;")
            self.health_dot.setToolTip("Ollama not reachable")
            self._system_line(
                f"Could not reach Ollama at {html.escape(cfg.OLLAMA_HOST)}. "
                f"Start it with 'ollama serve', then Refresh.", THEME["err"])

    def _load_models(self) -> None:
        worker = Worker(self.ollama.list_models)
        worker.signals.result.connect(self._populate_models)
        worker.signals.error.connect(self._show_error)
        self.pool.start(worker)

    def _populate_models(self, models: list[str]) -> None:
        if not models:
            return
        combos = [
            (self.chat_combo, cfg.DEFAULT_CHAT_MODEL, False),
            (self.code_combo, cfg.CODE_MODEL, True),
            (self.understanding_combo, cfg.UNDERSTANDING_MODEL, True),
            (self.embed_combo, cfg.DEFAULT_EMBED_MODEL, False),
        ]
        for combo, default, allow_auto in combos:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if allow_auto:
                combo.addItem(_t("auto"))
            combo.addItems(models)
            if current and current != _t("auto") and current in models:
                combo.setCurrentText(current)
            elif default and default in models:
                combo.setCurrentText(default)
            elif allow_auto:
                combo.setCurrentText(_t("auto"))
            else:
                combo.setCurrentText(models[0])
            combo.blockSignals(False)
        self.status.showMessage(f"{len(models)} local models available.", 5000)
        self._update_badges()
        self._update_embed_warning(self.embed_combo.currentText())

    # ------------------------------------------------------------------ #
    def _on_chat_model_changed(self, name: str) -> None:
        if name and name != _t("auto"):
            self.rag.chat_model = name
            self.agent.chat_model = name
            self._update_badges()

    def _on_code_model_changed(self, name: str) -> None:
        cfg.CODE_MODEL = "" if (not name or name == _t("auto")) else name

    def _on_understanding_model_changed(self, name: str) -> None:
        cfg.UNDERSTANDING_MODEL = "" if (not name or name == _t("auto")) else name

    def _on_embed_model_changed(self, name: str) -> None:
        self._update_embed_warning(name)
        if not name or name == _t("auto"):
            return
        self.rag.embed_model = name
        if self.store.count > 0 and self.store.embed_model \
                and self.store.embed_model != name:
            self._system_line(
                f"Embedding model changed to {html.escape(name)}. The current "
                f"index was built with {html.escape(self.store.embed_model)}; "
                f"rebuild it before searching.", THEME["warn"])

    # ------------------------------------------------------------------ #
    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select documents", "", "Documents (*.pdf *.xlsx *.xls)")
        for path in files:
            if path not in self.pending_files:
                self.pending_files.append(path)
                self.file_list.addItem(path.split("/")[-1])

    def _build_index(self) -> None:
        """Build index with debugging."""
        if self.busy:
            return
        if not self.pending_files:
            QMessageBox.information(self, "No files",
                "Add at least one PDF or Excel file first.")
            return
        
        print("\n" + "="*60)
        print("DEBUG: Starting index build")
        print(f"DEBUG: Files to process: {self.pending_files}")
        print(f"DEBUG: Number of files: {len(self.pending_files)}")
        print("="*60 + "\n")
        
        self._set_busy(True, "Indexing documents…")
        
        worker = Worker(self.rag.index_documents, list(self.pending_files))
        worker.signals.progress.connect(lambda m: self.status.showMessage(m))
        worker.signals.result.connect(self._on_index_built)
        worker.signals.error.connect(self._on_index_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    def _on_index_error(self, error_msg: str) -> None:
        """Handle indexing errors."""
        print(f"\n❌ INDEX ERROR: {error_msg}\n")
        self._show_error(f"Index error: {error_msg}")

    def _on_index_built(self, payload) -> None:
        stats, dataframes = payload
        self.agent.dataframes = dataframes
        self._refresh_index_label()
        self._update_badges()
        self._update_welcome()
        self._system_line(
            f"Indexed {stats['files']} file(s): {stats['pdf_chunks']} text "
            f"chunk(s) from PDFs, {stats['tables']} table(s) for querying.",
            THEME["muted"])

    def _load_index(self) -> None:
        if self.busy:
            return
        self._set_busy(True, "Loading saved index \u2026")
        worker = Worker(self.rag.load_persisted, cfg.INDEX_DIR)
        worker.signals.result.connect(self._after_load)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    def _after_load(self, dataframes) -> None:
        self.agent.dataframes = dataframes or {}
        self._refresh_index_label()
        self._update_badges()
        self._update_welcome()
        if self.store.embed_model:
            self.embed_combo.setCurrentText(self.store.embed_model)
        n_tables = len(self.agent.dataframes)
        self._system_line(
            f"Loaded saved index ({self.store.count} text chunk(s), "
            f"{n_tables} table(s) restored).", THEME["muted"])

    def _new_chat(self) -> None:
        """Start a fresh conversation: clear the transcript and the agent's
        memory, but keep the loaded index so the user can keep asking."""
        self.chat.clear()
        try:
            self.agent.clear_history()
        except Exception:
            pass
        self._pending_options = []
        self._prov_store.clear()
        self._chart_store.clear()
        if hasattr(self, "_last_question"):
            self._last_question = ""
        self._render_welcome()
        self.status.showMessage("New conversation.", 3000)

    def _clear_index(self) -> None:
        self.store.reset()
        self.agent.dataframes = {}
        self.pending_files.clear()
        self.file_list.clear()
        self._refresh_index_label()
        self._update_badges()
        self._update_welcome()
        self.status.showMessage("Index cleared.", 4000)

    def _refresh_index_label(self) -> None:
        chunks = self.store.count
        tables = len(self.agent.dataframes)
        if chunks == 0 and tables == 0:
            self.index_label.setText(_t("index_empty"))
        else:
            self.index_label.setText(
                f"Index: {chunks} text chunk(s), {tables} table(s)")

    # ------------------------------------------------------------------ #
    def _send(self) -> None:
        if self.busy:
            return
        text = self.input.text().strip()
        if not text:
            return
        self._show_chat()
        self.input.clear()
        self._last_question = text
        self._add_user_message(text)
        self._start_assistant_line()
        self._set_busy(True, "Working \u2026")
        worker = Worker(self.agent.handle, text)
        worker.signals.token.connect(self._append_token)
        worker.signals.progress.connect(lambda m: self.status.showMessage(m))
        worker.signals.result.connect(self._on_answer)
        worker.signals.error.connect(self._on_answer_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    @staticmethod
    def _md_to_html(text: str) -> str:
        """Minimal markdown -> Qt-rich-text: paragraphs, line breaks, bold,
        inline code, bullets. Keeps the answer readable like a chat reply."""
        import re as _re
        safe = html.escape(text)
        safe = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        safe = _re.sub(r"`([^`]+)`",
                       rf'<span style="background:{THEME["code_bg"]}; '
                       rf'font-family:monospace;">\1</span>', safe)
        lines = []
        for ln in safe.split("\n"):
            s = ln.strip()
            if s.startswith("- ") or s.startswith("* "):
                lines.append("&nbsp;&nbsp;\u2022 " + s[2:])
            else:
                lines.append(ln)
        return "<br>".join(lines)

    def _styled_table(self, table_html: str) -> str:
        t = table_html.replace(
            "<table", '<table border="0" cellspacing="0" cellpadding="6"', 1)
        t = t.replace("<th", f'<th bgcolor="{self.current_theme["panel2"]}"')
        return t

    @staticmethod
    def _parse_html_table(table_html: str):
        """Parse a pandas-generated <table> back into (headers, rows). Returns
        ([], []) if it can't. Presentation-only; no external deps."""
        import re as _re
        from html import unescape
        rows = []
        for tr in _re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, _re.S | _re.I):
            cells = _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, _re.S | _re.I)
            cells = [unescape(_re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
            if cells:
                rows.append(cells)
        if not rows:
            return [], []
        headers = rows[0]
        body = rows[1:]
        # pandas writes a blank top-left corner for the index column; drop the
        # leading index cell from each data row and the blank header
        if headers and headers[0] == "":
            headers = headers[1:]
            body = [r[1:] if len(r) > len(headers) else r for r in body]
        return headers, body

    # ------------------------------------------------------------------ #
    # Interactive charts                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chartable(headers: list, rows: list) -> bool:
        """A result is worth a chart when it has >= 2 rows and at least one
        numeric column (so there is a measure to plot against a category)."""
        if not headers or not rows or len(rows) < 2:
            return False
        try:
            from jarvisman.ui import charting
            df = charting.frame_from_table(headers, rows)
            _cats, nums = charting.infer_roles(df)
            return bool(nums) and df.shape[0] >= 2
        except Exception:
            return False

    def _maybe_offer_chart(self, question: str, result: dict) -> None:
        """If this answer has chartable tabular data, append a 'Visualize'
        link that opens the interactive chart window for it."""
        th = result.get("table_html")
        if not th:
            return
        headers, rows = self._parse_html_table(th)
        if not self._chartable(headers, rows):
            return
        self._chart_seq += 1
        key = f"c{self._chart_seq}"
        title = (question or "Result").strip()
        self._chart_store[key] = (title, headers, rows)
        self._append_html(
            f'<div style="margin:6px 0 2px 0;"><a href="chart:{key}" '
            f'style="text-decoration:none;">'
            f'<span style="background:{THEME["panel2"]}; '
            f'color:{THEME["accent"]}; border:1px solid {THEME["accent_dim"]}; '
            f'padding:5px 12px; border-radius:14px;">'
            f'{_t("visualize")}</span></a></div>')

    def _open_chart(self, key: str) -> None:
        data = self._chart_store.get(key)
        if not data:
            self.status.showMessage(
                "That chart belongs to an earlier answer.", 5000)
            return
        title, headers, rows = data
        try:
            from jarvisman.ui.chart_window import ChartWindow
            win = ChartWindow.from_table(title, headers, rows,
                                         theme=THEME, parent=self)
            self._chart_windows.append(win)   # keep a ref so Qt won't GC it
            win.show()
        except Exception as exc:  # never let a chart failure break the chat
            self.status.showMessage(f"Could not open chart: {exc}", 6000)

    def _render_options(self, options: list) -> None:
        self._opt_gen = getattr(self, "_opt_gen", 0) + 1
        self._pending_options = list(options)
        chips = "&nbsp;".join(
            f'<a href="opt:{self._opt_gen}:{i}" style="text-decoration:none;">'
            f'<span style="background:{THEME["accent_dim"]}; '
            f'color:#ffffff; padding:5px 14px; border-radius:14px;">'
            f'{html.escape(o)}</span></a>'
            for i, o in enumerate(options))
        self._append_html(
            f'<div style="margin:6px 0;">'
            f'<span style="color:{THEME["muted"]}">{_t("choose")}</span> '
            f'{chips}</div>')

    def _present_answer(self, question: str, result: dict) -> bool:
        """Render a polished answer: a heading restating the question, then
        bullets (CODE Name) for short entity lists, or a clean labeled table.
        Returns True if it handled rendering. Presentation only -- the numbers
        and rows are exactly what the engine produced."""
        th = result.get("table_html")
        if not th:
            return False
        headers, rows = self._parse_html_table(th)
        if not headers or not rows:
            return False
        q = (question or "").strip()
        heading = q[:1].upper() + q[1:] if q else "Result"
        if heading and heading[-1] not in ".?!":
            heading += ""
        accent = self.current_theme.get("accent", "#7aa2f7")
        out = [f'<div style="margin:4px 0 6px 0; font-weight:600; '
               f'color:{accent};">{html.escape(heading)}</div>']

        # detect a "code + name" entity list: exactly two text columns, one of
        # which looks like short codes (<=6 chars, mostly upper/alnum), and a
        # small number of rows -> bullets like "CY10  KR Falco Holdings Ltd"
        def _is_code(vals):
            ok = 0
            for v in vals:
                s = v.strip()
                if 1 <= len(s) <= 6 and any(ch.isdigit() for ch in s) and s.upper() == s:
                    ok += 1
            return ok >= max(1, len(vals) // 2)

        ncol = len(headers)
        col_vals = [[r[i] if i < len(r) else "" for r in rows] for i in range(ncol)]
        # dedupe rows (one row per entity) for listy answers
        uniq = []
        seen = set()
        for r in rows:
            key = tuple(r)
            if key not in seen:
                seen.add(key); uniq.append(r)

        if ncol == 2 and len(uniq) <= 40 and (_is_code(col_vals[0]) or _is_code(col_vals[1])):
            code_idx = 0 if _is_code(col_vals[0]) else 1
            name_idx = 1 - code_idx
            items = []
            for r in uniq:
                code = r[code_idx] if code_idx < len(r) else ""
                name = r[name_idx] if name_idx < len(r) else ""
                items.append(f'&nbsp;&nbsp;\u2022 <b>{html.escape(code)}</b>'
                             f'&nbsp;&nbsp;{html.escape(name)}')
            out.append("<br>".join(items))
            self._append_html("".join(out))
            return True

        # single value (one row, one col) -> a labeled total line
        if len(rows) == 1 and ncol == 1:
            out.append(f'<div style="font-size:15px;"><b>{html.escape(rows[0][0])}</b></div>')
            self._append_html("".join(out))
            return True

        # otherwise: a clean styled table under the heading
        out.append(self._styled_table(th))
        self._append_html("".join(out))
        return True

    def _on_answer(self, result: dict) -> None:
        # streamed replies are already in the transcript; render extras inline
        if result.get("streamed"):
            self._render_answer(result)
            return
        # otherwise collect the whole answer and wrap it in one card
        self._buf = []
        try:
            self._render_answer(result)
        finally:
            frags, self._buf = self._buf, None
        if not frags:
            return
        
        # Store answer text for copying
        MainWindow._prov_seq += 1
        copy_key = str(MainWindow._prov_seq)
        answer_html = "".join(frags)
        self._prov_store[f"copy:{copy_key}"] = answer_html
        
        T = self.current_theme
        card = (f'<div style="margin:12px 0 8px 0; padding:14px 16px; '
                f'background:{T["panel"]}; border-radius:8px;">'
                f'<div style="text-align: right; margin-bottom: 8px;">'
                f'<a href="copy:{copy_key}" style="text-decoration:none;">'
                f'<span style="color:{T["accent"]}; font-size:12px; background:{T["panel2"]}; '
                f'padding:4px 10px; border-radius:6px; border:1px solid {T["border"]}; cursor:pointer;">'
                f'📋 Copy</span></a>'
                f'</div>'
                f'{answer_html}'
                f'</div>')
        self.chat.append(card)
        cur = self._cursor_end()
        self.chat.setTextCursor(cur)
        self.chat.ensureCursorVisible()

        # Detect and show chart selector
        if result:
            try:
                response_text = str(result.get("answer", "") or result.get("text", "") or result)
                extraction = ResponseDataExtractor.detect_numerical_data(response_text)
                
                if extraction.has_data and extraction.df is not None:
                    self.data_summary.update_summary(extraction.df)
                    self.chart_selector.set_dataframe(extraction.df)
                    self.chart_selector.set_available_charts(extraction.suggested_charts)
                    self.chart_selector.setVisible(True)
                else:
                    self.chart_selector.setVisible(False)
            except Exception:
                self.chart_selector.setVisible(False)


    def _render_answer(self, result: dict) -> None:
        text = result.get("text", "") or ""
        has_table = bool(result.get("table_html"))
        question = getattr(self, "_last_question", "")
        if not result.get("streamed"):
            if has_table:
                # polished presentation: heading + bullets/labeled table,
                # then only the '(How: ...)' explanation line (never the same
                # data twice)
                presented = self._present_answer(question, result)
                import re as _re
                m = _re.search(r"\(How:[^)]*\)", text)
                if m:
                    self._append_html(
                        f'<span style="color:{THEME["muted"]}; '
                        f'font-size:12px;">{html.escape(m.group(0))}</span>')
                elif not presented and text and "\n" not in text.strip():
                    self._append_html(self._md_to_html(text))
                if presented:
                    if result.get("code"):
                        self._append_code(result["code"])
                    self._maybe_offer_chart(question, result)
                    options = result.get("options") or []
                    if options:
                        self._render_options(options)
                    return
            elif text:
                # aligned monospace blocks stay <pre>; prose gets markdown
                if "\n" in text and "  " in text:
                    self._append_html(
                        f'<pre style="font-size:13px; white-space:pre-wrap;">'
                        f'{html.escape(text)}</pre>')
                else:
                    self._append_html(self._md_to_html(text))
        if result.get("type") == "plot" and result.get("image"):
            self._append_image(result["image"])
        if has_table:
            self._append_html(self._styled_table(result["table_html"]))
        if result.get("code"):
            self._append_code(result["code"])
        options = result.get("options") or []
        if options:
            self._render_options(options)
        prov = result.get("provenance") or []
        if prov:
            MainWindow._prov_seq += 1
            key = str(MainWindow._prov_seq)
            self._prov_store[key] = "<br>".join(html.escape(p) for p in prov)
            self._append_html(
                f'<div style="margin-top:4px;"><a href="prov:{key}" '
                f'style="color:{THEME["muted"]}; text-decoration:none;">'
                f'{_t("how")}</a></div>')
        sources = result.get("sources") or []
        if sources:
            tags = "; ".join(
                f'{html.escape(s["source"])} {html.escape(s["location"])} '
                f'({s["score"]})' for s in sources)
            self._append_html(
                f'<div style="color:{THEME["muted"]}; font-size:12px; '
                f'margin-top:4px;">Sources: {tags}</div>')
        self._maybe_offer_chart(question, result)
        self._append_html('<div style="height:14px;"></div>')

    def _on_anchor(self, url) -> None:
        link = url.toString()
        
        # Handle copy button
        if link.startswith("copy:"):
            key = link  # e.g., "copy:1"
            html_content = self._prov_store.get(key, "")
            if html_content:
                # Clean HTML tags
                import re as _re
                from html import unescape
                text = _re.sub(r'<[^>]+>', '', html_content)
                text = unescape(text)
                
                # Copy to clipboard
                clipboard = QApplication.clipboard()
                clipboard.setText(text)
                self.status.showMessage("✓ Copied to clipboard!", 2000)
            return
        
        if link.startswith("opt:"):
            try:
                _, gen_s, idx_s = link.split(":")
                gen, idx = int(gen_s), int(idx_s)
            except ValueError:
                return
            if gen != getattr(self, "_opt_gen", 0) or not self._pending_options:
                self.status.showMessage(
                    "That choice belongs to an earlier question -- please "
                    "ask the question again.", 6000)
                return
            try:
                label = self._pending_options[idx]
            except IndexError:
                return
            self._pending_options = []
            self.input.setText(label)
            self._send()
        elif link.startswith("prov:"):
            block = self._prov_store.pop(link[5:], None)
            if block:
                self._append_html(
                    f'<div style="color:{self.current_theme["muted"]}; font-size:12px; '
                    f'margin:4px 0 8px 14px; padding:8px 12px; '
                    f'background:{self.current_theme["panel"]}; border-left:2px solid '
                    f'{self.current_theme["accent_dim"]}; border-radius:6px;">{block}</div>')
        elif link.startswith("chart:"):
            self._open_chart(link[6:])
        elif link.startswith("ask:"):
            from urllib.parse import unquote
            if self.busy:
                return
            self.input.setText(unquote(link[4:]))
            self._send()

    def _on_answer_error(self, message: str) -> None:
        first = message.splitlines()[0] if message else "Unknown error"
        self._append_html(
            f'<span style="color:{THEME["err"]}">{html.escape(first)}</span>'
            f'<div style="height:14px;"></div>')

    # ------------------------------------------------------------------ #
    def _render_welcome(self) -> None:
        self.chat.setVisible(False)
        self.welcome.setVisible(True)
        self._update_welcome()

    def _system_line(self, text: str, color: str) -> None:
        # status messages live in the status bar, not the transcript
        self.status.showMessage(text, 6000)

    def _cursor_end(self) -> QTextCursor:
        cur = self.chat.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        return cur

    def _append_html(self, fragment: str) -> None:
        if self._buf is not None:
            self._buf.append(fragment)
            return
        self.chat.append(fragment)
        cur = self._cursor_end()
        self.chat.setTextCursor(cur)
        self.chat.ensureCursorVisible()

    def _append_token(self, text: str) -> None:
        if not text:
            return
        cur = self._cursor_end()
        cur.insertText(text)
        self.chat.setTextCursor(cur)
        self.chat.ensureCursorVisible()

    def _add_user_message(self, text: str) -> None:
        safe = html.escape(text)
        T = self.current_theme
        self._append_html(
            f'<div style="margin:12px 0 8px 0; text-align: right;">'
            f'<span style="background:{T["panel"]}; color:{T["text"]}; '
            f'padding:12px 16px; border-radius:20px; display: inline-block; max-width:70%;">'
            f'{safe} 👤'
            f'</span>'
            f'</div>')
        

    def _start_assistant_line(self) -> None:
        T = self.current_theme
        self._append_html(
            f'<table width="100%"><tr><td style="width:100%; text-align: left;">'
            f'<b style="color:{T["accent"]};">🤖 {html.escape(cfg.ASSISTANT_NAME)}</b>'
            f'</td></tr></table>')
        
    def _append_image(self, png_bytes: bytes) -> None:
        image = QImage()
        if not image.loadFromData(png_bytes):
            self._append_html(
                f'<br><span style="color:{THEME["err"]}">'
                f'[could not render image]</span>')
            return
        self._img_counter += 1
        url = QUrl(f"plot://{self._img_counter}")
        self.chat.document().addResource(
            QTextDocument.ResourceType.ImageResource, url, image)
        width = min(cfg.PLOT_MAX_WIDTH, image.width())
        self._append_html(
            f'<br><img src="{url.toString()}" width="{width}"><br>')

    def _append_code(self, code: str) -> None:
        escaped = html.escape(code)
        self._append_html(
            f'<div style="color:{THEME["muted"]}; font-size:11px; '
            f'margin-top:8px;">Generated code</div>'
            f'<pre style="background:{THEME["code_bg"]}; color:#d6dee8; '
            f'padding:10px 12px; border-radius:8px; font-size:12px; '
            f'white-space:pre-wrap; border:1px solid {THEME["border"]};">'
            f'{escaped}</pre>')

    # ------------------------------------------------------------------ #
    def _set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self.busy = busy
        self.progress.setVisible(busy)
        if busy:
            self._start_typing()
        else:
            self._stop_typing()
        for w in (self.send_btn, self.input, self.add_btn, self.build_btn,
                  self.load_btn, self.clear_btn, self.refresh_btn):
            w.setEnabled(not busy)
        if message:
            self.status.showMessage(message)
        elif not busy:
            self.status.showMessage(_t("ready"), 3000)
        if not busy:
            self.input.setFocus()

    def _show_error(self, message: str) -> None:
        first = message.splitlines()[0] if message else "Unknown error"
        QMessageBox.warning(self, "Error", first)
        self.status.showMessage(first, 6000)

    def _on_chart_visualization_requested(self, chart_type: str, df) -> None:
        """Generate and display interactive chart when user selects from dropdown."""
        if df is None or df.empty:
            self._show_error("No data available for visualization")
            return
        
        try:
            from jarvisman.ui.interactive_charts import InteractiveCharts
            import plotly.io as pio
            
            # Get chart
            fig = InteractiveCharts.get_chart(chart_type, df)
            
            if fig is None:
                self._show_error(f"Cannot create {chart_type} chart with this data")
                return
            
            # Convert to HTML (interactive, no Chrome needed!)
            html = pio.to_html(fig, include_plotlyjs='cdn', div_id="chart")
            
            # Display in chat
            self.chat.insertHtml(html)
            cur = self._cursor_end()
            self.chat.setTextCursor(cur)
            self.chat.ensureCursorVisible()
            
            self._append_html('<div style="height:14px;"></div>')
            
        except Exception as e:
            import traceback
            error_msg = f"Chart error: {str(e)}"
            print(f"Full error:\n{traceback.format_exc()}")
            self._show_error(error_msg)

    def _get_stylesheet(self) -> str:
        """Get stylesheet using current theme."""
        T = self.current_theme
        return f"""
        QMainWindow, QWidget {{ background: {T['bg']}; color: {T['text']};
            font-size: 14px; }}
        QLabel {{ color: {T['text']}; }}
        QLabel#muted {{ color: {T['muted']}; font-size: 12px; }}
        QLabel#heading {{ color: {T['text']}; font-size: 12px; font-weight: 700;
            letter-spacing: 1px; }}
        QFrame#card {{ background: {T['panel']}; border: 1px solid {T['border']};
            border-radius: 12px; }}
        QFrame#sep {{ background: {T['border']}; max-height: 1px; border: none; }}
        QComboBox, QLineEdit {{ background: {T['panel2']}; color: {T['text']};
            border: 1px solid {T['border']}; border-radius: 8px;
            padding: 7px 10px; selection-background-color: {T['accent_dim']}; }}
        QComboBox:focus, QLineEdit:focus {{ border: 1px solid {T['accent']}; }}
        QComboBox QAbstractItemView {{ background: {T['panel2']};
            color: {T['text']}; selection-background-color: {T['accent_dim']};
            border: 1px solid {T['border']}; outline: none; }}
        QPushButton {{ background: {T['panel2']}; color: {T['text']};
            border: 1px solid {T['border']}; border-radius: 8px;
            padding: 8px 14px; }}
        QPushButton:hover {{ border: 1px solid {T['accent']}; }}
        QPushButton:disabled {{ color: {T['muted']}; }}
        QPushButton#primary {{ background: {T['accent']}; color: #06121f;
            border: none; font-weight: 700; }}
        QPushButton#primary:hover {{ background: #5aa6ff; }}
        QPushButton#primary:disabled {{ background: {T['panel2']};
            color: {T['muted']}; }}
        QToolButton {{ background: transparent; color: {T['muted']};
            border: none; font-size: 18px; padding: 2px 6px; }}
        QToolButton:hover {{ color: {T['accent']}; }}
        QListWidget {{ background: {T['panel2']}; color: {T['text']};
            border: 1px solid {T['border']}; border-radius: 8px; padding: 4px; }}
        QListWidget::item {{ padding: 5px 6px; border-radius: 6px; }}
        QListWidget::item:selected {{ background: {T['accent_dim']}; }}
        QTextBrowser {{ background: {T['bg']}; color: {T['text']};
            border: none; font-size: 14px; }}
        QProgressBar {{ background: {T['panel2']}; border: 1px solid {T['border']};
            border-radius: 6px; height: 6px; }}
        QProgressBar::chunk {{ background: {T['accent']}; border-radius: 6px; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {T['border']};
            border-radius: 5px; min-height: 30px; }}
        QScrollBar::handle:vertical:hover {{ background: {T['muted']}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
        QStatusBar {{ background: {T['panel']}; color: {T['muted']}; }}
        QSplitter::handle {{ background: {T['border']}; }}
        QPushButton#chip {{ background: {T['panel2']}; color: {T['text']};
            border: 1px solid {T['border']}; border-radius: 16px;
            padding: 9px 16px; text-align: center; }}
        QPushButton#chip:hover {{ border: 1px solid {T['accent']};
            color: {T['accent']}; }}
        QToolButton#advanced {{ color: {T['muted']}; font-size: 12px;
            padding: 2px 0; }}
        QToolButton#advanced:hover {{ color: {T['accent']}; }}
        QLabel#badge {{ color: {T['muted']}; font-size: 11px;
            background: {T['panel']}; border: 1px solid {T['border']};
            border-radius: 10px; padding: 3px 10px; }}
        QLabel#warn {{ color: {T['warn']}; font-size: 11px; }}
        QLabel#hero {{ color: {T['text']}; font-size: 25px; font-weight: 800; }}
        QLabel#herosub {{ color: {T['muted']}; font-size: 16px; }}
        QLabel#heromark {{ color: {T['accent']}; font-size: 30px; }}
        QLabel#suggest {{ color: {T['muted']}; font-size: 12px; }}
        QLabel#typing {{ color: {T['accent']}; font-size: 12px; }}
        """


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
