"""Application-wide palettes for widgets, dialogs and startup; no audio imports."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor,QPalette

DEFAULT_THEME="dark"
THEME_KEY="appearance/theme"
THEMES={
    "dark":dict(background="#18212d",surface="#202b3b",base="#141d29",text="#dce5f0",
        title="#eef4ff",muted="#a1afc2",border="#364459",hover="#293c54",accent="#326cb8",
        selection="#294563",selected="#f1f6ff",disabled="#8190a5",soft="#253142",link="#8ebcff"),
    "light":dict(background="#f3f5f8",surface="#ffffff",base="#ffffff",text="#233247",
        title="#182d48",muted="#65768c",border="#d8e0ea",hover="#edf3ff",accent="#356ad4",
        selection="#e3edff",selected="#194986",disabled="#8794a5",soft="#edf1f7",link="#2459bb"),
}


def normalize_theme(value):
    return value if value in THEMES else DEFAULT_THEME


def stylesheet(colors):
    c=colors
    return f"""
QWidget {{ font-family:'Microsoft YaHei'; font-size:13px; }}
QMainWindow, QDialog, QWidget#startupWindow {{ background:{c['background']}; color:{c['text']}; }}
QLabel, QCheckBox, QRadioButton {{ color:{c['text']}; }}
QLabel[role="title"] {{ font-size:22px; font-weight:600; color:{c['title']}; }}
QLabel[role="muted"] {{ color:{c['muted']}; }}
QCheckBox::indicator:unchecked {{ width:12px; height:12px; border:1px solid {c['muted']}; border-radius:2px; background:{c['base']}; }}
QPushButton, QToolButton {{ padding:7px 12px; border:1px solid {c['border']}; border-radius:6px; background:{c['surface']}; color:{c['text']}; }}
QPushButton:hover, QToolButton:hover {{ background:{c['hover']}; border-color:{c['link']}; }}
QPushButton[primary="true"] {{ background:{c['accent']}; color:#ffffff; border-color:{c['accent']}; font-weight:600; }}
QPushButton[primary="true"]:hover {{ border-color:{c['link']}; }}
QPushButton:checked {{ background:{c['selection']}; color:{c['selected']}; border-color:{c['link']}; }}
QPushButton:disabled, QToolButton:disabled {{ background:{c['soft']}; color:{c['disabled']}; border-color:{c['border']}; }}
QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit, QPlainTextEdit {{ padding:5px; border:1px solid {c['border']}; border-radius:4px; background:{c['base']}; color:{c['text']}; selection-background-color:{c['selection']}; selection-color:{c['selected']}; }}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color:{c['link']}; }}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled, QCheckBox:disabled {{ color:{c['disabled']}; }}
QAbstractItemView {{ background:{c['base']}; alternate-background-color:{c['surface']}; border:1px solid {c['border']}; border-radius:5px; color:{c['text']}; selection-background-color:{c['selection']}; selection-color:{c['selected']}; gridline-color:{c['border']}; }}
QListWidget::item {{ padding:2px 4px; }}
QListWidget#recentProjects::item {{ padding:2px 6px; border:1px solid transparent; border-radius:3px; }}
QListWidget#recentProjects::item:hover {{ background:{c['hover']}; border-color:{c['border']}; }}
QListWidget#recentProjects::item:selected, QListWidget#recentProjects::item:selected:!active {{ background:{c['selection']}; color:{c['selected']}; border-color:{c['link']}; }}
QAbstractItemView::item:selected {{ background:{c['selection']}; color:{c['selected']}; }}
QHeaderView::section, QTableCornerButton::section {{ background:{c['soft']}; color:{c['muted']}; border:0; padding:7px; }}
QMenuBar {{ background:{c['surface']}; color:{c['text']}; padding:3px; }}
QMenuBar::item:selected, QMenu::item:selected {{ background:{c['selection']}; color:{c['selected']}; }}
QMenu {{ background:{c['surface']}; color:{c['text']}; border:1px solid {c['border']}; }}
QMenu::item {{ padding:7px 22px; }}
QMenu::item:disabled {{ color:{c['disabled']}; }}
QMenu::separator {{ height:1px; background:{c['border']}; margin:4px 8px; }}
QToolTip {{ background:{c['surface']}; color:{c['text']}; border:1px solid {c['border']}; padding:5px; }}
QGroupBox {{ border:1px solid {c['border']}; border-radius:5px; margin-top:12px; padding-top:8px; color:{c['text']}; }}
QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left; padding:0 5px; }}
QProgressBar {{ border:0; background:{c['soft']}; border-radius:3px; color:{c['text']}; text-align:center; }}
QProgressBar::chunk {{ background:{c['accent']}; border-radius:3px; }}
QSplitter::handle {{ background:{c['border']}; }}
"""


def apply_theme(application,name):
    name=normalize_theme(name)
    c=THEMES[name]
    application.styleHints().setColorScheme(Qt.ColorScheme.Dark if name=="dark" else Qt.ColorScheme.Light)
    palette=QPalette()
    roles={"Window":"background","WindowText":"text","Base":"base","AlternateBase":"surface",
        "Text":"text","Button":"surface","ButtonText":"text","ToolTipBase":"surface",
        "ToolTipText":"text","Highlight":"selection","HighlightedText":"selected",
        "PlaceholderText":"muted","Link":"link","LinkVisited":"link",
        "Light":"border","Midlight":"soft","Mid":"border","Dark":"base","Shadow":"base"}
    for role,color in roles.items():
        palette.setColor(getattr(QPalette.ColorRole,role),QColor(c[color]))
    palette.setColor(QPalette.ColorRole.BrightText,QColor("#ffffff"))
    for role in (QPalette.ColorRole.Text,QPalette.ColorRole.WindowText,QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled,role,QColor(c["disabled"]))
    application.setPalette(palette)
    application.setStyleSheet(stylesheet(c))
    return name
