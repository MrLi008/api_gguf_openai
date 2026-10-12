
"""
PyQt6 GUI wrapper for the OpenAI-compatible GGUF LLM server.

关键设计：
  - 服务在独立 QThread 中运行 uvicorn，可随时通过 should_exit 停止
  - GGUF 路径 / 端口 / 上下文长度在运行时注入到 llm_server.config
  - 每次启动前重置 model_manager._model，确保加载新模型
  - 日志通过 QtLogHandler 桥接到界面

依赖：PyQt6, uvicorn, fastapi, llama-cpp-python, ...
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

# --- 导入原始服务模块 ---
import api_gguf_v5 as srv
# import api_gguf_v4 as srv

import uvicorn


# ============================================================
# 日志桥接：Python logging → Qt 信号 → 界面
# ============================================================
class LogEmitter(QObject):
    log_signal = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """把 logging 记录转发到 Qt 信号（线程安全）。"""

    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.emitter.log_signal.emit(msg)
        except Exception:
            self.handleError(record)


# ============================================================
# 服务线程
# ============================================================
class ServerThread(QThread):
    log_signal = pyqtSignal(str)
    started_signal = pyqtSignal()
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, app, host: str, port: int):
        super().__init__()
        self._app = app
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._stop_requested = False

    def run(self) -> None:
        # 1) 加载 / 重载 GGUF 模型（可能耗时较久）
        try:
            self.log_signal.emit("[Init] Loading GGUF model ...")
            srv.model_manager._model = None   # 强制重载
            gc.collect()
            srv.model_manager.warmup()
            self.log_signal.emit("[Init] Model loaded.")
        except Exception as e:
            self.log_signal.emit(f"[Init] Model load failed: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.error_signal.emit(str(e))
            self.stopped_signal.emit()
            return

        if self._stop_requested:
            self.stopped_signal.emit()
            return

        # 2) 启动 uvicorn
        try:
            cfg = uvicorn.Config(
                app=self._app,
                host=self._host,
                port=self._port,
                log_level="info",
                log_config=None,   # 使用已有的 root logger（会转发到 GUI）
            )
            self._server = uvicorn.Server(cfg)
            self.log_signal.emit(
                f"[Server] Listening on http://{self._host}:{self._port}"
            )
            self.started_signal.emit()
            self._server.run()          # 阻塞直到 should_exit=True
        except Exception as e:
            self.log_signal.emit(f"[Server] Error: {e}")
            self.log_signal.emit(traceback.format_exc())
            self.error_signal.emit(str(e))
        finally:
            self._server = None
            self.stopped_signal.emit()

    def stop(self) -> None:
        """从 GUI 线程安全地请求停止。"""
        self._stop_requested = True
        if self._server is not None:
            self._server.should_exit = True


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGUF LLM Server (OpenAI-compatible)")
        self.resize(960, 720)

        self.server_thread: ServerThread | None = None

        self._build_ui()
        self._setup_logging()
        self._update_ui_state(running=False)

    # ------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---------- 配置区 ----------
        cfg_box = QGroupBox("服务配置")
        form = QFormLayout()

        # GGUF 文件选择
        file_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择 .gguf 模型文件 ...")
        self.model_path_edit.setText(getattr(srv.config, "MODEL_PATH", "") or "")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(self.model_path_edit, 1)
        file_row.addWidget(self.browse_btn)
        file_widget = QWidget()
        file_widget.setLayout(file_row)
        form.addRow("模型文件:", file_widget)

        # 主机 / 端口
        net_row = QHBoxLayout()
        self.host_edit = QLineEdit(getattr(srv.config, "HOST", "127.0.0.1"))
        self.host_edit.setMaximumWidth(180)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(getattr(srv.config, "PORT", 8102)))
        net_row.addWidget(QLabel("主机:"))
        net_row.addWidget(self.host_edit)
        net_row.addSpacing(20)
        net_row.addWidget(QLabel("端口:"))
        net_row.addWidget(self.port_spin)
        net_row.addStretch(1)
        net_widget = QWidget()
        net_widget.setLayout(net_row)
        form.addRow("网络:", net_widget)

        # 上下文长度
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 262144)
        self.ctx_spin.setSingleStep(512)
        self.ctx_spin.setValue(int(getattr(srv.config, "MAX_CONTEXT", 32048)))
        form.addRow("上下文长度 (n_ctx):", self.ctx_spin)

        cfg_box.setLayout(form)
        root.addWidget(cfg_box)

        # ---------- 控制按钮 ----------
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("启动服务")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("停止服务")
        self.stop_btn.clicked.connect(self._on_stop)

        self.status_label = QLabel("状态: 已停止")
        f = QFont()
        f.setBold(True)
        self.status_label.setFont(f)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.status_label)
        root.addLayout(btn_row)

        # ---------- 日志区 ----------
        log_box = QGroupBox("日志")
        log_layout = QVBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_view)
        log_box.setLayout(log_layout)
        root.addWidget(log_box, 1)

    # ------------------------------------------------
    def _setup_logging(self) -> None:
        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self._append_log)

        handler = QtLogHandler(self.log_emitter)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(handler)

    # ------------------------------------------------
    def _append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    # ------------------------------------------------
    def _update_ui_state(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.model_path_edit.setEnabled(not running)
        self.browse_btn.setEnabled(not running)
        self.host_edit.setEnabled(not running)
        self.port_spin.setEnabled(not running)
        self.ctx_spin.setEnabled(not running)

    # ------------------------------------------------
    def _on_browse(self) -> None:
        start_dir = os.path.dirname(self.model_path_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 GGUF 模型文件", start_dir,
            "GGUF files (*.gguf);;All files (*.*)",
        )
        if path:
            self.model_path_edit.setText(path)

    # ------------------------------------------------
    def _on_start(self) -> None:
        if self.server_thread is not None and self.server_thread.isRunning():
            QMessageBox.information(self, "提示", "服务已在运行中")
            return

        model_path = self.model_path_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的 GGUF 模型文件")
            return

        host = self.host_edit.text().strip() or "127.0.0.1"
        port = self.port_spin.value()
        ctx = self.ctx_spin.value()

        # 运行时注入配置
        srv.config.MODEL_PATH = model_path
        srv.config.MODEL_NAME = os.path.splitext(os.path.basename(model_path))[0]
        srv.config.HOST = host
        srv.config.PORT = port
        srv.config.MAX_CONTEXT = ctx

        # 构建 FastAPI app + 中间件
        try:
            app = srv.create_openai_app()
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建服务失败: {e}")
            self._append_log(traceback.format_exc())
            return

        # 启动线程
        self.server_thread = ServerThread(app, host, port)
        self.server_thread.log_signal.connect(self._append_log)
        self.server_thread.started_signal.connect(self._on_server_started)
        self.server_thread.stopped_signal.connect(self._on_server_stopped)
        self.server_thread.error_signal.connect(self._on_server_error)
        self.server_thread.start()

        self._append_log(
            f"[GUI] Starting server with model={model_path} on {host}:{port} ..."
        )
        self.status_label.setText("状态: 正在启动...")
        self._update_ui_state(running=True)

    # ------------------------------------------------
    def _on_stop(self) -> None:
        if self.server_thread is None or not self.server_thread.isRunning():
            return
        self._append_log("[GUI] Stop requested ...")
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态: 正在停止...")
        self.server_thread.stop()

    # ------------------------------------------------
    def _on_server_started(self) -> None:
        self.status_label.setText("状态: 运行中")

    def _on_server_stopped(self) -> None:
        self.status_label.setText("状态: 已停止")
        self._update_ui_state(running=False)
        self._append_log("[GUI] Server stopped.")

    def _on_server_error(self, msg: str) -> None:
        QMessageBox.critical(self, "服务错误", msg)

    # ------------------------------------------------
    def closeEvent(self, event) -> None:
        if self.server_thread is not None and self.server_thread.isRunning():
            reply = QMessageBox.question(
                self, "退出", "服务正在运行，确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.server_thread.stop()
            if not self.server_thread.wait(10000):
                self._append_log("[GUI] Server did not stop within 10s, forcing exit.")
        event.accept()


# ============================================================
# 入口
# ============================================================
def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("GGUF LLM Server")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()