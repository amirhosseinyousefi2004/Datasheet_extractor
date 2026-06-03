from PyQt6.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView
)

from extractor import DatasheetExtractor, FIELDS


class WorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


class Worker(QRunnable):
    def __init__(self, path, extractor):
        super().__init__()
        self.path = path
        self.extractor = extractor
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.extractor.process_pdf(self.path))
        except Exception as exc:
            self.signals.error.emit(str(exc))


class DropZone(QLabel):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__("Drag & Drop Datasheet PDFs")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(120)
        self.setStyleSheet("border:2px dashed gray;font-size:16px;padding:20px;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.files_dropped.emit([f for f in files if f.lower().endswith(".pdf")])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Datasheet Extractor V4")
        self.resize(1500, 700)

        self.extractor = DatasheetExtractor()
        self.pool = QThreadPool()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        dz = DropZone()
        dz.files_dropped.connect(self.process_files)
        layout.addWidget(dz)

        btn = QPushButton("Open PDFs")
        btn.clicked.connect(self.process_files)
        layout.addWidget(btn)

        self.table = QTableWidget(0, len(FIELDS))
        self.table.setHorizontalHeaderLabels(FIELDS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.SelectedClicked)
        layout.addWidget(self.table)

        b1 = QPushButton("Export JSON")
        b1.clicked.connect(self.export_json)
        layout.addWidget(b1)

        b2 = QPushButton("Export Excel")
        b2.clicked.connect(self.export_excel)
        layout.addWidget(b2)

    def process_files(self, files):
        if not files:
            files, _ = QFileDialog.getOpenFileNames(self, "PDFs", "", "PDF (*.pdf)")
        for f in files:
            worker = Worker(f, self.extractor)
            worker.signals.finished.connect(self.add_result)
            worker.signals.error.connect(self.show_error)
            self.pool.start(worker)

    def add_result(self, result):
        row = self.table.rowCount()
        self.table.insertRow(row)

        for col, field in enumerate(FIELDS):
            data = result.get(field, {})
            value = str(data.get("value", ""))
            conf = data.get("confidence", 0.0)

            item = QTableWidgetItem(value)

            if conf >= 0.8:
                item.setBackground(QColor(220,255,220))
            # elif conf >= 0.5:
            #     item.setBackground(QColor(255,245,180))
            else:
                item = QTableWidgetItem("Not Found!")
                item.setBackground(QColor(255,220,220))

            item.setToolTip(data.get("source", ""))
            self.table.setItem(row, col, item)

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)")
        if path:
            self.extractor.export_json(path)

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel (*.xlsx)")
        if path:
            self.extractor.export_excel(path)

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
