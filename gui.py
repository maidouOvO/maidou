from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QSpinBox, QLineEdit, QMessageBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys
import os
from pdf_to_images import convert_pdf_to_images

class PDFProcessThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, int)

    def __init__(self, pdf_path, export_path, width, height):
        super().__init__()
        self.pdf_path = pdf_path
        self.export_path = export_path
        self.width = width
        self.height = height

    def run(self):
        try:
            convert_pdf_to_images(self.pdf_path, self.export_path, self.width, self.height)
            self.finished.emit(True, "PDF processed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

class PDFProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Text Extractor")
        self.setMinimumSize(600, 400)
        self.process_thread = None
        self.aspect_ratio_lock = True
        self.updating_size = False

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # PDF Upload Section
        pdf_group = QGroupBox("PDF File Selection")
        pdf_layout = QHBoxLayout(pdf_group)
        self.pdf_path_edit = QLineEdit()
        self.pdf_path_edit.setPlaceholderText("Select PDF file...")
        self.pdf_path_edit.setReadOnly(True)
        pdf_button = QPushButton("Browse PDF")
        pdf_button.clicked.connect(self.select_pdf)
        pdf_layout.addWidget(self.pdf_path_edit)
        pdf_layout.addWidget(pdf_button)
        layout.addWidget(pdf_group)

        # Canvas Size Section
        size_group = QGroupBox("Canvas Size Settings")
        size_layout = QVBoxLayout(size_group)

        # Size controls
        size_controls = QHBoxLayout()
        size_controls.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 4000)
        self.width_spin.setValue(800)
        self.width_spin.valueChanged.connect(self.on_width_changed)
        size_controls.addWidget(self.width_spin)

        size_controls.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 4000)
        self.height_spin.setValue(1280)
        self.height_spin.valueChanged.connect(self.on_height_changed)
        size_controls.addWidget(self.height_spin)

        # Aspect ratio lock button
        self.lock_button = QPushButton("🔒")
        self.lock_button.setCheckable(True)
        self.lock_button.setChecked(True)
        self.lock_button.clicked.connect(self.toggle_aspect_ratio_lock)
        self.lock_button.setToolTip("Lock aspect ratio")
        size_controls.addWidget(self.lock_button)

        size_layout.addLayout(size_controls)

        # Add preset buttons
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        presets = [
            ("Default (800x1280)", (800, 1280)),
            ("HD (1280x720)", (1280, 720)),
            ("4K (3840x2160)", (3840, 2160))
        ]
        for name, (w, h) in presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, w=w, h=h: self.apply_preset(w, h))
            preset_layout.addWidget(btn)
        size_layout.addLayout(preset_layout)

        layout.addWidget(size_group)

        # Export Path Section
        export_group = QGroupBox("Export Settings")
        export_layout = QHBoxLayout(export_group)
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText("Select export directory...")
        self.export_path_edit.setReadOnly(True)
        export_button = QPushButton("Browse Export")
        export_button.clicked.connect(self.select_export_path)
        export_layout.addWidget(self.export_path_edit)
        export_layout.addWidget(export_button)
        layout.addWidget(export_group)

        # Process Button
        self.process_button = QPushButton("Process PDF")
        self.process_button.clicked.connect(self.process_pdf)
        self.process_button.setStyleSheet("QPushButton { padding: 10px; }")
        layout.addWidget(self.process_button)

        # Status Label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def on_width_changed(self, new_width):
        if not self.updating_size and self.aspect_ratio_lock:
            self.updating_size = True
            ratio = 1280 / 800  # Default aspect ratio
            self.height_spin.setValue(int(new_width * ratio))
            self.updating_size = False

    def on_height_changed(self, new_height):
        if not self.updating_size and self.aspect_ratio_lock:
            self.updating_size = True
            ratio = 800 / 1280  # Default aspect ratio
            self.width_spin.setValue(int(new_height * ratio))
            self.updating_size = False

    def toggle_aspect_ratio_lock(self):
        self.aspect_ratio_lock = self.lock_button.isChecked()
        self.lock_button.setText("🔒" if self.aspect_ratio_lock else "🔓")

    def apply_preset(self, width, height):
        self.updating_size = True
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.updating_size = False

    def select_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        if file_name:
            if os.path.getsize(file_name) > 0:
                self.pdf_path_edit.setText(file_name)
                self.status_label.setText("")
            else:
                QMessageBox.warning(self, "Invalid File", "Selected PDF file is empty.")

    def select_export_path(self):
        current_dir = self.export_path_edit.text() or os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory",
            current_dir
        )
        if directory:
            try:
                # Create a test file to verify write permissions
                test_file = os.path.join(directory, '.write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)

                # Check if directory has enough space (minimum 100MB)
                if hasattr(os, 'statvfs'):
                    stats = os.statvfs(directory)
                    available_space = stats.f_frsize * stats.f_bavail
                    if available_space < 100 * 1024 * 1024:  # 100MB
                        QMessageBox.warning(self, "Space Warning",
                                         "Selected directory has less than 100MB free space. "
                                         "This might not be enough for large PDFs.")

                self.export_path_edit.setText(directory)
                self.status_label.setText("")

                # Create output directory if it doesn't exist
                if not os.path.exists(directory):
                    os.makedirs(directory)

            except (OSError, IOError) as e:
                QMessageBox.critical(self, "Permission Error",
                                  f"Cannot write to selected directory: {str(e)}\n"
                                  "Please choose another location.")

    def process_pdf(self):
        pdf_path = self.pdf_path_edit.text()
        export_path = self.export_path_edit.text()

        if not pdf_path or not export_path:
            QMessageBox.warning(self, "Input Error",
                              "Please select both PDF file and export directory")
            return

        # Validate PDF file still exists
        if not os.path.exists(pdf_path):
            QMessageBox.critical(self, "File Error",
                               "Selected PDF file no longer exists!")
            return

        # Validate export directory still exists and is writable
        if not os.path.exists(export_path):
            try:
                os.makedirs(export_path)
            except OSError as e:
                QMessageBox.critical(self, "Directory Error",
                                   f"Cannot create export directory: {str(e)}")
                return

        # Disable UI elements during processing
        self.process_button.setEnabled(False)
        self.width_spin.setEnabled(False)
        self.height_spin.setEnabled(False)
        self.lock_button.setEnabled(False)
        self.status_label.setText("Processing PDF...")

        # Start processing in background thread
        self.process_thread = PDFProcessThread(
            pdf_path, export_path,
            self.width_spin.value(),
            self.height_spin.value()
        )
        self.process_thread.finished.connect(self.on_process_complete)
        self.process_thread.start()

    def on_process_complete(self, success, message):
        self.process_button.setEnabled(True)
        self.status_label.setText(message)

        if success:
            QMessageBox.information(self, "Success",
                                  f"PDF processed successfully!\nOutput saved to: {self.export_path_edit.text()}")
        else:
            QMessageBox.critical(self, "Error", message)

def main():
    import sys
    from PyQt6.QtCore import QTimer
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform compatibility

    # Set up offscreen platform for testing
    if 'pytest' in sys.modules or not sys.stdout.isatty():
        app.setProperty('platform', 'offscreen')

    window = PDFProcessorGUI()
    window.show()

    # For testing: automatically close after 5 seconds if running headless
    if 'pytest' in sys.modules or not sys.stdout.isatty():
        def auto_close():
            print("GUI initialized successfully")
            app.quit()
        QTimer.singleShot(5000, auto_close)

    return app.exec()

if __name__ == "__main__":
    main()
