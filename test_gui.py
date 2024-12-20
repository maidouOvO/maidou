import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication, QPushButton
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QTimer
from gui import PDFProcessorGUI

class TestPDFProcessorGUI:
    def setup_method(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = PDFProcessorGUI()

    def teardown_method(self):
        self.window.close()
        del self.window
        self.app.quit()

    def test_initial_values(self):
        assert self.window.width_spin.value() == 800
        assert self.window.height_spin.value() == 1280
        assert self.window.aspect_ratio_lock is True
        assert self.window.pdf_path_edit.text() == ""
        assert self.window.export_path_edit.text() == ""

    def test_aspect_ratio_lock(self):
        self.window.lock_button.setChecked(True)
        self.window.toggle_aspect_ratio_lock()

        initial_height = self.window.height_spin.value()
        self.window.width_spin.setValue(1000)
        expected_height = int(1000 * (1280/800))
        assert self.window.height_spin.value() == expected_height

        self.window.lock_button.setChecked(False)
        self.window.toggle_aspect_ratio_lock()

        self.window.width_spin.setValue(1200)
        assert self.window.height_spin.value() == expected_height

    def test_preset_buttons(self):
        presets = [
            ("Default (800x1280)", (800, 1280)),
            ("HD (1280x720)", (1280, 720)),
            ("4K (3840x2160)", (3840, 2160))
        ]

        for name, (width, height) in presets:
            button = None
            for child in self.window.findChildren(QPushButton):
                if child.text() == name:
                    button = child
                    break

            assert button is not None
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

            assert self.window.width_spin.value() == width
            assert self.window.height_spin.value() == height

    def test_process_button_state(self):
        assert self.window.process_button.isEnabled()
        self.window.process_button.setEnabled(False)
        assert not self.window.process_button.isEnabled()
        self.window.process_button.setEnabled(True)
        assert self.window.process_button.isEnabled()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
