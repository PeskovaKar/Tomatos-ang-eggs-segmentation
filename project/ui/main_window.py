import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QGridLayout,
    QMessageBox,
)

from core.analyzer import ImageAnalyzer
from core.models import AnalysisResult


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сегментация объектов на изображениях")
        self.setMinimumSize(1100, 700)

        self.analyzer = ImageAnalyzer()
        self.current_image_path = None
        self.current_result = None

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        control_layout = QVBoxLayout()

        self.load_button = QPushButton("Загрузить изображение")
        self.process_button = QPushButton("Обработать")
        self.reset_button = QPushButton("Сброс")

        self.view_selector = QComboBox()
        self.view_selector.addItems([
            "Исходное изображение",
            "Результат обработки",
        ])

        self.status_label = QLabel("Статус: ожидание загрузки")
        self.status_label.setWordWrap(True)

        result_grid = QGridLayout()
        result_grid.addWidget(QLabel("Всего объектов:"), 0, 0)
        result_grid.addWidget(QLabel("Красных помидоров:"), 1, 0)
        result_grid.addWidget(QLabel("Жёлтых помидоров:"), 2, 0)
        result_grid.addWidget(QLabel("Яиц:"), 3, 0)

        self.total_value = QLabel("0")
        self.red_value = QLabel("0")
        self.yellow_value = QLabel("0")
        self.egg_value = QLabel("0")

        result_grid.addWidget(self.total_value, 0, 1)
        result_grid.addWidget(self.red_value, 1, 1)
        result_grid.addWidget(self.yellow_value, 2, 1)
        result_grid.addWidget(self.egg_value, 3, 1)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.process_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(QLabel("Режим просмотра:"))
        control_layout.addWidget(self.view_selector)
        control_layout.addSpacing(20)
        control_layout.addLayout(result_grid)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        self.image_label = QLabel("Здесь будет изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed gray;
                background-color: #f5f5f5;
                font-size: 18px;
            }
        """)

        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.image_label, 3)

        # Сигналы
        self.load_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_image)
        self.reset_button.clicked.connect(self.reset_ui)
        self.view_selector.currentIndexChanged.connect(self.update_display)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        self.current_image_path = file_path
        image = cv2.imread(file_path)

        if image is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть изображение")
            return

        self.current_result = AnalysisResult(
            original_image=image,
            processed_image=image,
            status_message="Изображение загружено"
        )

        self.total_value.setText("0")
        self.red_value.setText("0")
        self.yellow_value.setText("0")
        self.egg_value.setText("0")

        self.status_label.setText("Статус: изображение загружено")
        self.update_display()

    def process_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return

        self.current_result = self.analyzer.analyze(self.current_image_path)

        self.total_value.setText(str(self.current_result.total_count))
        self.red_value.setText(str(self.current_result.red_count))
        self.yellow_value.setText(str(self.current_result.yellow_count))
        self.egg_value.setText(str(self.current_result.egg_count))
        self.status_label.setText(f"Статус: {self.current_result.status_message}")

        self.update_display()

    def reset_ui(self):
        self.current_image_path = None
        self.current_result = None

        self.total_value.setText("0")
        self.red_value.setText("0")
        self.yellow_value.setText("0")
        self.egg_value.setText("0")
        self.status_label.setText("Статус: ожидание загрузки")
        self.image_label.setText("Здесь будет изображение")
        self.image_label.setPixmap(QPixmap())

    def update_display(self):
        if self.current_result is None:
            return

        selected_view = self.view_selector.currentText()
        image = None

        if selected_view == "Исходное изображение":
            image = self.current_result.original_image
        elif selected_view == "Результат обработки":
            image = self.current_result.processed_image

        if image is None:
            self.image_label.setText("Для этого режима изображение пока отсутствует")
            self.image_label.setPixmap(QPixmap())
            return

        self.show_image(image)

    def show_image(self, image):
        if len(image.shape) == 2:
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(
                image.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_Grayscale8
            )
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(
                rgb_image.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()