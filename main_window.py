from PyQt5 import QtWidgets
from annotation import create_annotation
from copy1 import copy_dataset  
import os

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.folder_path_label = QtWidgets.QLabel('Выберите папку с исходным датасетом:')
        self.folder_path_button = QtWidgets.QPushButton('Выбрать папку', self)
        self.create_annotation_button = QtWidgets.QPushButton('Создать аннотацию', self)
        self.copy_dataset_button = QtWidgets.QPushButton('Создать датасет c файлами типа class_0000.jpg', self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.folder_path_label)
        layout.addWidget(self.folder_path_button)
        layout.addWidget(self.create_annotation_button)
        layout.addWidget(self.copy_dataset_button)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.folder_path_button.clicked.connect(self.get_folder_path)
        self.create_annotation_button.clicked.connect(self.create_annotation)
        self.copy_dataset_button.clicked.connect(self.copy_dataset)

        self.selected_folder_path = None

    def get_folder_path(self):
        self.selected_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку с исходным датасетом')

    def create_annotation(self):
        if not self.selected_folder_path:
            return QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Выберите папку с исходным датасетом.')
        relative_path = os.path.relpath(self.selected_folder_path)
        create_annotation(relative_path)
        print(f"Аннотация создана для папки: {relative_path}")

    def copy_dataset(self):
        if not self.selected_folder_path:
            return QtWidgets.QMessageBox.warning(self, 'Ошибка', 'Выберите папку с исходным датасетом.')
        new_directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку для нового датасета')
        relative_path1 = os.path.relpath(self.selected_folder_path)
        relative_path2 = os.path.relpath(new_directory)
        copy_dataset(relative_path1, relative_path2)
        print(f"Датасет скопирован в новую папку: {relative_path2}")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())