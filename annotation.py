import os
import csv

def create_annotation(dataset_path : str) -> None:
    """
    Creates a CSV annotation for a dataset.

    Parameters:
    - dataset_path (str): Path to root directory.

    Returns:
    None
    """
    output_file = 'annotation1.csv'
    # Открываем файл для записи аннотации
    with open(output_file, 'w', newline='') as csvfile:
        # Создаем объект для записи CSV
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Absolute path', 'Relative path', 'Class'])
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                relative_path = os.path.join(root, file)
                absolute_path = os.path.abspath(relative_path)
                class_img = os.path.basename(root)
                csv_writer.writerow([absolute_path, relative_path, class_img])

if __name__ == "__main__":
    directory = 'dataset/leopard'
    create_annotation(directory)