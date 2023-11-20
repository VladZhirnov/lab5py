import os
import shutil
import csv

def copy_dataset(original : str, new_directory : str) -> None:
    """
    Copies the dataset folder to another directory and creates an annotation for it.

    Parameters:
    - original (str): Path to root directory.
    - new_directory (str): Path to new directory.
    
    Returns:
    None
    """
    output_file = 'copy.csv'
    with open(output_file, 'w', newline='') as csvfile:
        # Создаем объект для записи CSV
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Absolute path', 'Relative path', 'Class'])
        for root, dirs, files in os.walk(original):
            for file in files:
                class_img = os.path.basename(root)
                new_filename = f"{class_img}_{file}"
                source_path = os.path.join(root, file)
                relative_path = os.path.join(new_directory, new_filename)
                shutil.copy(source_path, relative_path)
                absolute_path = os.path.abspath(relative_path)
                csv_writer.writerow([absolute_path, relative_path, class_img])

if __name__ == "__main__":
    original = 'dataset'
    new_directory = 'new_directory'
    copy_dataset(original, new_directory)