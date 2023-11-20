import os
import shutil
import csv
import random

def copy_dataset_with_random_numbers(original : str) -> None:
    """
    Creates a copy of the dataset folder and an annotation for it.

    Parameters:
    - original (str): Path to root directory.

    Returns:
    None
    """
    new_directory = f'copy_{original}'
    os.makedirs(new_directory, exist_ok=True)
    output_file = 'annotation2.csv'
    used_random_numbers = set()
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Absolute path', 'Relative path', 'Class'])
        for root, dirs, files in os.walk(original):
            for file in files:
                class_label = os.path.basename(root)
                random_number = None
                while random_number is None or random_number in used_random_numbers:
                    random_number = random.randint(0, 10000)
                used_random_numbers.add(random_number)
                new_filename = f"{random_number}.jpg"
                source_path = os.path.join(root, file)
                relative_path = os.path.join(new_directory, new_filename)
                shutil.copy(source_path, relative_path)
                absolute_path = os.path.abspath(relative_path)
                csv_writer.writerow([absolute_path, relative_path, class_label])

if __name__ == "__main__":
    original_dataset = 'dataset'
    copy_dataset_with_random_numbers(original_dataset)