import os

def get_next_instance(class_label : str) -> str:
    """
    Returns the path of each file.

    Parameters:
    - class_label (str): Names of the folder with pictures.

    Returns:
    str
    """
    dataset_path = 'dataset'
    class_path = os.path.join(dataset_path, class_label)
    # Получаем список файлов в папке класса
    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    if files:
        for file in files:
            yield os.path.join(class_path, file)
    else:
        return None

if __name__ == "__main__":
    leopard_generator = get_next_instance('leopard')
    tiger_generator = get_next_instance('tiger')
    for i in range(5):
        leopard_instance = next(leopard_generator, None)
        tiger_instance = next(tiger_generator, None)

        if leopard_instance:
            print(f'Leopard instance: {leopard_instance}')
        if tiger_instance:
            print(f'Tiger instance: {tiger_instance}')