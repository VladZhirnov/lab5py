import random
import os

class InstanceIterator:
    def __init__(self, class_label = None, dataset_path = None) -> None:
        self.class_label = class_label
        self.dataset_path = dataset_path
        self.class_path = os.path.join(self.dataset_path, class_label)
        self.instances = self.get_instances()

    def get_instances(self) -> list:
        if not os.path.exists(self.class_path):
            print(f"Папка {self.class_label} не найдена.")
            return None
        instances = os.listdir(self.class_path)
        random.shuffle(instances)
        return instances
    
    def __iter__(self):
        return self

    def __next__(self) -> str:
        if not self.instances:
            raise StopIteration("Экземпляры закончились.")
        return os.path.join(self.class_path, self.instances.pop(0))