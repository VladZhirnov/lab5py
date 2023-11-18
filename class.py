import os

class InstanceIterator:
    def __init__(self, class_img):
        self.class_img = class_img
        self.dataset_path = 'dataset'
        self.class_path = os.path.join(self.dataset_path, self.class_img)
        self.files = [f for f in os.listdir(self.class_path) if os.path.isfile(os.path.join(self.class_path, f))]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.files):
            file = self.files[self.index]
            self.index += 1
            return os.path.join(self.class_path, file)
        else:
            raise StopIteration

if __name__ == "__main__":
    leopard_iterator = InstanceIterator('tiger')
    for instance in leopard_iterator:
        print(f'Leopard instance: {instance}')