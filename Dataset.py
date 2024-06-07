import os
import cv2

class ImageDatasetLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_filenames = os.listdir(dataset_dir)
        self.src_images = []

        self.load_dataset()

    def load_dataset(self):
        for filename in self.image_filenames:
            filepath = os.path.join(self.dataset_dir, filename)
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.src_images.append(image)

    def __getitem__(self, index):
        return self.src_images[index]

    def __len__(self):
        return len(self.src_images)
