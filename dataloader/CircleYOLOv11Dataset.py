import os
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class CircleYOLOv11Dataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=448, S=7, B=1, C=1, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[index])[0] + '.txt')

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = self.transform(image)

        # Initialize label grid
        label_grid = torch.zeros((self.S, self.S, self.C + 4 * self.B))  # e.g., [7, 7, 5] if C=1, B=1

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, r = map(float, line.strip().split())

                    grid_x = int(x * self.S)
                    grid_y = int(y * self.S)

                    if grid_x >= self.S: grid_x = self.S - 1
                    if grid_y >= self.S: grid_y = self.S - 1

                    x_cell = x * self.S - grid_x  # relative to cell
                    y_cell = y * self.S - grid_y

                    # Format: [p(class), conf, x_cell, y_cell, r]
                    label_grid[grid_y, grid_x, 0] = 1.0  # class probability (only 1 class)
                    label_grid[grid_y, grid_x, 1] = 1.0  # confidence
                    label_grid[grid_y, grid_x, 2:5] = torch.tensor([x_cell, y_cell, r])

        return image_tensor, label_grid
