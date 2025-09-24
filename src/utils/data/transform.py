import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

class Resize:
    def __init__(self, image_size):
        # size: int for square images
        self.image_size = image_size

    def __call__(self, image):
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        image = TF.resize(image, (self.image_size, self.image_size))
        image = TF.to_tensor(image)
        image.permute(1, 2, 0)
        return image
