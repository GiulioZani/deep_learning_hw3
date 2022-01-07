import numpy as np
import cv2
import torch


class ImgManager:
    def __init__(self, env):

        self.env = env
        self.ROWS = 80
        self.COLS = 100
        self.REM_STEP = 4
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)

    def get_image(self):
        img = self.env.render(mode='rgb_array')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS),
                                     interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis=0)
        self.image_memory[0, :, :] = img_rgb_resized

        return torch.tensor(np.expand_dims(self.image_memory, axis=0)).float()
