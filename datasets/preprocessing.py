"""
Image-based data preprocessing and augmentation utilities.

This module implements the "Random Erasing" data augmentation technique,
as proposed in:
    Zhong et al., "Random Erasing Data Augmentation", 2017.
    https://arxiv.org/pdf/1708.04896.pdf

Random Erasing randomly selects a rectangular region in an image and replaces
its pixels with a constant value, simulating occlusion and improving model
robustness.
"""

import random
import math


class RandomErasing(object):
    """
    Randomly selects a rectangular region in an image and erases its pixels.

    Random erasing helps the model learn to recognize objects even when parts
    of the image are occluded, acting as a simple yet effective form of
    data augmentation for person ReID.

    Args:
        probability (float): Probability that the operation will be applied.
        sl (float): Minimum proportion of erased area relative to image area.
        sh (float): Maximum proportion of erased area relative to image area.
        r1 (float): Minimum aspect ratio of the erased region.
        mean (tuple): RGB values used to fill the erased region.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        Apply random erasing to a tensor image.

        Args:
            img (Tensor): Image tensor of shape [C, H, W].

        Returns:
            Tensor: The image with a random region erased (or unchanged).
        """
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
