"""
Base Dataset class for the Image-to-Image person ReID pipeline.

This module defines the ImageTextDataset (a torch.utils.data.Dataset) that
wraps the raw annotation list into a PyTorch-compatible dataset. Each sample
consists of a target image, a candidate image, and the associated person ID.
"""

from torch.utils.data import Dataset
from utils.iotools import read_image


def create_dataset_list(dataset):
    """
    Reorganize raw dataset entries into a uniform (id, tar, can) tuple format.

    The input dataset may contain entries in an older order; this function
    ensures compatibility with the indexed access pattern used by the DataLoader.

    Args:
        dataset (list): Raw dataset list.

    Returns:
        list: Reorganized list of (id, tar_path, can_path) tuples.
    """
    nums = len(dataset)
    dataset_copy = dataset.copy()
    pid_list = [i[0] for i in dataset_copy]
    tar_images = [i[1] for i in dataset_copy]
    can_images = [i[2] for i in dataset_copy]

    for i in range(nums):
        tmp = (pid_list[i], tar_images[i], can_images[i])
        dataset[i] = tmp

    return dataset


class ImageTextDataset(Dataset):
    """
    PyTorch Dataset for image-based person re-identification.

    Each item provides:
        - 'id':      The unique person identity label (int).
        - 'tar_img': Preprocessed target (query) image tensor.
        - 'can_img': Preprocessed candidate (gallery) image tensor.
        - 'index':   The dataset index of this sample.

    Args:
        dataset: Raw annotation list (from RSTPReid._process_json).
        args:    Parsed arguments namespace.
        transform: torchvision.transforms pipeline (optional).
        text_length: Maximum token length for text inputs (unused here but kept for compatibility).
        truncate: Whether to truncate text during tokenization (compatibility flag).
        my_aug_img: Whether to use custom image augmentation (currently unused).
    """

    def __init__(self, dataset, args, transform=None, text_length: int = 77, truncate: bool = True, my_aug_img=False):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
        self.my_aug_img = my_aug_img

        self.dataset = create_dataset_list(dataset)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieve a single data sample.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'id' (int): Person identity label.
                - 'tar_img' (Tensor): Preprocessed target image.
                - 'can_img' (Tensor): Preprocessed candidate image.
                - 'index' (int): Dataset index.
        """
        pid, tar_path, can_path = self.dataset[index]

        tar_img = read_image(tar_path)
        can_img = read_image(can_path)

        if self.transform is not None:
            tar_img = self.transform(tar_img)
            can_img = self.transform(can_img)

        ret = {
            'id': pid,
            'tar_img': tar_img,
            'can_img': can_img,
            'index': index,
        }

        return ret