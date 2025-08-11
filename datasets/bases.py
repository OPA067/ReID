from torch.utils.data import Dataset
from utils.iotools import read_image

def create_dataset_list(dataset):
    nums = len(dataset)
    dataset_copy = dataset.copy()
    can_images = [i[2] for i in dataset_copy]
    tar_images = [i[1] for i in dataset_copy]
    id = [i[0] for i in dataset_copy]

    for i in range(nums):
        tmp = (id[i], tar_images[i], can_images[i])
        dataset[i] = tmp

    return dataset

class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 my_aug_img=False):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
        self.my_aug_img = my_aug_img

        self.dataset = create_dataset_list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        id, tar_path, can_path = self.dataset[index]

        tar_img = read_image(tar_path)
        can_img = read_image(can_path)

        if self.transform is not None:
            tar_img = self.transform(tar_img)
            can_img = self.transform(can_img)

        ret = {
            'id': id,
            'tar_img': tar_img,
            'can_img': can_img,
            'index': index,
        }

        return ret