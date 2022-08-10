import os
import glob
import natsort

import cv2
from torch.utils.data import Dataset


class ImageDirDataset(Dataset):
    """Build a dataset out of an image directory.

    Args:
        root (string): root directory where dataset is located
        transform (callable, optional): a function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        _, __: unused arguments (fit LitePose format)
    """

    def __init__(self, root, _, __, transform):
        self.name = 'image_dir'
        self.root = root
        self.transform = transform
        self.types = ('*.jpg', '*.png')
        all_imgs = []
        for files in self.types:
            all_imgs.extend([os.path.basename(x) for x in glob.glob(os.path.join(self.root, files))])
        self.all_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.all_imgs[idx])
        img = cv2.imread(img_loc, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img
