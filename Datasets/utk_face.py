from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch import Tensor
import glob
import numpy as np

class UTKFaceDataset(Dataset):
    def __init__(self, dir="UTKFace"):
        self.dir = dir
        self.transform = (lambda x : x.float() / 255.)
        self.target_transform = lambda arr : Tensor([int(x) for x in arr])
        self.paths = glob.glob(self.dir + "/*")

    def __len__(self):
        return len(self.paths)

    # label = [age, gender, race], unused date&time
    # From UTKFace description:
    # [age] is an integer from 0 to 116, indicating the age
    # [gender] is either 0 (male) or 1 (female)
    # [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    # unused [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
    def __getitem__(self, idx):
        img_path: str = self.paths[idx]
        image = read_image(img_path)
        label = img_path[len(self.dir)+1].split("_")[:3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label