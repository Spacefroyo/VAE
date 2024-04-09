from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch import Tensor
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class UTKFaceDataset(Dataset):
    def __init__(self, dir="UTKFace"):
        self.dir = dir
        self.transform = (lambda x : x.float() / 255.)
        mins, maxs = [0.0, 0.0, 0.0, 2017010997264384.0], [116.0, 1.0, 4.0, 2.0170116468781875e+17]
        self.target_transform = lambda arr : torch.tensor([(float(arr[i]) - mins[i]) / maxs[i] for i in range(len(arr))])
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

        prefix = "UTKFace/"
        postfix = ".jpg.chip.jpg"
        label = img_path[len(prefix):-len(postfix)].split("_")
        if (len(label) != 4):
            label.insert(1, 0.5) # uncertain gender
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
dataset = UTKFaceDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                        shuffle=True, num_workers=0)
data_iter = iter(dataloader)

mins, maxs = [np.inf] * 4, [0] * 4
for _ in range(len(dataset)):
    images, label = next(data_iter)
    if (images == None): break
    label = label[0]
    if label.shape[0] != 4:
        print(label)
    for i in range(len(mins)):
        mins[i] = min(mins[i], label[i].item())
        maxs[i] = max(maxs[i], label[i].item())

print(mins, maxs)