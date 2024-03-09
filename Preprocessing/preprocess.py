from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import glob

class UTKFaceImageDataset(Dataset):
    def __init__(self, dir="UTKFace", target_transform=None):
        self.dir = dir
        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.target_transform = target_transform
        self.len = len(glob.glob(self.dir))

    def next_file(self):
        for file in glob.glob(self.dir):
            yield file

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.next_file()
        image = read_image(img_path)
        label = img_path.split("_", ".")[:4]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label