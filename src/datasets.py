import os
import glob
import random
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, opt, unaligned=True, mode="train"):
        self.unaligned = unaligned
        self.transform = self.get_transform(opt, mode)
        
        self.files_A = sorted(glob.glob(os.path.join(opt.data_source, "%s/blurry" % mode) + "/*/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(opt.data_source, "%s/sharp" % mode) + "/*/*.*"))
#         self.files_A = sorted(glob.glob(os.path.join(opt.data_source, "%s/blurry" % mode) + "/*.*"))
#         self.files_B = sorted(glob.glob(os.path.join(opt.data_source, "%s/sharp" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def get_transform(self, opt, mode):
        if mode == "train":
            transforms_ = [
                transforms.Resize(int(max(opt.cropX, opt.cropY)*1.5)),
                transforms.RandomCrop((opt.cropX, opt.cropY)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        elif mode == "test":
            transforms_ = [
                transforms.Resize((opt.resizeX, opt.resizeY)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        else:
            raise Exception('No such mode!')
        
        return transforms.Compose(transforms_)
    