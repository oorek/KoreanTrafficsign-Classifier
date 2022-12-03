from torch.utils.data import Dataset
import cv2
import torch
import pdb
import albumentations as A

class TrafficSign(Dataset):
    def __init__(self, args, img_paths, labels, mode):
        self.mode = mode
        self.img_paths = img_paths
        self.labels = labels
        self.img_paths = img_paths
        self.img_size = args.img_size
        self.transform = A.Compose([A.Resize(self.img_size, self.img_size)])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(f'{img_path}')
        img = self.transform(image=img)

        if self.mode in ['train', 'val']:
            label = self.labels[idx]
            #pdb.set_trace()
            return {'path': img_path, 'img' : img['image'], 'label': torch.tensor(label)}
        else:
            img_name = img_path.split('/')[-1]
            return {'path': img_path, 'img' : img['image'], 'img_name' : img_name}

    def __len__(self):
        return len(self.img_paths)