from torch.utils.data import Dataset
import cv2
import torch
import pdb

class TrafficSign(Dataset):
    def __init__(self, args, img_paths, labels, mode):
        self.mode = mode
        self.img_paths = img_paths
        self.labels = labels
        self.img_paths = img_paths

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(f'{img_path}')
        if self.mode in ['train', 'val']:
            label = self.labels[idx]
            #pdb.set_trace()
            return {'path': img_path, 'img' : img, 'label': torch.tensor(label)}
        else:
            img_name = img_path.split('/')[-1]
            return {'path': img_path, 'img' : img, 'img_name' : img_name}

    def __len__(self):
        return len(self.img_paths)