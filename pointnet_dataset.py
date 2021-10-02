import os
import numpy as np
from torch.utils.data import Dataset

from utils import parse_pcd_file

class PointNetDataset(Dataset):
    def __init__(self, toppath, dtype=np.float32):
        classdirs = os.listdir(toppath)
        classdirs.sort()
        self.points = []
        self.labels = []
        for i in range(len(classdirs)):
            classdir = classdirs[i]
            files = os.listdir(classdir)
            for filename in files:
                if filename.split('.')[-1] != ".pcd":
                    raise ValueError("Found non .pcd file in toppath")
                filepath = os.path.join(toppath, filename)
                self.points.append(parse_pcd_file(filepath,
                                                  dtype=dtype))
                self.labels.append(i)

    def __getitem__(self, index):
        return self.points[index], self.labels[index]

    def __len__(self):
        return len(self.points)