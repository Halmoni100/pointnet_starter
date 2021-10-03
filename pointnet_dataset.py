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
            classpath = os.path.join(toppath, classdir)
            files = os.listdir(classpath)
            files.sort()
            for filename in files:
                if not filename.endswith(".pcd"):
                    raise ValueError("Found non .pcd file in toppath: ", filename)
                filepath = os.path.join(classpath, filename)
                self.points.append(parse_pcd_file(filepath,
                                                  dtype=dtype))
                self.labels.append(np.array(i, dtype=dtype))

    def __getitem__(self, index):
        return self.points[index], self.labels[index]

    def __len__(self):
        return len(self.points)