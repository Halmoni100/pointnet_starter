#!/usr/bin/env python3

import os, sys

dirpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirpath, 'torch_helper'))
import data_prep

transformed_data_path = os.path.join(dirpath, "../pointnet_custom_shapes/data/transformed")
split_data_path = os.path.join(dirpath, "data/transformed_split")
data_prep.rm_and_mkdir(split_data_path)
data_prep.split_data(transformed_data_path, split_data_path, ratio=0.2)