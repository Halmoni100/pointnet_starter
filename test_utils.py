#!/usr/bin/env python3

from utils import parse_pcd_file

points = parse_pcd_file("/home/mrincredible/temp/test_cube.pcd")
print(points[0:3])
print(points[-6:-1])