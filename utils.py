import numpy as np

def parse_pcd_file(filepath, num_points_expected=1024, leeway=16, dtype=np.float32):
    """
    Parse .pcd file of 3D points to numpy array of fixed size
    Args:
        filepath: path to .pcd file
        num_points_expected: npoints to return
        leeway: deviation allowed in number of points discovered in .pcd file
        dtype: data type of array to return
    Returns:
        dtype array of size (num_points_expected x 3)
    """
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()

    npoints = int(lines[9].split(' ')[1])
    if npoints > num_points_expected + leeway // 2:
        raise ValueError("Width is too large: npoints_expected=", num_points_expected, ", npoints=", npoints)
    if npoints < num_points_expected - leeway // 2:
        raise ValueError("Width is too small: npoints_expected=", num_points_expected, ", npoints=", npoints)

    points = np.zeros((num_points_expected, 3), dtype=dtype)
    for i in range(npoints):
        lines_idx = i + 11
        points[i] = lines[lines_idx].split(' ')

    for i in range(npoints, num_points_expected):
        points[i] = points[0]

    return points