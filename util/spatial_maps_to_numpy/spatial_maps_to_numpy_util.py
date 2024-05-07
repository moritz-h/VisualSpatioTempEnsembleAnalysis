# translate stuttgart spatial maps data to numpy array
import os
import csv
import re

import numpy as np


seconds_pattern = re.compile(r""".*spatial_map_(?P<seconds>\d+)s?(?!h)\.csv$""")


# extract seconds from filename
def filename_to_seconds(f):
    s = seconds_pattern.match(f)
    sec = s.group("seconds")
    return int(sec)


def spatial_maps_to_numpy(filenames: list):
    """
    Provided a list of filenames, each a path to a spatial_map(some_seconds)s.csv file, parse file to array and
        concatenate all arrays to a data matrix.

    Returns (t, y, x, data) where
        t: np.array of shape (amount_timesteps, ) containing the time for each entry belonging to data.shape[0]
        y: np.array of shape (height, ) containing the y position for each entry belonging to data.shape[1]
        x: np.array of shape (width, ) containing the x position for each entry belonging to data.shape[2]
        data: np.array of shape (amount_timesteps, height, width, 2) containing the (saturation, concentration) value
            for each entry of data[:, :, :]
    :param filenames:
    :return:
    """
    assert len(filenames) > 0, "got empty filenames list"

    # sort by seconds
    filenames = sorted(filenames, key=filename_to_seconds)

    # read data
    t = np.array([filename_to_seconds(s) for s in filenames])  # seconds
    data = []

    x_np = np.array([])
    y_np = np.array([])

    for j, filename in enumerate(filenames):
        with open(filename, "rt") as f:
            csvreader = csv.reader(f, delimiter=',')
            x = []
            y = []
            saturation = []
            concentration = []

            for i, row in enumerate(csvreader):
                if i == 0:
                    continue  # skip header

                x.append(float(row[0]))
                y.append(float(row[1]))
                saturation.append(float(row[2]))
                concentration.append(float(row[3]))

            x = np.array(x)
            y = np.array(y)

            if j == 0:
                x_np = x
                y_np = y
            else:
                assert (x == x_np).all() and (y == y_np).all(), "spatial maps have different samples"

            saturation = np.array(saturation)
            concentration = np.array(concentration)

            data.append(np.stack([saturation, concentration], -1))

    # reshape arrays
    x_new = sorted(list(set(x_np.tolist())))
    y_new = sorted(list(set(y_np.tolist())))

    # return the index to value
    idx_lookup_x = {v: idx for idx, v in enumerate(x_new)}
    idx_lookup_y = {v: idx for idx, v in enumerate(y_new)}

    variables = [saturation, concentration]

    data_mat = np.empty((len(t), len(y_new), len(x_new), len(variables)), dtype=float)

    # fill data_mat
    for i, dt in enumerate(data):
        for j in range(len(dt)):
            # get indices
            x_idx = idx_lookup_x[x_np[j]]
            y_idx = idx_lookup_y[y_np[j]]

            data_mat[i, y_idx, x_idx] = dt[j]

    return t, np.array(y_new), np.array(x_new), np.array(data_mat)


def save_spatial_maps_data(t: np.array, y: np.array, x: np.array, data: np.array, out_dir: str,
                           out_base_filename: str = "spatial_maps"):
    print("saving spatial maps to out dir '{}'".format(out_dir))

    if not os.path.exists(out_dir):
        print("out_dir does not exist, create out dir")
        os.mkdir(out_dir)

    # save data as numpy arrays
    np.save(os.path.join(out_dir, out_base_filename + "_t.npy"), t)
    np.save(os.path.join(out_dir, out_base_filename + "_x.npy"), x)
    np.save(os.path.join(out_dir, out_base_filename + "_y.npy"), y)
    np.save(os.path.join(out_dir, out_base_filename + "_sat_con.npy"), data)

