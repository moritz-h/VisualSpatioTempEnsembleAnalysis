import os
import pathlib
import re
import numpy as np
import pandas
from pathlib import Path


fluidflower_basedir = "C:/Users/bauerrn/Projekte/Fluidflower/"

experiment_group_name = "experiment"

experiment_group_dir = os.path.join(fluidflower_basedir, experiment_group_name)
experiment_spatial_maps_data_dir = os.path.join(experiment_group_dir, "benchmarkdata", "spatial_maps")

out_dir = "C:/Users/bauerrn/Data/Fluidflower"
out_base_filename = "segmentation"

seconds_pattern = re.compile(r""".*segmentation_(?P<seconds>\d+)s?(?!h)\.csv$""")


def get_segmentation_map_filenames_from_dir(path_to_dir):
    # returns the segmentation map filenames which end on a second
    # only returns the name of the file, not the full path
    pattern = re.compile(r""".*segmentation_\d+s?(?!h)\.csv$""")

    re.match(pattern, "")

    # returns the spatial map filenames in the provided directory
    return [f for f in os.listdir(path_to_dir) if pattern.match(f)]


# extract seconds from filename
def filename_to_seconds(f):
    s = seconds_pattern.match(f)
    sec = s.group("seconds")
    return int(sec)


def segmentation_maps_to_numpy(segmentation_files: list):
    assert len(segmentation_files) > 0, "got empty filenames list"

    # sort by seconds
    filenames = sorted(segmentation_files, key=filename_to_seconds)

    # read data
    t = np.array([filename_to_seconds(s) for s in filenames])  # seconds

    data = []

    binary_set = {0, 1}

    for j, filename in enumerate(filenames):
        # read .csv file
        d = pandas.read_csv(filename, sep=',', comment="#")

        # get concentration
        con = d.to_numpy(int, True)
        con[con == 2] = 1  # set saturation to also be concentration
        assert set(np.unique(con)).issubset(binary_set), "con has to be binary!"

        # get saturation
        sat = d.to_numpy(int, True)
        sat[sat == 1] = 0  # remove concentration
        sat[sat == 2] = 1  # set saturation 2 to just 1 to keep binary
        assert set(np.unique(sat)).issubset(binary_set), "sat has to be binary!"

        data.append((sat, con))

    # merge data into one numpy array
    shapes = set()
    for d in data:
        for d_sc in d:
            shapes.add(d_sc[0].shape)

    assert len(shapes) == 1, f"got unequal shapes: {shapes}"

    data = [np.stack(d_sc, -1) for d_sc in data]  # stack all tuples of (sat, con)
    data = np.stack(data, 0)  # stack all stacked tuples to one big array

    # invert y axis (bottom is y=0)
    data = data[:, ::-1, :]

    return t, data


def save_segmentation_maps_data(t: np.array, data: np.array, out_dir: str, filename: str):
    print("saving segmentation maps to out dir '{}'".format(out_dir))

    # create dir
    path_outdir = Path(out_dir)
    path_outdir.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(out_dir, filename + "_t.npy"), t)
    np.save(os.path.join(out_dir, filename + "_sat_con.npy"), data)


def main():
    runs = os.listdir(experiment_spatial_maps_data_dir)

    print(f"found {len(runs)} many runs: {runs}")

    # load data per run
    for run in runs:
        run_dir_path = os.path.join(experiment_spatial_maps_data_dir, run)
        print(f"reading segmentation maps from run dir: {run_dir_path}")
        segmentation_files = get_segmentation_map_filenames_from_dir(run_dir_path)

        filenames = list(map(lambda x: os.path.join(run_dir_path, x), segmentation_files))

        t, data = segmentation_maps_to_numpy(filenames)

        save_segmentation_maps_data(t, data, os.path.join(out_dir, experiment_group_name, run), out_base_filename)


if __name__ == '__main__':
    main()
