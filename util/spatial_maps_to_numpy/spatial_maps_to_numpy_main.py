from spatial_maps_to_numpy_util import spatial_maps_to_numpy, save_spatial_maps_data
import os
import re

import sys


print("got args:", sys.argv)


assert len(sys.argv) == 3, "expecting first argument to be the data directory of the cloned repositories and the " \
                           "second to be the data directory to store the data"

assert os.path.exists(sys.argv[1]) and os.path.isdir(sys.argv[1]), \
    "source data path must be an existing directory"
assert os.path.exists(sys.argv[2]) and os.path.isdir(sys.argv[2]), \
    "destination data path must be an existing directory"

fluidflower_basedir = sys.argv[1]  # "C:/Users/bauerrn/Projekte/Fluidflower/"
out_dir = sys.argv[2]  # "C:/Users/bauerrn/Data/Fluidflower"


def ffjoin(*args):
    return os.path.join(fluidflower_basedir, *args)


# name of group and path to dir where spatial maps reside
spatial_maps_locations = {
    "stuttgart": ffjoin("stuttgart", "spatial_maps_first_day"),
    "stanford": ffjoin("stanford", "spatial_maps"),
    # "mit": ffjoin("mit")  # does not have any
    "melbourne": ffjoin("melbourne", "Spatial maps"),
    "lanl": ffjoin("lanl"),
    # "imperial": ffjoin("imperial"),
    "heriot-watt": ffjoin("heriot-watt", "spatial_map_10min"),
    "delft-DARSim": ffjoin("delft", "delft-DARSim"),
    "delft-DARTS": ffjoin("delft", "delft-DARTS"),
    "csiro": ffjoin("csiro", "spatial_map_10mins"),
    "austin": ffjoin("austin", "spatial_maps")
}


def get_spatial_map_filenames_from_dir(path_to_dir):
    # returns the spatial map filenames which end on a second
    # only returns the name of the file, not the full path
    pattern = re.compile(r""".*spatial_map_\d+s?(?!h)\.csv$""")

    re.match(pattern, "")

    # returns the spatial map filenames in the provided directory
    return [f for f in os.listdir(path_to_dir) if pattern.match(f)]


out_base_filename = "spatial_maps"


def main():
    print("collecting spatial maps and saving them as numpy arrays for groups: {}".format(
        list(spatial_maps_locations.keys())))

    for group in spatial_maps_locations:
        print("going for group '{}' with location '{}'".format(group, spatial_maps_locations[group]))

        # debug: skip if files already exist
        #if os.path.exists(os.path.join(out_dir, group)):
        #    continue

        filenames = list(map(lambda x: os.path.join(spatial_maps_locations[group], x),
                             get_spatial_map_filenames_from_dir(spatial_maps_locations[group])))

        t, y, x, data = spatial_maps_to_numpy(filenames)

        save_spatial_maps_data(t, y, x, data, os.path.join(out_dir, group), out_base_filename)

    print("finished")


if __name__ == '__main__':
    main()
