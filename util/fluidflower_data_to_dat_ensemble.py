from s4_util.PythonExtras import volume_tools as vt
import numpy as np
import os
import sys


print("got args:", sys.argv)


assert len(sys.argv) == 3, "expecting first argument to be the source directory and second to be the destination"


assert os.path.exists(sys.argv[1]) and os.path.isdir(sys.argv[1]), \
    "source data path must be an existing directory"
assert os.path.exists(sys.argv[2]) and os.path.isdir(sys.argv[2]), \
    "destination data path must be an existing directory"

source_dir = sys.argv[1]  # "./data"
fname_fmt = "spatial_maps_{}.npy"
out_root_dir = sys.argv[2]  # "./data"

out_dir_readme_fname = "README.md"

out_dir_readme_description_fmt = """
# Description

ensemble data that is not normalized or stored as uchar and all ensembles were aligned.

Temporally: all are clipped to 144 time steps.
If they are shorter, the missing time steps are being filled with repeated last time step.

Spatially:
They are centered then clipped to fit the smallest one.

This only contains the {} values.
"""


def add_z_axis(d: np.array):
    d = d.reshape((d.shape[0], 1, *d.shape[1:]))  # add z axis
    return d


def store_as_dat_sequence(d: np.array, out_file: str):
    vt.write_volume_sequence(out_file, d, clearDir=False, dataFormat=vt.VolumeMetadata.Format.datRaw)


def main():
    # convert data for each variable and the combination individually
    for v in ["sat", "con", "sat_con"]:
        print(f"processing {v}...")
        groups = os.listdir(source_dir)
        groups = list(
            filter(lambda s: not s.startswith("ensemble") and not s.startswith(".") and not s.startswith("experiment"),
                   groups))

        data = {group: np.load(os.path.join(source_dir, group, fname_fmt.format(v))) for group in groups}

        # only add z axis (as it is rescaled)
        for group in data:
            data[group] = add_z_axis(data[group])

        # change dtype to np.uint or np.float32
        for group in data:
            data[group] = data[group].astype(np.float32)

        # create out dir if it does not exist
        out_dir = os.path.join(out_root_dir, "ensemble_{}".format(v))
        os.makedirs(out_dir, exist_ok=True)

        # store README.md
        with open(os.path.join(out_dir, out_dir_readme_fname), "wt") as f:
            f.write(out_dir_readme_description_fmt.format(v))

        # store dat ensemble
        if v == "sat":
            accessor = slice(0, 1)
        elif v == "con":
            accessor = slice(1, 2)
        else:
            # both
            accessor = slice(0, 2)

        for group in data:
            store_as_dat_sequence(data[group][:, :, :, :, accessor], os.path.join(out_dir, group))

    print("done")


if __name__ == '__main__':
    main()

