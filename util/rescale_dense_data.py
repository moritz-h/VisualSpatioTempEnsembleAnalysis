import os
import numpy as np
import pandas
import pybrutil.np_util.data

groups = ['stuttgart', 'stanford', 'melbourne', 'lanl', 'imperial', 'heriot-watt',
          'delft-DARSim', 'delft-DARTS', 'csiro', 'austin']

experimental = "experiment"

ffdir = "C:/Users/bauerrn/Data/Fluidflower"
ffdir_out = "C:/Users/bauerrn/Data/Fluidflower_rescaled"


def load_data():
    data = {}
    for group in groups:
        gd = {}
        for suffix in ["t", "x", "y", "sat_con"]:
            gd.update({
                suffix: np.load(os.path.join(ffdir, group, "spatial_maps_" + suffix + ".npy"))
            })

        # load time_series
        gd.update({
            "time_series": pandas.read_csv(os.path.join(ffdir, group, "time_series.csv"))
        })

        data[group] = gd

    # load experimental
    runs = os.listdir(os.path.join(ffdir, experimental))
    for run in runs:
        gd = {}
        for suffix in ["t", "sat_con"]:
            gd.update({
                suffix: np.load(os.path.join(ffdir, experimental, run, "segmentation_" + suffix + ".npy"))
            })

        data[experimental + "_" + run] = gd

    return data


def save_data(data: dict):
    for group in data:
        print(f"save group: {group}")

        out_dir = os.path.join(ffdir_out, group)
        if not os.path.exists(out_dir):
            print(f"create out dir: {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

        # write dense data
        if group.startswith(experimental):
            prefix = "segmentation_"
        else:
            prefix = "spatial_maps_"

        np.save(os.path.join(out_dir, prefix + "sat_con.npy"), data[group]["sat_con"])

        # write other data
        np.save(os.path.join(out_dir, prefix + "t.npy"), data[group]["t"])
        np.save(os.path.join(out_dir, prefix + "y.npy"), data[group]["x"])
        np.save(os.path.join(out_dir, prefix + "x.npy"), data[group]["y"])

        # write time series data frame
        if not group.startswith(experimental):
            data[group]["time_series"].to_csv(os.path.join(out_dir, "time_series.csv"))

    print("done saving")


def get_range(x):
    return [min(x), max(x)]


def main():
    data = load_data()

    # need to hard code since they didn't provide x and y values
    experimental_x_range = [0.035, 2.825]
    experimental_y_range = [0.035, 1.515]

    step_xy = 0.01
    expected_range_x = [0.005, 2.855]  # inclusive
    expected_x_values = np.arange(start=expected_range_x[0], stop=expected_range_x[1]+step_xy, step=step_xy)

    expected_range_y = [0.005, 1.225]  # inclusive
    expected_y_values = np.arange(start=expected_range_y[0], stop=expected_range_y[1] + step_xy, step=step_xy)

    step_t = 600
    expected_range_t = [0, 60 * 60 * 24]  # inclusive  # first 6 hours in 10 minute intervals
    expected_t_values = np.arange(start=expected_range_t[0], stop=expected_range_t[1] + step_t, step=step_t)

    n_variables = 2
    rescaled_data_shape = (len(expected_t_values), len(expected_y_values), len(expected_x_values), n_variables)

    rescaled_data = dict()
    for group in data:
        sat_con = np.zeros(rescaled_data_shape, dtype=float)
        orig_sat_con = data[group]["sat_con"]

        # compute offsets of to be filled
        orig_x_range = get_range(data[group]["x"]) if not group.startswith(experimental) else experimental_x_range
        orig_y_range = get_range(data[group]["y"]) if not group.startswith(experimental) else experimental_y_range
        orig_t_values = data[group]["t"]

        x_offset_tbf = max(0, round((orig_x_range[0] - expected_range_x[0]) / step_xy))
        y_offset_tbf = max(0, round((expected_range_y[0] - orig_y_range[0]) / step_xy))

        x_offset_orig = max(0, round((expected_range_x[0] - orig_x_range[0]) / step_xy))
        y_offset_orig = max(0, round((orig_y_range[0] - expected_range_y[0] ) / step_xy))

        prev_t_j = None
        for t_i, t in enumerate(expected_t_values):
            if t not in orig_t_values:
                if prev_t_j is None:
                    continue
                else:
                    pybrutil.np_util.data.fill_array_with(sat_con[t_i], orig_sat_con[prev_t_j,
                                                                        y_offset_orig:y_offset_orig + len(
                                                                            expected_y_values),
                                                                        x_offset_orig:x_offset_orig + len(
                                                                            expected_x_values)],
                                                          offsets=[y_offset_tbf, x_offset_tbf, 0])
            else:
                t_j = orig_t_values.tolist().index(t)
                prev_t_j = t_j
                pybrutil.np_util.data.fill_array_with(sat_con[t_i], orig_sat_con[t_j, y_offset_orig:y_offset_orig+len(expected_y_values), x_offset_orig:x_offset_orig+len(expected_x_values)], offsets=[y_offset_tbf, x_offset_tbf, 0])

        rescaled_data[group] = {
            "sat_con": sat_con
        }

        for v, values in [("x", expected_x_values), ("y", expected_y_values), ("t", expected_t_values)]:
            rescaled_data[group][v] = values

        # trim time series
        if not group.startswith(experimental):  # no time series data for experimental groups
            time_series = data[group]["time_series"]

            rescaled_time_series = pandas.DataFrame(columns=time_series.columns, index=range(len(expected_t_values)))
            rescaled_time_series.t = expected_t_values

            prev_t_j = None
            for t_j, t in enumerate(expected_t_values):
                if t not in orig_t_values or any([np.isnan(v) for v in time_series.iloc[t_j]]):
                    if prev_t_j is None:
                        for j, column in enumerate(rescaled_time_series.columns):
                            if column == "t":
                                continue
                            rescaled_time_series.iloc[t_j, j] = time_series.iloc[0, j]  # fill up first values with first values
                    else:
                        for j, column in enumerate(rescaled_time_series.columns):
                            if column == "t":
                                continue
                            rescaled_time_series.iloc[t_j, j] = time_series.iloc[prev_t_j, j]
                else:
                    t_k = orig_t_values.tolist().index(t)
                    t_k = min(len(expected_t_values), t_k)
                    prev_t_j = t_k
                    for j, column in enumerate(rescaled_time_series.columns):
                        if column == "t":
                            continue
                        rescaled_time_series.iloc[t_j, j] = time_series.iloc[prev_t_j, j]

            rescaled_data[group]["time_series"] = rescaled_time_series

    save_data(rescaled_data)


if __name__ == '__main__':
    main()

