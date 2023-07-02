import functools
import os
import re

import pandas

fluidflower_basedir = "C:/Users/bauerrn/Projekte/Fluidflower/"


def ffjoin(*args):
    return os.path.join(fluidflower_basedir, *args)


# name of group and path to dir where spatial maps reside
sparse_data_time_series_spatial_parameters_location = {
    "stuttgart": ffjoin("stuttgart"),
    "stanford": ffjoin("stanford"),
    "mit": ffjoin("mit"),
    "melbourne": ffjoin("melbourne"),
    "lanl": ffjoin("lanl"),
    "imperial": ffjoin("imperial"),
    "heriot-watt": ffjoin("modeling", "sparse_data", "heriot-watt"),
    "delft-DARSim": ffjoin("delft", "delft-DARSim"),
    "delft-DARTS": ffjoin("delft", "delft-DARTS"),
    "csiro": ffjoin("csiro"),
    "austin": ffjoin("austin")
}


search_files = {
    "sparse_data": "sparse_data.csv",
    "spatial_parameters": "spatial_parameters.csv",
    "time_series": "time_series.csv"
}

out_dir = "C:/Users/bauerrn/Data/Fluidflower"

expected_keys = {
    "p_1": "pressure [N/m^2] (sensor 1)",
    "p_2": "pressure [N/m^2] (sensor 2)",
    "mob_A": "mobile CO2 [g] in Box A",
    "imm_A": "immobile CO2 [g] in Box A",
    "diss_A": "dissolved CO2 [g] in Box A",
    "seal_A": "sealed CO2 [g] in Box A",
    "mob_B": "mobile CO2 [g] in Box B",
    "imm_B": "immobile CO2 [g] in Box B",
    "diss_B": "dissolved CO2 [g] in Box B",
    "seal_B": "sealed CO2 [g] in Box B",
    "M_C": "convection M [m] in Box C",
    "total_CO2_mass": "total CO2 [g] mass",
}


def handle_column_name(s: str, group: str):
    # strip leading or trailing spaces
    s = s.strip(" ")

    # strip #
    s = s.replace("#", "")

    # handle stanford
    if s == "p1":
        s = "p_1"
        return s

    if s == "p2":
        s = "p_2"
        return s

    if s == "totM_co2":
        s = "total_CO2_mass"

    if group == "mit":
        # remove trailing unit
        if "_" in s:
            s = s.rsplit("_", 1)[0]

        if len(s) > 1:
            # add _
            s = s[:-1] + "_" + s[-1]

        if s.startswith("immob"):
            s = "imm" + s[len("immob"):]

        if s == "M":
            s = "M_C"

        if s == "CO2to_t":
            s = "total_CO2_mass"

    if "tot" in s:
        s = "total_CO2_mass"

    if group == "heriot-watt":
        if s.startswith("p_bot"):
            s = "p_1"
        if s.startswith("p_top"):
            s = "p_2"

    s = s.strip(" ")

    return s


def main():
    available_files = functools.reduce(lambda a, b: a + b, [[(os.path.exists(os.path.join(sparse_data_time_series_spatial_parameters_location[group], search_files[file_name])), group, file_name)
                for file_name in search_files] for group in sparse_data_time_series_spatial_parameters_location])

    print("all available?", all(map(lambda x: x[0], available_files)))
    print("not available:", list(filter(lambda x: not x[0], available_files)))
    o = len(list(filter(lambda x: x[0], available_files)))
    print("available:", o)

    available_files = filter(lambda x: x[0], available_files)
    available_files = list(map(lambda x: (x[0], x[1], x[2],
                                          os.path.join(sparse_data_time_series_spatial_parameters_location[x[1]],
                                                       search_files[x[2]])), available_files))
    available_files.append((True, 'stanford', 'time_series', os.path.join(ffjoin('stanford'), "time_series_final.csv")))
    # available_files.append((True, 'mit', 'spatial_parameters', os.path.join(ffjoin('mit'), '')))  # not available
    available_files.append((True, 'heriot-watt', 'spatial_parameters', os.path.join(ffjoin('heriot-watt'),
                                                                                    'spatial_parameters.csv')))
    available_files.append((True, 'heriot-watt', 'time_series', ffjoin('heriot-watt', 'HWU-FinalTimeSeries.csv')))
    print("manually added:", len(list(filter(lambda x: x[0], available_files)))-o)

    # group by group
    paths = dict()
    for group in sparse_data_time_series_spatial_parameters_location:
        by_group = list(filter(lambda x: x[1] == group, available_files))
        paths[group] = dict()
        for file_name in search_files:
            by_file = list(filter(lambda x: x[2] == file_name, by_group))
            if group == "mit" and file_name == "spatial_parameters":
                continue
            assert len(by_file) == 1, f"{len(by_file)}, {file_name}, {group}, {list(by_group)}"
            paths[group][file_name] = by_file[0]

    # load files
    i = 0
    for group in paths:
        files = paths[group]
        for file_name in files:
            table = pandas.read_csv(files[file_name][3])

            if file_name == "time_series":
                table.columns = list(map(lambda s: handle_column_name(s, group), table.columns))

                for k in expected_keys:
                    if k not in table.columns:
                        print(f"{k} not in table for group {group}")

            # save files
            table.to_csv(os.path.join(out_dir, group, f"{file_name}.csv"))
            i += 1

    print(f"saved {i} many files")


if __name__ == '__main__':
    main()
