import os
import time

import pandas
from PySide6.QtWidgets import QApplication
from scipy.interpolate import RegularGridInterpolator
import numpy as np

from PySide6.QtDataVisualization import QCustom3DVolume
from PySide6.QtGui import QColor, QImage

from threading import Thread

from fastdtw.fastdtw import fastdtw


def is_power_of_x(a: np.array, x):
    """
    Checks if a is a power of x and returns True or False for each entry of a or a itself.
    :param a:
    :param x:
    :return:
    """
    res = np.log(a) / np.log(x)
    res_i = res.astype(int)
    r = res - res_i
    return r == 0


def resize_to_power_of_two_and_scale_to_uint8(d: np.array, scale_to_uint8=False, reverse_y=True,
                                              interpolate=False) -> np.array:
    """
    Upscales the image and returns
    :param d:
    :param scale_to_uint8: forces the rescaling to uint8. Note: dtype stays int (uint8 takes forever when setting it to texture)
    :param reverse_y: reverse the y axis
    :param interpolate: if True: interpolates the array to the new shape -> returned shape is new shape.
        if False: only fills the array into the new shape (keep padded zeros) -> returned shape is old shape
    :return: returns the new array and the shape of the actual data in the new array. ignores padded zeros.
    """
    old_shape = d.shape
    if scale_to_uint8:
        d = np.interp(d, (d.min(), d.max()), (0, 255)).astype(d.dtype)

    if reverse_y:
        d = d[:, ::-1, :]

    if interpolate:
        d = upsample_array_to_power_of_two(d, "linear")
        return d, d.shape
    else:
        d_ = np.zeros(np.power(2, np.ceil(np.log2(d.shape))).astype(int), dtype=d.dtype)
        d = fill_array_with(d_, d)
        return d, old_shape


def fill_array_with(to_be_filled: np.array, data: np.array, offsets=None):
    """
    Fills the array to_be_filled with the data of the array data starting at provided offsets in to_be_filled
    :param to_be_filled:
    :param data:
    :param offsets:
    :return:
    """

    if offsets is None:
        offsets = np.zeros((len(to_be_filled.shape),), dtype=int)

    selector = tuple([slice(offsets[i], offsets[i] + data.shape[i]) for i in range(len(data.shape))])

    to_be_filled[selector] = data[:]

    return to_be_filled


def upsample_array_to_power_of_two(array: np.array, method="linear"):
    """
    Upsamples the provided array to the next shape which has a power of two for every dimension.
    Uses RegularGridInterpolator with the specified method.
    :returns
    :param array:
    :param method:
    :return:
    """
    upsampled_shape = np.power(2, np.ceil(np.log2(array.shape))).astype(int)

    return resample_array_to_shape(array, upsampled_shape, method)


def resample_array_to_shape(array: np.array, new_shape, method="linear"):
    """
    Resamples the array to the provided shape using RegularGridInterpolator with specified method
    :param array:
    :param new_shape:
    :param method:
    :return:
    """
    # generate data
    entries = [np.arange(s) for s in array.shape]
    interp = RegularGridInterpolator(entries, array, method=method)

    # new grid
    new_entries = [np.linspace(0, array.shape[i] - 1, new_shape[i]) for i in range(len(array.shape))]
    new_grid = np.meshgrid(*new_entries, indexing='ij')

    return interp(tuple(new_grid)).astype(array.dtype)


def make_custom_3d_volume_item_from_data(data: np.array) -> QCustom3DVolume:
    """
    create a custom 3d volume item with the provided data set
    :param data:
    :return:
    """
    volume_item = QCustom3DVolume()

    # QApplication.processEvents()
    # print(volume_item.thread())

    volume_item.setTextureFormat(QImage.Format_Indexed8)
    volume_item.setTextureDimensions(data.shape[2], data.shape[1], data.shape[0])
    # volume_item.setTextureData(np.ascontiguousarray(data.flatten()))
    volume_item.setTextureData(data.flatten().tolist())

    # Generate and set default color table
    color_table_1 = [QColor(0, 0, 0, 0).rgba()] + [QColor(i, i, i, 20).rgba() for i in range(255)]
    volume_item.setColorTable(color_table_1)

    return volume_item


expected_fluid_flower_spatial_maps_files = {"sat_con": "spatial_maps_sat_con.npy",
                                            "t": "spatial_maps_t.npy",
                                            "x": "spatial_maps_x.npy",
                                            "y": "spatial_maps_y.npy"}

expected_fluid_flower_other_files = {
    "sparse_data": "sparse_data.csv",
    "spatial_parameters": "spatial_parameters.csv",
    "time_series": "time_series.csv"
}


def load_numpy_file_and_store_to_dict(filename: str, d: dict, key: str, rescale_to_uint8: bool = False):
    """
    Loads filename with numpy.load and stores the result in d[key].
    :param filename:
    :param d:
    :param key:
    :param rescale_to_uint8: whether to rescale the 'sat_con' data to uint8, i.e., values to range [0, 255]
    :return:
    """
    d_ = np.load(filename)

    if rescale_to_uint8:
        d_sat = d_[:, :, :, 0]
        d_con = d_[:, :, :, 1]
        d_[:, :, :, 0] = np.interp(d_sat, (d_sat.min(), d_sat.max()), (0, 255))
        d_[:, :, :, 1] = np.interp(d_con, (d_con.min(), d_con.max()), (0, 255))

        d_ = d_.astype(np.uint8)

    d[key] = d_  # [:144]


def load_csv_file_and_store_to_dict(filename: str, d: dict, key: str):
    """
    Loads filename with pandas and stores the result in d[key].
    :param filename:
    :param d:
    :param key:
    :return:
    """
    try:
        d_ = pandas.read_csv(filename)  # [:144]  # only first 144 time steps
    except RuntimeError as e:
        print(f"failed to read '{filename}'.")
        return

    d[key] = d_


def load_fluidflower_data_from_group(group_dir: str, group_name: str, result_dict: dict,
                                     rescale_to_uint8: bool = False):
    """
    Loads the data in the fluidflower group specified by group_dir and puts the result in result_dict.
    :param group_dir:
    :param result_dict:
    :param group_name:
    :param rescale_to_uint8: whether to rescale the 'sat_con' data to uint8, dtype stays int
    :return:
    """
    running_threads = []
    result_dict[group_name] = dict()

    # dense data
    for ef_k in expected_fluid_flower_spatial_maps_files:
        t = Thread(target=load_numpy_file_and_store_to_dict(
            os.path.join(group_dir, expected_fluid_flower_spatial_maps_files[ef_k]),
            result_dict[group_name], ef_k,
            rescale_to_uint8=True if rescale_to_uint8 and ef_k == "sat_con" else False))
        t.start()
        # t.join()
        running_threads.append(t)

    for ef_k in expected_fluid_flower_other_files:
        t = Thread(target=load_csv_file_and_store_to_dict(
            os.path.join(group_dir, expected_fluid_flower_other_files[ef_k]),
            result_dict[group_name], ef_k
        ))
        t.start()
        running_threads.append(t)

    # join running threads
    for t in running_threads:
        t.join()


def load_fluidflower_data(fluid_flower_data_dir: str, rescale_to_uint8: bool = False):
    """
    Loads all the fluid flower data (approx. 1.3GB) into a dict.
    :param fluid_flower_data_dir:
    :param rescale_to_uint8: whether to rescale the 'sat_con' data to uint8
    :return:
    """
    groups = [g for g in os.listdir(fluid_flower_data_dir) if not g.startswith("ensemble") and not
    g.startswith("experiment") and not g in ["mit", "imperial"]]

    running_threads = []

    data = {}

    for group in groups:
        if group == "mit":
            continue  # mit has not dense data
        # assert existence of all expected files
        assert all([os.path.exists(os.path.join(fluid_flower_data_dir, group, ef)) for ef in
                    list(expected_fluid_flower_spatial_maps_files.values())]), \
            "missing one of the files in group: '{}': {}".format(group, list(
                expected_fluid_flower_spatial_maps_files.values()))

        t = Thread(target=lambda: load_fluidflower_data_from_group(os.path.join(fluid_flower_data_dir, group), group,
                                                                   data, rescale_to_uint8=rescale_to_uint8))
        t.start()
        # t.join()
        running_threads.append(t)

    # join threads
    for t in running_threads:
        t.join()

    # load experimental data extra since it is missing lots of stuff (only got segmentation maps a.t.m)
    data.update(load_experimental_data_seg_maps(fluid_flower_data_dir))

    return data


def load_experimental_data_seg_maps(ffdd: str) -> dict:
    data = dict()

    experimental_dir = os.path.join(ffdd, "experiment")
    runs = os.listdir(experimental_dir)

    for run in runs:
        data.update({
            "experiment_" + run: {
                "t": np.load(os.path.join(experimental_dir, run, "segmentation_t.npy")),
                "sat_con": np.load(os.path.join(experimental_dir, run, "segmentation_sat_con.npy"))
            }
        })

    return data


def create_custom3d_volume_and_store_to_dict(data: np.array, out_dict: dict, key: str, as_tuple=False):
    """
    :param as_tuple: store result as tuple (data, shape) instead of dict {"data": data, "data_shape": shape}
    :param data:
    :param out_dict:
    :param key:
    :return:
    """
    custom_3d_volume, data_shape = create_custom3d_volume_and_return(data, as_tuple=as_tuple)

    if as_tuple:
        out_dict[key] = (custom_3d_volume, data_shape)
    else:
        out_dict[key] = {"data": custom_3d_volume, "data_shape": data_shape}
    print("done")


def create_custom3d_volume_and_return(data: np.array, as_tuple=False):
    data, data_shape = resize_to_power_of_two_and_scale_to_uint8(data)
    assert data.dtype == int or data.dtype == np.uint8, "dtype has to be int or uint8"
    custom_3d_volume = make_custom_3d_volume_item_from_data(data)

    print("done")
    if as_tuple:
        return custom_3d_volume, data_shape
    else:
        return {"data": custom_3d_volume, "data_shape": data_shape}


def int_or_none(s):
    if s == 'None':
        return None
    else:
        return int(s)


def create_custom3d_volumes(data: dict, default: str):
    """
    Creates a custom 3d volume for each of the groups in data. Modifies the provided data object. Return may be ignored.
    :param data:
    :return:
    """
    running_threads = []

    for group in data:
        if group == default:
            continue

        group_data = data[group]

        sat_con = group_data["sat_con"]

        group_data["volumes"] = {}

        for i, k in zip([0, 1], ["sat", "con"]):
            print("doing: ", group, k)
            t = Thread(target=lambda: create_custom3d_volume_and_store_to_dict(sat_con[:, :, :, i],
                                                                               group_data["volumes"], k))
            t.start()
            # t.join()
            time.sleep(5)
            running_threads.append(t)

    # join threads
    for t in running_threads:
        t.join()

    return data


def l1_norm(a, b):
    return np.linalg.norm(a - b, 1)


def dtw_metric(a, b):
    return fastdtw(a, b, dist=l1_norm)[0]


def get_attributes(obj):
    return list(map(lambda m: getattr(obj, m), filter(lambda m: not m.startswith("_"), dir(obj))))
