import argparse
import sys

import pandas
from PySide6.QtWidgets import QApplication
from src.mainwindow import MainWindow
import numpy as np
import os

from PySide6.QtGui import QSurfaceFormat
# from src.util import load_fluidflower_data


def load_data():
    ffdir = "D:/Data/Fluidflower_rescaled"
    groups = ['stuttgart', 'stanford', 'melbourne', 'lanl', 'heriot-watt',
              'delft-DARSim', 'delft-DARTS', 'csiro', 'austin', 'experiment_run1', 'experiment_run2', 'experiment_run4',
              'experiment_run5']
    experimental = "experiment"
    data = dict()
    for group in groups:
        gd = {}

        for suffix in ["t", "x", "y", "sat_con"]:
            gd.update({
                suffix: np.load(os.path.join(ffdir, group, ("spatial_maps_" if not group.startswith(experimental) else "segmentation_") + suffix + ".npy"))
            })

        data[group] = gd

    # load time series
    for group in groups:
        if group.startswith(experimental):
            continue
        data[group]["time_series"] = pandas.read_csv(os.path.join(ffdir, group, "time_series.csv"))

    return data

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


def main():
    options = argparse.ArgumentParser()
    options.add_argument("-f", "--file", type=str, required=False)
    args = options.parse_args()

    app = QApplication(sys.argv)

    # set default format
    format = QSurfaceFormat()
    format.setDepthBufferSize(24)
    format.setStencilBufferSize(8)
    format.setVersion(4, 6)
    format.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(format)

    # load data into RAM
    data = load_data()

    # init main window
    window = MainWindow(data=data)
    # window = MainWindow(left_d, right_d)
    window.show()

    sys.exit(app.exec())


# def multithreaded_test:
    # left_d = None
    # right_d = None

    # using manager and
    # with Manager() as manager:
    #     volume_item_cache_left = manager.dict()
    #     volume_item_cache_right = manager.dict()
    #
    #     processes = []
    #     for group in data:
    #         volume_item_cache_left[group] = manager.dict()
    #         volume_item_cache_right[group] = manager.dict()
    #
    #         for i, k in enumerate(["Saturation", "Concentration"]):
    #             p = Process(target=create_custom3d_volume_and_store_to_dict, args=(data[group]["sat_con"][:, :, :, i],
    #                                                                                volume_item_cache_left[group],
    #                                                                                k,
    #                                                                                True))
    #             p.start()
    #             processes.append(p)
    #
    #     for p in processes:
    #         p.join()
    #
    #     left_d = volume_item_cache_left.copy()
    #     right_d = volume_item_cache_right.copy()

    #  # begin pathos test
    #  q_threadpool = QThreadPool()
    #  q_threadpool.setMaxThreadCount(100)
    #  print("max threads: ", q_threadpool.maxThreadCount())
#
    #  volume_item_cache_left = dict()
    #  volume_item_cache_right = dict()
    #  workers = []
#
    #  for group in data:
    #      volume_item_cache_left[group] = dict()
    #      volume_item_cache_right[group] = dict()
    #      for i, k in enumerate(["Saturation", "Concentration"]):
    #          worker = Worker(create_custom3d_volume_and_return,
    #                          data[group]["sat_con"][:, :, :, i], True)
    #          workers.append(worker)
    #          # q_threadpool.start(worker)
#
    #          worker = Worker(create_custom3d_volume_and_return,
    #                          data[group]["sat_con"][:, :, :, i], True)
    #          workers.append(worker)
    #          # q_threadpool.start(worker)
#
    #  for w in workers:
    #      q_threadpool.start(w)
#
    #  # q_threadpool.waitForDone()


if __name__ == "__main__":
    main()
