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
    ffdir = "./data"
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


def main():
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
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
