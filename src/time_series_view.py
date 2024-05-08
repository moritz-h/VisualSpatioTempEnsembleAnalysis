import random

import PySide6
from PySide6.QtGui import QMouseEvent, QPainter, QColor, QPixmap, QColorConstants
from qtpex.qt_widgets.iqchartview import IQChartView
from PySide6.QtCharts import QChart, QValueAxis
from PySide6.QtCore import Qt, Signal

from qtpex.qt_objects.configurable_line_series import ConfigurableLineSeries

import numpy as np
from src.constants import time_series_namings, time_series_namings_keys, time_series_naming_time, groups, \
    group_to_color, screenshot_locations, screenshot_randint_max


class TimeSeriesViewWidget(IQChartView):
    patch_hovered_signal = Signal(int, int, bool)  # from / to

    def __init__(self, data: dict, parent=None):
        super().__init__(QChart(), parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.data = data
        self.data_keys = [g for g in groups if g in self.data and not g.startswith("experiment")]

        self.displayed_time_series_key = "total_CO2_mass"

        # init line series
        self.line_serieses = dict()
        self.axises = dict()

        # self.setRubberBand(self.RectangleRubberBand)

        self.x_axis = QValueAxis()

        self.patch_start = 0
        self._max_t = max(len(self.data[g]["time_series"]) for g in self.data_keys)
        self.patch_end = self._max_t

        self.start_key = "_start"
        self.end_key = "_end"

        for key in time_series_namings_keys:
            self.line_serieses[key] = dict()
            self.axises[key] = dict()

            x_min, x_max = None, None
            y_min, y_max = None, None

            # need a series per group
            for group in self.data_keys:
                series = ConfigurableLineSeries()

                table = self.data[group]["time_series"]  # [:self._max_t]

                # build series
                for idx in table.index:
                    assert "t" in table.columns, group + " " + str(table.columns)
                    x = table["t"][idx] / 60  # / 60 to make it as minutes
                    y = table[key][idx]
                    if np.isnan(x) or np.isnan(y):
                        continue

                    # update min max
                    if x_min is None or x < x_min:
                        x_min = x
                    if x_max is None or x > x_max:
                        x_max = x
                    if y_min is None or y < y_min:
                        y_min = y
                    if y_max is None or y > y_max:
                        y_max = y

                    series.append(x, y)

                series.setColor(group_to_color[group])
                series.setName(group)

                series.setMarkerSize(0)

                self.line_serieses[key][group] = series
                self.chart().addSeries(series)

                pen = series.pen()
                pen.setWidthF(1.5)
                series.setPen(pen)

            # add start and end series
            for k, t in [(self.start_key, self.patch_start), (self.end_key, self.patch_end)]:
                series = ConfigurableLineSeries()
                # build series
                series.append(t*10, y_min)
                series.append(t*10, y_max)

                series.setColor(QColor(1, 1, 1, 255))
                series.setName(k)

                series.setMarkerSize(0)

                self.line_serieses[key][k] = series
                self.chart().addSeries(series)

                pen = series.pen()
                pen.setWidthF(1.5)
                series.setPen(pen)

            assert all([o is not None for o in [x_min, x_max, y_min, y_max]])

            x_axis = QValueAxis()
            x_axis.setRange(x_min, x_max)
            x_axis.setTitleText("time in minutes")  # (time_series_naming_time[1])
            x_axis.setTickCount(12 + 1)
            self.axises[key]["x_axis"] = x_axis

            y_axis = QValueAxis()
            y_axis.setRange(y_min, y_max)
            y_axis.setTitleText(time_series_namings[key])
            self.axises[key]["y_axis"] = y_axis

            for group_or_k in self.data_keys + [self.start_key, self.end_key]:
                self.chart().addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
                self.line_serieses[key][group_or_k].attachAxis(x_axis)
                self.chart().addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
                self.line_serieses[key][group_or_k].attachAxis(y_axis)

            x_axis.setVisible(False)
            y_axis.setVisible(False)

        self.enabled_groups = {
            group: True for group in self.data_keys
        }

        self.mouse_is_down = False

        # self.ps = 50
        # self.pe = 110
#
        # # add visual indicator of selected time

#
        # for key in time_series_namings_keys:
        #     for s, t in [(self.start_key, self.ps), (self.end_key, self.pe)]:
        #         series = ConfigurableLineSeries()
        #         self.line_serieses[key][s] = series
#
        #         x_axis = self.axises[key]["x_axis"]
        #         y_axis = self.axises[key]["y_axis"]
        #         series.attachAxis(x_axis)
        #         series.attachAxis(y_axis)
#
        #         # add initial points
        #         series.append(t * 10, y_axis.min())
        #         series.append(t * 10, y_axis.max())
#
        #         series.setColor(QColor(0, 0, 0, 255))
#
        #         self.chart().addSeries(series)
#
        #         pen = series.pen()
        #         pen.setWidthF(1.5)
        #         series.setPen(pen)

        self.update_lineseries()

        # connect mouse move signal to emitting patch hovered signal
        self.mouse_pressed_signal.connect(self.mouse_pressed)
        self.mouse_released_signal.connect(self.mouse_released)
        self.mouse_moved_signal.connect(self.mouse_moved)

    def save_screenshot(self, rand_int: int = None):
        pixmap = QPixmap(self.size())
        pixmap.fill(QColorConstants.Transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.render(painter)

        if rand_int is None:
            rand_int = random.randint(0, screenshot_randint_max)

        pixmap.save(f"{screenshot_locations}/{self.displayed_time_series_key}_{rand_int}.png")

        del painter
        del pixmap

    def mouse_pressed(self, event: QMouseEvent):
        self.mouse_is_down = True

    def mouse_released(self, event: QMouseEvent):
        self.mouse_is_down = False

    def mouse_moved(self, event: QMouseEvent):
        # map mouse pos to patch position
        point = self.chart().mapToValue(event.localPos())

        x = int(point.x() / 10)

        if not self.mouse_is_down:
            self.patch_start = max(0, x)
            self.patch_start = min(self._max_t - 3, self.patch_start)
            patch_end = min(self._max_t, self.patch_start + 3)
        else:
            patch_end = min(self._max_t, x)
            if patch_end < self.patch_start:
                patch_end = min(self._max_t, self.patch_start + 3)

        # print(self.patch_start, patch_end)

        self.patch_end = patch_end

        self.patch_hovered_signal.emit(self.patch_start, patch_end, self.mouse_is_down)

    # def leaveEvent(self, event: PySide6.QtCore.QEvent) -> None:
    #     self.patch_start = 0
    #     self.patch_hovered_signal.emit(self.patch_start, self._max_t)

    def update_time_frame_sliders(self, time_from, time_to):
        # force update by changing series
        # update time-frame series
        for s, t in [(self.start_key, time_from), (self.end_key, time_to)]:
            series = self.line_serieses[self.displayed_time_series_key][s]
            y1 = series.at(0).y()
            y2 = series.at(1).y()
            series.replace(0, t * 10, y1)
            series.replace(1, t * 10, y2)

    def update_enabled_group(self, group: str, enabled: bool) -> bool:
        if group not in self.enabled_groups:
            return False

        if self.enabled_groups[group] == enabled:
            return True

        self.enabled_groups[group] = enabled

        if not any(self.enabled_groups.values()):
            # at least one has to be True
            self.enabled_groups[group] = True
            return False

        self.update_lineseries()
        # self.update() # TODO: why does this not work for removing series?
        # self.repaint()
        return True

    def get_enabled_data_keys(self):
        return [k for k in self.data_keys if self.enabled_groups[k]]

    def get_not_enabled_data_keys(self):
        return [k for k in self.data_keys if not self.enabled_groups[k]]

    def set_time_series_data(self, key: str):
        self.displayed_time_series_key = key
        self.update_lineseries()

    def update_lineseries(self):
        # print(f"update for key '{self.displayed_time_series_key}'")
        # make all series invisible
        for key in time_series_namings_keys:
            # if key == self.displayed_time_series_key:
            #     continue
            for group in self.line_serieses[key]:
                # if group in [self.start_key, self.end_key]:
                #     continue
                self.line_serieses[key][group].setVisible(False)
            for axis in ["x_axis", "y_axis"]:
                self.axises[key][axis].setVisible(False)

        # add all enabled series for this key + time frame
        for group_or_k in self.get_enabled_data_keys() + [self.start_key, self.end_key]:
            series = self.line_serieses[self.displayed_time_series_key][group_or_k]
            series.setVisible(True)
            for axis in ["x_axis", "y_axis"]:
                self.axises[self.displayed_time_series_key][axis].setVisible(True)

        # update ranges of the enabled groups
        x_min, x_max = None, None
        y_min, y_max = None, None

        # need a series per group
        for group in self.get_enabled_data_keys():
            table = self.data[group]["time_series"]  # [:self._max_t]

            # build series
            for idx in table.index:
                assert "t" in table.columns, group + " " + str(table.columns)
                x = table["t"][idx] / 60  # / 60 to make it as minutes
                y = table[self.displayed_time_series_key][idx]
                if np.isnan(x) or np.isnan(y):
                    continue

                # update min max
                if x_min is None or x < x_min:
                    x_min = x
                if x_max is None or x > x_max:
                    x_max = x
                if y_min is None or y < y_min:
                    y_min = y
                if y_max is None or y > y_max:
                    y_max = y

        assert all([o is not None for o in [x_min, x_max, y_min, y_max]])

        self.axises[self.displayed_time_series_key]["x_axis"].setRange(x_min, x_max)
        self.axises[self.displayed_time_series_key]["y_axis"].setRange(y_min, y_max)

