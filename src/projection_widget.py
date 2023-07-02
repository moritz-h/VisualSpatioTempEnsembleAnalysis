import copy
import math
import os.path
import random
import time

import PySide6
import fastdtw
import numpy
import scipy.stats
import sklearn
import sklearn.manifold
from PySide6.QtGui import QPainter, QColor, QCursor, QFont, QBrush, QPixmap
from PySide6.QtWidgets import QSizePolicy, QGraphicsTextItem, QStyleOptionGraphicsItem
from qtpex.qt_utility.series import find_point_idx
from qtpex.qt_widgets.iqchartview import IQChartView
from PySide6.QtCharts import QChart, QValueAxis, QScatterSeries, QLineSeries
from PySide6.QtCore import Qt, Slot, Signal, QRect, QMargins, QSize

from qtpex.qt_objects.configurable_scatter_series import ConfigurableScatterSeries
from pybrutil.scipy_util.ndim_pairwise_distance import pairwise_distance, flatten_extra_dimensions
from pybrutil.general.strings import remove_except_between_chars
from pybrutil.np_util.data import interp_variables

import keras
import json
import pickle
from s4_util.Siamese.layers import CUSTOM_KERAS_LAYERS
from s4_util.Siamese.data_loading import downsample_volume
import numpy as np
import umap

from src.constants import time_series_namings, time_series_namings_keys, group_to_color, groups, screenshot_locations, \
    screenshot_randint_max
from src.util import int_or_none
from hashlib import sha1

import warnings

# default path
# run_dir_con = "C:/Users/bauerrn/Projekte/S4/out/fluidflower_ensemble_concentration.py/221011-183802-crimson-lab_fluidflower_ensemble_concentration"
# run_dir_sat = "C:/Users/bauerrn/Projekte/S4/out/fluidflower_ensemble_saturation.py/221011-183540-still-frog_fluidflower_ensemble_saturation"
# run_dir_multivar = "C:/Users/bauerrn/Projekte/S4/out/fluidflower_ensemble_multivar.py/221011-184021-bitter-rain_fluidflower_ensemble_multivar"

# rescaled full spatial
run_dir_con = "C:\\Users\\bauerrn\\Projekte\\S4\\out\\rescaled_ff_con\\221125-103018-super-feather_fluidflower_ensemble_concentration"
run_dir_sat = "C:\\Users\\bauerrn\\Projekte\\S4\\out\\rescaled_fluidflower_ensemble_saturation.py\\221123-195603-wandering-recipe_fluidflower_ensemble_saturation"
run_dir_multivar = "C:\\Users\\bauerrn\\Projekte\\S4\\out\\rescaled_fluidflower_ensemble_multivar.py\\221123-200047-sweet-hill_fluidflower_ensemble_multivar"

# rescaled ts
run_dir_con_ts = "C:\\Users\\bauerrn\\Projekte\\S4\\out\\rs_ts_fluidflower_ensemble_ts_con\\221124-110724-throbbing-fog_fluidflower_ensemble_ts_con"
run_dir_sat_ts = "C:\\Users\\bauerrn\\Projekte\\S4\\out\\rs_ts_fluidflower_ensemble_ts_saturation\\221123-211417-floral-feather_fluidflower_ensemble_ts_saturation"
run_dir_multivar_ts = "C:\\Users\\bauerrn\\Projekte\\S4\\out\\rs_ts_fluidflower_ensemble_ts_multivar\\221123-222817-lucky-water_fluidflower_ensemble_ts_multivar"

# run_dir_patch_size_55k = "C:/Users/bauerrn/Projekte/S4/out/fluidflower_ensemble3_time_space/221111-174954-bold-snowflake_fluidflower_ensemble_time_space"  #  "C:/Users/bauerrn/Projekte/S4/out/fluidflower_ensemble2_time_space/221027-123234-shy-dawn_fluidflower_ensemble_time_space"


# fake encoder
class FakeEncoder:
    def predict(self, x):
        return np.random.random((len(x), 255))

    @property
    def input_shape(self):
        return tuple([None, 3, 124, 286, 2])


useFake = False  # encoder = FakeEncoder()
if useFake:
    warnings.warn("using fake encoder, i.e., all results using S4 are random.")

useCache = True
initWithClearCache = False


def make_hash():
    def hash_fn_(arg) -> str:
        hash_fn = sha1()
        hash_fn.update(str(arg).encode("utf-8"))
        return hash_fn.hexdigest()

    return hash_fn_


class PickleCache:
    def __init__(self, cache_dir: str, clear_cache_dir: bool = False):
        assert os.path.isdir(cache_dir)
        self.cache_dir = cache_dir

        self.prefix = "hash_"
        self.suffix = ".p"

        self.key_fmt = self.prefix + "{}" + self.suffix

        self.hash = make_hash()

        if useFake:
            self.fake_cache = dict()

        if clear_cache_dir:
            files = os.listdir(self.cache_dir)
            for f in files:
                if f.startswith(self.prefix) and f.endswith(self.suffix):
                    os.remove(os.path.join(self.cache_dir, f))

    def _key_to_filename(self, item) -> str:
        return self.key_fmt.format(self.hash(item))

    def _filename_to_key(self, filename) -> int:
        return filename[len(self.prefix):-len(self.suffix)]

    def __getitem__(self, item):
        if not useCache:
            return KeyError("not using cache")

        key = self._key_to_filename(item)

        if useFake:
            if key in self.fake_cache:
                return self.fake_cache[key]

        # print("get")
        path = os.path.join(self.cache_dir, key)
        if not os.path.exists(path):
            return KeyError(f"'{path}' does not exist")

        with open(path, "rb") as f:
            o = pickle.load(f)
            return o

    def __setitem__(self, key, value):
        if not useCache:
            print("don't set item. not using cache")
            return

        item = key
        key = self._key_to_filename(key)

        if useFake:
            self.fake_cache[key] = value
            return

        path = os.path.join(self.cache_dir, key)
        # print("set")
        # print(os.path.abspath(path))

        with open(path, "wb") as f:
            pickle.dump(value, f)

        with open(path + ".json", "wt") as f:
            json.dump({
                "item": item,
                "key": key
            }, f)

    def __delitem__(self, key):
        if not useCache:
            print("don't delete item. not using cache")
            return

        if self.has_key(key):
            key = self._key_to_filename(key)
            os.remove(os.path.join(self.cache_dir, key))

    def has_key(self, key):
        if not useCache:
            return False

        if useFake:
            if self._key_to_filename(key) in self.fake_cache:
                return True

        return self.hash(key) in iter([self._filename_to_key(f) for f in os.listdir(self.cache_dir)])


def load_encoder_normalizer_and_config(run_dir: str):
    model_path = os.path.join(run_dir, "model.hdf")
    normalizer_path = os.path.join(run_dir, "normalizer.pcl")
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "rt") as f:
        config = json.load(f)

    # load model
    if not useFake:
        model = keras.models.load_model(model_path, custom_objects=CUSTOM_KERAS_LAYERS)
        encoder = model.get_layer("encoder")
    else:
        encoder = FakeEncoder()

    # load normalizer
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    return encoder, normalizer, config


enc_norm_conf_con = load_encoder_normalizer_and_config(run_dir_con)
enc_norm_conf_sat = load_encoder_normalizer_and_config(run_dir_sat)
enc_norm_conf_multivar = load_encoder_normalizer_and_config(run_dir_multivar)


enc_norm_conf_con_ts = load_encoder_normalizer_and_config(run_dir_con_ts)
enc_norm_conf_sat_ts = load_encoder_normalizer_and_config(run_dir_sat_ts)
enc_norm_conf_multivar_ts = load_encoder_normalizer_and_config(run_dir_multivar_ts)

# enc_norm_conf_patch_size_55k = load_encoder_normalizer_and_config(run_dir_patch_size_55k)


class ProjectionWidget(IQChartView):
    group_clicked_signal = Signal(str, int, int,
                                  str)  # group, patch time start, patch time end, "left" or "right" mouse
    group_patch_hovered_signal = Signal(str, int, int)  # group, patch time start, patch time end

    class ProjAlgorithm:
        # PCA = "PCA"
        MDS = "MDS"
        UMAP = "UMAP"
        TSNE = "TSNE"
        # SPLOM_View = "As Scatterplot Matrix"

    class DataReductionMode:
        GROUP = "Group"  # Per group
        PATCH = "Patch"  # Per patch

    class ProjectionDim:
        OneD = "1D (+T)"  # +T if per patch is used
        TwoD = "2D"

    class DissimilarityMeasure:
        S4 = "S4"
        S4_Spatially_subdivided = "S4 small patches"
        Euclidean = "Euclidean"
        Manhattan = "Manhattan"
        Wasserstein = "Wasserstein"

    class PartitionStrategy:
        Overlapping = "Overlapping"  # steps of 1
        NonOverlapping = "Adjacent"  # steps of patch size

    class TemporalPatchSize:
        Single = "Single"  # single step size. if used with S4, the data is repeated to match the original patch size.
        Full = "Full"  # takes S4's original patch size

    class DataMode:
        Multivariate = "Multivariate (Sat + Con)"
        Saturation = "Saturation"
        Concentration = "Concentration"
        TimeSeriesData = "Time-series Data"  # using this will hide/ignore data_red_mode
        TimeSeriesData_CombinedFeatures = "Time-series Data with Combined Features"

    def __init__(self, data, min_max, *,
                 data_red_mode=DataReductionMode.GROUP,
                 proj_alg=ProjAlgorithm.MDS,
                 proj_dim=ProjectionDim.TwoD,
                 dissim_measure=DissimilarityMeasure.S4,
                 part_strat=PartitionStrategy.NonOverlapping,
                 temp_p_size=TemporalPatchSize.Full,
                 data_mode=DataMode.Multivariate,
                 compare_all_via_seg_map=False,
                 show_spline_series=False,
                 parent=None):
        super().__init__(QChart(), parent=parent)
        self.min_max = min_max

        self.compare_all_via_seg_map = compare_all_via_seg_map
        self.show_spline_series = show_spline_series

        self._is_full_run = False
        self._compute_dummy_input = False  # if True, only computes dummy inputs

        self.show_labels = True

        self._use_fast_dtw = True

        self._minmin = 0.001

        # seg map thresholds
        self.seg_map_th = [self._minmin, self._minmin]  # None means take minimum

        # patch range
        self.patch_range = [None, None]  # None means full range

        # configurations
        self.proj_alg = proj_alg
        self.data_red_mode = data_red_mode
        self.proj_dim = proj_dim
        self.dissim_measure = dissim_measure
        self.part_strat = part_strat
        self.temp_p_size = temp_p_size
        self.data_mode = data_mode

        self._original_data_sat_con = {group: data[group]["sat_con"] for group in data if group in groups}
        self.data = self.prepare_data(self._original_data_sat_con, *self.get_enc_norm_conf()[1:])
        self.data_time_series = {group: data[group]["time_series"] for group in data if group in groups and not group.startswith("experiment")}

        # caches to reuse data from previous configurations
        # set caches to functions
        cache_dir = "./cache"
        self._cached_update_dissimilarity_matrix_cache = PickleCache(cache_dir, initWithClearCache)
        self._cached_update_dissim_input_cache = PickleCache(cache_dir, initWithClearCache)
        self._cached_update_input_cache = PickleCache(cache_dir, initWithClearCache)

        self.data_keys = [g for g in groups if g in self._original_data_sat_con]  # needed to specify order in dissim mat
        print(self.data_keys)

        self.series_marker_size = 8
        self.series_exp_marker_size = self.series_marker_size + 4

        self.halo_serieses = dict()

        for group in self.data_keys:
            halo_series = QScatterSeries()
            self.halo_serieses[group] = halo_series
            halo_series.setBorderColor(Qt.transparent)
            halo_series.setSelectedColor(QColor(0, 0, 0, 255))
            halo_series.setColor(Qt.transparent)
            halo_series.setMarkerSize(self.series_marker_size + 4)
            if group.startswith("experiment"):
                halo_series.setMarkerShape(halo_series.MarkerShapeStar)
                halo_series.setMarkerSize(self.series_exp_marker_size + 4)

        self.serieses = dict()

        def make_point_clicked_on_group_closure(group: str):
            return lambda: self.point_clicked(group)

        def make_point_hovered_on_group_closure(group: str):
            return lambda point, state: self.point_hovered(point, state, group=group)

        for group in self.data_keys:
            series = ConfigurableScatterSeries()
            self.serieses[group] = series

            series.setBorderColor(Qt.transparent)
            series.clicked.connect(make_point_clicked_on_group_closure(group))
            series.hovered.connect(make_point_hovered_on_group_closure(group))
            series.setMarkerSize(self.series_marker_size)
            series.setSelectedColor(group_to_color[group])
            series.setColor(group_to_color[group])
            series.setName(group)

            if group.startswith("experiment"):
                series.setMarkerShape(series.MarkerShapeStar)

        # Spline Series
        self.spline_serieses = {group: QLineSeries() for group in self.data_keys}
        # set colors and size and
        for group in self.data_keys:
            # set color
            ss = self.spline_serieses[group]
            ss.setColor(group_to_color[group])

            # set name
            ss.setName(group)

            # set size
            pen = ss.pen()
            pen.setWidthF(1.5)
            ss.setPen(pen)

            self.chart().addSeries(ss)
            self.chart().legend().markers(ss)[0].setVisible(False)

        self.eps = 0.1

        chart = self.chart()

        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.legend().setVisible(False)

        for halo_series in self.halo_serieses.values():
            chart.addSeries(halo_series)

        for series in self.serieses.values():
            chart.addSeries(series)

        for halo_series in self.halo_serieses.values():
            chart.legend().markers(halo_series)[0].setVisible(False)

        for series in self.serieses.values():
            chart.legend().markers(series)[0].setVisible(False)

        self.x_axis = QValueAxis()
        self.x_axis.setTickCount(2)
        self.x_axis.setRange(-self.eps, 1 + self.eps)
        chart.addAxis(self.x_axis, Qt.AlignBottom)
        for series in self.serieses.values():
            series.attachAxis(self.x_axis)
        for halo_series in self.halo_serieses.values():
            halo_series.attachAxis(self.x_axis)

        self.y_axis = QValueAxis()
        self.y_axis.setTickCount(2)
        self.y_axis.setRange(-self.eps, 1 + self.eps)
        chart.addAxis(self.y_axis, Qt.AlignLeft)
        for series in self.serieses.values():
            series.attachAxis(self.y_axis)
        for halo_series in self.halo_serieses.values():
            halo_series.attachAxis(self.y_axis)

        self.x_axis.setVisible(False)
        self.y_axis.setVisible(False)

        # spline series attach axis
        for group in self.data_keys:
            ss = self.spline_serieses[group]

            ss.attachAxis(self.x_axis)
            ss.attachAxis(self.y_axis)

        self.mouse_pressed_signal.connect(self.mouse_pressed)
        self.mouse_released_signal.connect(self.mouse_released)

        self.mouse_button = None

        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setSizePolicy(size)

        self.setRenderHint(QPainter.Antialiasing)

        # ---
        # experimental groups are not enabled by default
        self.enabled_groups = {
            group: (True if not group.startswith("experiment") else False) for group in self.data_keys
        }

        self.enabled_times_series = {
            key: True for key in time_series_namings
        }
        # ---

        self.encodings = {}
        self.group_beginnings = []
        self.enabled_group_beginnings = []
        self.dissim_input = self.init_dissim_input()
        self.dissimilarity_matrix = self.init_dissim_matrix()
        self.projection = self.init_projection()
        self.text_labels = {group: QGraphicsTextItem(group, self.chart()) for group in self.data_keys}

        # set font
        text_color = QColor(21, 21, 21)
        for tl in self.text_labels.values():
            font = QFont("Inter", 20, QFont.Bold)
            tl.setFont(font)
            tl.setDefaultTextColor(text_color)

        self.update_attributes()

        self.chart().plotAreaChanged.connect(self.update_labels())

        # chart background
        chart.setBackgroundBrush(QBrush(QColor(238, 238, 238)))

    def set_use_fastdtw_reduction(self, useFDTW: bool):
        self._use_fast_dtw = useFDTW
        self.update_attributes(self.update_dissimilarity_matrix)

    def set_show_labels(self, show: bool):
        self.show_labels = show
        self.update_attributes(self.update_labels)

    def save_projection_as_screenshot(self, randint=None):
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        self.render(painter)

        if randint is None:
            randint = random.randint(0, screenshot_randint_max)

        if "Time" in self.data_mode:
            pixmap.save(f"{screenshot_locations}/{self.proj_alg}_{self.proj_dim}_{self.data_mode}_{self.show_spline_series}_{self.compare_all_via_seg_map}_{randint}.png")
        else:
            pixmap.save(f"{screenshot_locations}/{self.proj_alg}_{self.data_red_mode}_{self.proj_dim}_{self.dissim_measure}_{self.data_mode}_{self.show_spline_series}_{self.compare_all_via_seg_map}_{randint}.png")

        del painter
        del pixmap

    def set_segmentation_threshold(self, th: list):
        assert len(th) == 2, "th should be of length 2"
        assert th[0] is not None and th[1] is not None
        self.seg_map_th = th
        self.update_attributes()

    def set_patch_range(self, from_patch, to_patch):
        self.patch_range = [from_patch, to_patch]
        self.update_attributes(self.update_projection)

    def set_compare_all_via_seg_map(self, compare_all_via_seg_map: bool):
        self.compare_all_via_seg_map = compare_all_via_seg_map
        self.update_attributes()

    def update_enabled_groups(self, groups: list, enabled: bool, force_no_update_attributes=False):
        for group in groups[:-1]:
            self.update_enabled_group(group, enabled, False)

        # only update attributes at the last call
        self.update_enabled_group(groups[-1], enabled, not force_no_update_attributes)

    def update_enabled_group(self, group: str, enabled: bool, update_attribute=True) -> bool:
        if self.enabled_groups[group] == enabled:
            return True

        self.enabled_groups[group] = enabled

        if not any(self.enabled_groups.values()):
            # at least one has to be True
            self.enabled_groups[group] = True
            return False

        self.update_enabled_group_beginnnings()

        if update_attribute:
            self.update_attributes(self.update_projection)

        return True

    def update_enabled_group_beginnnings(self):
        # compute lengths of groups in projection
        self.enabled_group_beginnings = [
            (sum([len(self.encodings[k]) for k in self.get_enabled_data_keys()[:i]])) for i, group in
            enumerate(self.get_enabled_data_keys())]

    def get_enabled_data_keys(self):
        return [k for k in self.data_keys if self.enabled_groups[k]]

    def get_not_enabled_data_keys(self):
        return [k for k in self.data_keys if not self.enabled_groups[k]]

    def get_group_idx_of_dissimilarity_matrix_row(self, row: int):
        group_idx = 0
        for j in range(len(self.data_keys)):
            if row >= self.group_beginnings[j]:
                group_idx = j
        return group_idx

    def get_group_idx_of_projection_row(self, row: int):
        group_idx = 0
        group_beg_idx = 0
        idx = 0
        for j in range(len(self.data_keys)):
            if self.enabled_groups[self.data_keys[j]]:
                if row >= self.enabled_group_beginnings[idx]:
                    group_idx = j
                    group_beg_idx = idx
                idx += 1
        return group_idx, group_beg_idx

    def update_enabled_times_series(self, key: str, enabled: bool) -> bool:
        if self.enabled_times_series[key] == enabled:
            return True

        self.enabled_times_series[key] = enabled

        if not any(self.enabled_times_series.values()):
            # at least one has to be True
            self.enabled_times_series[key] = True
            return False

        if self.data_mode == self.DataMode.TimeSeriesData:
            self.update_attributes(self.update_projection)
        elif self.data_mode == self.DataMode.TimeSeriesData_CombinedFeatures:
            self.update_attributes(self.update_dissimilarity_matrix)

        return True

    def wheelEvent(self, event: PySide6.QtGui.QWheelEvent) -> None:
        factor = 1.5
        if event.angleDelta().y() > 0:
            factor = factor
        else:
            factor = 1 / factor

        r = QRect(self.chart().plotArea().left(), self.chart().plotArea().top(),
                  self.chart().plotArea().width() / factor,
                  self.chart().plotArea().height() / factor)
        mouse_pos = self.mapFromGlobal(QCursor.pos())
        r.moveCenter(mouse_pos)
        self.chart().zoomIn(r)
        delta = self.chart().plotArea().center() - mouse_pos
        self.chart().scroll(delta.x(), -delta.y())

        self.update_labels()

    def get_enc_norm_conf(self):
        if self.dissim_measure == self.DissimilarityMeasure.S4_Spatially_subdivided:
            if self.data_mode == self.DataMode.Multivariate:
                return enc_norm_conf_multivar_ts
            elif self.data_mode == self.DataMode.Concentration:
                return enc_norm_conf_con_ts
            elif self.data_mode == self.DataMode.Saturation:
                return enc_norm_conf_sat_ts
            else:
                raise RuntimeError(f"unknown data mode: {self.data_mode}")
        else:
            if self.data_mode == self.DataMode.Multivariate:
                return enc_norm_conf_multivar
            elif self.data_mode == self.DataMode.Concentration:
                return enc_norm_conf_con
            elif self.data_mode == self.DataMode.Saturation:
                return enc_norm_conf_sat
            else:
                # NOTE: this is a hotfix for time series data
                return enc_norm_conf_multivar
                # raise RuntimeError(f"unknown data mode: {self.data_mode}")

    def set_selected_data_mode(self, data_mode):
        if self.data_mode == self.DataMode.TimeSeriesData or data_mode == self.DataMode.TimeSeriesData_CombinedFeatures:
            update_all = True
        else:
            update_all = False

        self.data_mode = data_mode

        if update_all:
            self.update_attributes()
        else:
            self.update_attributes()

    def set_proj_alg(self, proj_alg):
        self.proj_alg = proj_alg
        self.update_attributes(self.update_projection)

    def set_proj_temp_size(self, proj_temp_size):
        self.temp_p_size = proj_temp_size
        self.update_attributes()

    def set_proj_dim(self, proj_dim):
        self.proj_dim = proj_dim
        self.update_attributes(self.update_projection)

    def set_red_mode(self, data_red_mode):
        self.data_red_mode = data_red_mode
        self.update_attributes(self.update_dissim_input)

    def set_proj_dis_metric(self, proj_dis_metric):
        self.dissim_measure = proj_dis_metric
        self.update_attributes()#self.update_dissimilarity_matrix)

    def set_proj_part_strat(self, proj_part_strat):
        self.part_strat = proj_part_strat
        self.update_attributes()

    def _to_binary(self, x: np.array):
        """
        Returns a copy of x where for each variable dimension (T, Y, X, variable_dimension)
            all values bigger than the min in x are set to 1 and all values equal the min
            of x are set to 0.
        :param x:
        :return:
        """
        assert x.shape[-1] <= 2, "check this"

        x = x.copy()

        # use provided threshold if not None else minimum
        if self.data_mode == self.DataMode.Multivariate:
            for v_i in range(x.shape[-1]):
                where = [x[..., v_i] > self.seg_map_th[v_i]]
                not_where = [x[..., v_i] <= self.seg_map_th[v_i]]
                x[..., v_i][tuple(where)] = 1
                x[..., v_i][tuple(not_where)] = 0
        else:
            if x.shape[-1] == 1:
                access_idx = 0
            elif self.data_mode == self.DataMode.Saturation:
                access_idx = 0
            else:
                access_idx = 1

            if self.data_mode == self.DataMode.Saturation:
                th = self.seg_map_th[0]
            else:
                th = self.seg_map_th[1]

            where = [x[..., access_idx] > th]
            not_where = [x[..., access_idx] <= th]
            x[..., access_idx][tuple(where)] = 1
            x[..., access_idx][tuple(not_where)] = 0

        return x

    def _cached_update_dissim_input(self):
        cache = self._cached_update_dissim_input_cache
        key = (self._cached_update_dissim_input.__name__, ) + self.get_update_dissim_input_cache_keys()

        do_not_use_cache = True  # don't use cached dissim inpt

        if cache.has_key(key) and not do_not_use_cache:
            v = cache[key]
        else:
            if self.data_red_mode == self.DataReductionMode.GROUP:
                # each group is one row in the matrix, accessed by index
                X = []

                for group in self.data_keys:
                    X.append(self.encodings[group])

            elif self.data_red_mode == self.DataReductionMode.PATCH:
                # each patch is one row in the matrix
                # store ranges which patch belongs to which group
                X = []
                for i, group in enumerate(self.data_keys):
                    for encoding in self.encodings[group]:
                        X.append(encoding)
            else:
                raise RuntimeError("unknown data red mode: {}".format(self.data_red_mode))

            if self.data_mode == self.DataMode.Multivariate:
                # do nothing
                pass
            elif self.data_mode == self.DataMode.Saturation:
                # select only saturation values
                X = [enc[..., 0:1] for enc in X]
            elif self.data_mode == self.DataMode.Concentration:
                X = [enc[..., 1:2] for enc in X]
            else:
                raise RuntimeError("unknown data mode: {}".format(self.data_mode))

            # doing this now already in update inputs
            # if self.compare_all_via_seg_map:
            #     # change precompute to make segmentation map
            #     X = [self._to_binary(x) for x in X]

            v = X

            if not do_not_use_cache:
                cache[key] = v

        self.dissim_input = v

    @staticmethod
    def compute_dissim_input_for_time_series(encodings, data_keys):
        """
        Returns dissim input, one list for each ensemble member. each list contains again measurements
        :param encodings:
        :param data_keys:
        :return:
        """
        dissim_input = [[encodings[group][key].values for key in time_series_namings_keys] for group in data_keys]
        assert all([not np.isnan(di).any() for di in dissim_input]), "time_series contains nan values"

        return dissim_input

    def update_dissim_input(self):
        t = time.time()
        if self.data_mode == self.DataMode.TimeSeriesData or self.data_mode == self.DataMode.TimeSeriesData_CombinedFeatures:
            self.dissim_input = self.compute_dissim_input_for_time_series(self.encodings, self.data_keys)
        else:
            self._cached_update_dissim_input()

        print("update {} took: {} seconds".format("dissim input", time.time() - t))

    def init_dissim_input(self):
        return np.eye(0)

    def get_proj_dim_num(self):
        if self.proj_dim == self.ProjectionDim.OneD:
            return 1
        if self.proj_dim == self.ProjectionDim.TwoD:
            return 2
        raise RuntimeError("unknown projection dim: {}".format(self.proj_dim))

    def get_proj_algorithm(self, get_all=False):
        dim = self.get_proj_dim_num()

        ret = []

        # if self.proj_alg == self.ProjAlgorithm.SPLOM_View:
        #     get_all = True

        if self.proj_alg == self.ProjAlgorithm.MDS or get_all:
            ret.append((self.ProjAlgorithm.MDS, lambda x: sklearn.manifold.MDS(n_components=dim, dissimilarity="precomputed",
                                                  n_jobs=-1).fit_transform(x)))
        if self.proj_alg == self.ProjAlgorithm.UMAP or get_all:
            """def make_metric(dissim_mat):
                assert dissim_mat is not None, "dissim_mat cannot be None"

                @njit()
                def dissim_mat_metric(i, j):
                    return dissim_mat[int(i[0]), int(j[0])]

                return dissim_mat_metric

            def make_umap(x):
                indices = np.arange(len(x), dtype=int)
                indices = np.column_stack([indices, indices])

                return umap.UMAP(n_components=dim, metric=make_metric(x)).fit_transform(indices)"""
            n_neighbors = max(2, int(len(
                self.get_enabled_data_keys()) / 2)) if self.data_red_mode == self.DataReductionMode.GROUP else 15
            ret.append((self.ProjAlgorithm.UMAP, lambda x: umap.UMAP(n_components=dim, metric="precomputed", n_neighbors=n_neighbors).fit_transform(
                x)))  # x is here the dissimilarity matrix
        if self.proj_alg == self.ProjAlgorithm.TSNE or get_all:
            n_neighbors = max(2, int(len(
                self.get_enabled_data_keys()) / 2)) if self.data_red_mode == self.DataReductionMode.GROUP else 15
            ret.append((self.ProjAlgorithm.TSNE, lambda x: sklearn.manifold.TSNE(n_components=dim, perplexity=n_neighbors,
                                                   metric="precomputed").fit_transform(x)))
        if len(ret) == 0:
            raise RuntimeError("unknown projection algorithm: {}".format(self.proj_alg))

        return ret

    def init_dissim_matrix(self):
        if self.data_red_mode == self.DataReductionMode.GROUP:
            amount_patches = sum([len(self.data[d]) for d in self.data])
        elif self.data_red_mode == self.DataReductionMode.PATCH:
            amount_patches = len(self.data)
        else:
            raise RuntimeError("unknown projection mode")

        return np.random.random((amount_patches, amount_patches))

    def init_projection(self):
        # algs = self.get_proj_algorithm(True)

        if self.data_red_mode == self.DataReductionMode.GROUP:
            return np.random.random((len(self.data), 2))  # {alg_name: np.random.random((len(self.data), 2)) for alg_name in algs}
        elif self.data_red_mode == self.DataReductionMode.PATCH:
            return np.random.random((sum([len(self.data[d]) for d in self.data]), 2))  # {alg_name: np.random.random((sum([len(self.data[d]) for d in self.data]), 2)) for alg_name in algs}
        else:
            raise RuntimeError("unknown projection mode")

    def update_attributes(self, starting_from=None):
        """
        Calls all functions in correct order to update the attributes, starting from provided function.
        If None is given, updates all.
        :param starting_from:
        :return:
        """
        to_update = [self.update_inputs,
                     self.update_dissim_input,
                     self.update_dissimilarity_matrix,
                     self.update_projection,
                     self.update_series,
                     self.update_labels]

        if starting_from is not None:
            assert starting_from in to_update, "starting_from has to be in {}".format(to_update)
        else:
            starting_from = to_update[0]

        # ignore update inputs, dissim inputs if we already have the content of dissim matrix cached for non time series stuff
        can_ignore_inputs = False
        if can_ignore_inputs and "Time" not in self.data_mode:
            cache = self._cached_update_dissimilarity_matrix_cache
            key = (self._cached_update_dissimilarity_matrix.__name__,) + self.get_update_dissim_mat_cache_keys()
            if cache.has_key(key):
                print("cache has key, ignore computing inputs")
                # starting_from = self.update_dissimilarity_matrix
                # self._compute_dummy_input = True

        first = 0
        for i in range(len(to_update)):
            if to_update[i] == starting_from:
                first = i
                break

        for i in range(first, len(to_update)):
            to_update[i]()

    def resizeEvent(self, event: PySide6.QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        #s = min(event.size().width(), event.size().height())  # TODO: find better way to keep aspect ratio fixed
        #self.chart().setMaximumSize(QSize(s, s))  # DEBUG: for taking screenshots
        self.update_labels()

    def update_labels(self):
        t = time.time()

        for group in self.data_keys:
            text_item = self.text_labels[group]
            text_item.setVisible(self.enabled_groups[group])
            if not self.enabled_groups[group]:
                continue

            series = self.serieses[group]
            p = series.at(series.count() - 1)
            pos = self.chart().mapToPosition(p, series)
            text_item.setPos(pos)

        # draw text labels in foreground
        for tl in self.text_labels.values():
            tl.adjustSize()
            pos = tl.pos()
            width = tl.textWidth()
            tl.setZValue(100)

            if self.proj_dim == self.ProjectionDim.OneD and self.data_red_mode == self.DataReductionMode.GROUP:
                tl.setPos(pos.x() + 7, pos.y() - 2 * self.series_marker_size)  # set to right and middle
            else:
                tl.setPos(pos.x() - 0.5 * width, pos.y() + 7)  # center and push down

        # hide all spline series
        for group in self.get_enabled_data_keys():
            self.spline_serieses[group].setVisible(False)

        for series in self.serieses.values():
            series.setVisible(False)

        for halo_series in self.halo_serieses.values():
            halo_series.setVisible(False)
            halo_series.setVisible(True)

        # make spline series visible
        if self.show_spline_series:
            for group in self.get_enabled_data_keys():
                self.spline_serieses[group].setVisible(True)

        for series in self.serieses.values():
            series.setVisible(True)

        chart_legend = self.chart().legend()

        for halo_series in self.halo_serieses.values():
            chart_legend.markers(halo_series)[0].setVisible(False)

        for series in self.serieses.values():
            chart_legend.markers(series)[0].setVisible(False)

        # hide markers of spline series
        for group in self.data_keys:
            chart_legend.markers(self.spline_serieses[group])[0].setVisible(True)  # (False)

        # for better visibility:

        # hide all text labels
        if not self.show_labels:
            for tl in self.text_labels.values():
                tl.setVisible(False)

        # show legend on the left
        chart_legend.setVisible(False)
        chart_legend.setFont(QFont("Inter", 20, QFont.Bold))
        chart_legend.setAlignment(Qt.AlignLeft)
        # for group in self.get_enabled_data_keys():
        #     chart_legend.markers(self.spline_serieses[group])[0].setVisible(True)
        # for group in self.get_not_enabled_data_keys():
        #     chart_legend.markers(self.spline_serieses[group])[0].setVisible(False)

        print("update {} took: {} seconds".format("labels", time.time() - t))

    def set_show_spline_series(self, show: bool):
        self.show_spline_series = show
        self.update_attributes(self.update_labels)

    @staticmethod
    def set_saturation(color: QColor, sat: float):
        """
        Returns a copy of color with saturation set to @sat
        :param color:
        :param sat: saturation [0, 1]
        :return:
        """
        assert 0 <= sat <= 1, "sat has to be in [0, 1], got {}".format(sat)
        c = QColor()
        # c.setHsv(color.hue(), int(255 * sat), color.value(), color.alpha())
        f = 0.7
        c.setHsv(color.hue(), int(f * (255 * sat) + (1-f) * 255), color.value(), 255)  # int(f * (255 * sat) + (1-f) * 255))
        # print(c, sat)
        return c

    @Slot()
    def mouse_pressed(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_button = "left"
        elif event.button() == Qt.MouseButton.RightButton:
            self.mouse_button = "right"
        else:
            self.mouse_button = ""

    @Slot()
    def mouse_released(self, event):
        self.mouse_button = None

    @Slot()
    def point_clicked(self, group: str):
        self.group_clicked_signal.emit(group,
                                       0,  # full range of volume
                                       len(self._original_data_sat_con[group]),
                                       self.mouse_button)

    @Slot()
    def point_hovered(self, point, state: bool, group: str):
        if not state:
            return

        if self.data_red_mode == self.DataReductionMode.GROUP or self.data_mode == self.DataMode.TimeSeriesData:
            return
            # self.group_patch_hovered_signal.emit(self.data_keys[find_point_idx(point, self.series)],
            #                                0,  # full range of volume
            #                                len(self.data[self.data_keys[find_point_idx(point, self.series)]]),
            #                                self.mouse_button)
        elif self.data_red_mode == self.DataReductionMode.PATCH:
            patch_idx = find_point_idx(point, self.serieses[group])

            patch_start = patch_idx if self.part_strat == self.PartitionStrategy.Overlapping else patch_idx * self.get_temp_patch_size()
            patch_end = patch_start + self.get_temp_patch_size()
            self.group_patch_hovered_signal.emit(
                group, patch_start, patch_end
            )
        else:
            raise RuntimeError("unknown proj_mode")

    def select_group_or_patch(self, groups_to_select: list, time_step_from: int, time_step_to: int):
        # self.series.deselectAllPoints()  # deselect all other points first
        if self.data_red_mode != self.DataReductionMode.PATCH:
            return

        # compute patch idxs
        patch_start = int(time_step_from / self.get_temp_patch_size())
        patch_end = int(time_step_to / self.get_temp_patch_size())

        if patch_start == patch_end:
            patch_end += 1

        enabled_group_keys = self.get_enabled_data_keys()

        for group in groups_to_select:
            if group not in enabled_group_keys:
                continue

            to_select = list(range(patch_start, patch_end))

            # print("to select", group, to_select)

            # deselect points in halo series
            self.halo_serieses[group].selectPoints(to_select)

            # select points in series
            self.serieses[group].selectPoints(to_select)

    def update_series(self):
        t = time.time()
        for series in self.serieses.values():
            series.clear()
            series.clearPointsConfiguration()

        for halo_series in self.halo_serieses.values():
            halo_series.clear()
            halo_series.clearPointsConfiguration()

        # clear spline series
        for group in self.data_keys:
            self.spline_serieses[group].clear()

        projection = self.projection[self.proj_alg]

        if self.data_red_mode == self.DataReductionMode.GROUP or self.data_mode == self.DataMode.TimeSeriesData:
            for group in self.data_keys:
                series = self.serieses[group]
                if group.startswith("experiment"):
                    series.setMarkerSize(self.series_exp_marker_size * 2)
                else:
                    series.setMarkerSize(self.series_marker_size * 2)
            # self.halo_series.setMarkerSize(self.series_marker_size * 2 * 1.25)

            for i, k in enumerate(self.get_enabled_data_keys()):
                new_point = (projection[i][0], projection[i][1])

                self.serieses[k].append(*new_point)

                # add point to spline series
                self.spline_serieses[k].append(*new_point)

        elif self.data_red_mode == self.DataReductionMode.PATCH:
            for group in self.data_keys:
                series = self.serieses[group]
                if group.startswith("experiment"):
                    series.setMarkerSize(self.series_exp_marker_size)
                else:
                    series.setMarkerSize(self.series_marker_size)
                series.block_qt_configuration_updates(True)

            # self.halo_series.setMarkerSize(self.series_marker_size * 1.25)
            group_beginnings = self.enabled_group_beginnings

            for i, p_i in enumerate(projection):
                new_point = (p_i[0], p_i[1])

                group_idx, group_beg_idx = self.get_group_idx_of_projection_row(i)
                group = self.data_keys[group_idx]

                self.serieses[group].append(*new_point)

                patch_num_in_group = i - group_beginnings[group_beg_idx]

                c = group_to_color[group]
                sat = (float(patch_num_in_group) / len(self.encodings[group]))
                # print(i, group_idx, group_beg_idx, patch_num_in_group, len(self.encodings[self.data_keys[group_idx]]),
                #       sat, group_beginnings)

                self.serieses[group].setPointConfiguration(patch_num_in_group, {
                                                      ConfigurableScatterSeries.PointConfiguration.Color:
                                                          self.set_saturation(c, sat),
                                                      # ConfigurableScatterSeries.PointConfiguration.Size: point_size
                                                      })

                # add point to spline series
                self.spline_serieses[self.data_keys[group_idx]].append(*new_point)

            for series in self.serieses.values():
                series.block_qt_configuration_updates(False)
                series.update_points_configuration()
        else:
            raise RuntimeError("unknown projection mode")

        if not self._is_full_run:
            # copy series to halo series
            for group in self.data_keys:
                series = self.serieses[group]
                halo_series = self.halo_serieses[group]

                for p_idx, p in enumerate(series.points()):
                    halo_series.insert(p_idx, p)
                    halo_series.deselectAllPoints()
        print("update {} took: {} seconds".format("series", time.time() - t))

    def get_temp_patch_size(self):
        encoder, _, _ = self.get_enc_norm_conf()

        if self.temp_p_size == self.TemporalPatchSize.Single:
            return 1
        if self.temp_p_size == self.TemporalPatchSize.Full:
            return encoder.input_shape[1]

        raise RuntimeError("unknown temp patch size: {}".format(self.temp_p_size))

    def _compute_input(self, data: np.array):
        """
        Computes full spatial patches from the original data with provided temporal patch size from encoder
        :param data:
        :return:
        """
        patch_dim_t = self.get_temp_patch_size()

        data_dim_t = data.shape[0]

        # compute amount patches
        assert data_dim_t >= patch_dim_t, "cannot compute patch of size {} from data of size {}".format(
            patch_dim_t, data_dim_t)

        if self.part_strat == self.PartitionStrategy.Overlapping:
            # use overlapping patches
            num_patches = data_dim_t - patch_dim_t + 1

            # create empty input_batch
            input_batch = np.empty((num_patches, patch_dim_t, *data.shape[1:]))

            # build input batch of copies from original data
            for i in range(num_patches):
                input_batch[i] = data[i:i + patch_dim_t]
        elif self.part_strat == self.PartitionStrategy.NonOverlapping:
            # don't use overlapping patches with except maybe the last one
            num_patches = math.ceil(data_dim_t / patch_dim_t)

            # create empty input_batch
            input_batch = np.empty((num_patches, patch_dim_t, *data.shape[1:]))

            # build input batch of copies from original data
            for i in range(num_patches - 1):
                input_batch[i] = data[i * patch_dim_t:(i + 1) * patch_dim_t]

            # add last one
            input_batch[num_patches - 1] = data[-patch_dim_t:]
        else:
            raise RuntimeError("unknown partition strategy: {}".format(self.part_strat))

        return input_batch

    @staticmethod
    def add_z_axis(d: np.array):
        d = d.reshape((d.shape[0], 1, *d.shape[1:]))  # add z axis
        return d

    @staticmethod
    def prepare_data(data: dict, normalizer, config):
        """
        Prepare the loaded data to be in correct shape for normalizer and then prediction.
        Expects data to be in shape (T, Y, X, V), returns shape (T, 1, Y, X, V) with dtype np.float32

        Also performs volume cropping and downsampling.
        :param self:
        :param x:
        :return:
        """
        #_, normalizer, config = enc_norm_conf_multivar  # always use multivar for normalizing here
        data = data.copy()

        # min x and y to minimum of width and height
        min_x = min(data[group].shape[2] for group in data)
        min_y = min(data[group].shape[1] for group in data)

        # clip time to 144, center width and height and clip to minimum, and add z-axis
        # for group in data:
        #     x_start = int((data[group].shape[2] - min_x) / 2)
        #     y_start = int((data[group].shape[1] - min_y) / 2)
#
        #     data[group] = ProjectionWidget.add_z_axis(data[group][:, y_start:min_y + y_start, x_start:min_x + x_start])

        # only add z axis since we now use rescaled data
        for group in data:
            data[group] = ProjectionWidget.add_z_axis(data[group])

        # change dtype to np.uint or np.float32
        for group in data:
            data[group] = data[group].astype(np.float32)

        # crop and downsample volumes
        for group in data:
            x = data[group]

            if "volumeCrop" in config:
                vc = [c for c in config["volumeCrop"][1:-1].split('slice') if "(" in c and ")" in c]
                vc = [[a.strip() for a in remove_except_between_chars(c, '(', ')').split(',')] for c in vc]
                vc = (*[slice(*[int_or_none(a) for a in c]) for c in vc],)

                x = x[vc]

            if "downsampleFactor" in config:
                ds = int(config["downsampleFactor"])
                if ds != 1:
                    x = downsample_volume(x, ds)

            # reshape to minibatch
            x = x.reshape((1, 1, *x.shape))

            # normalize
            normalizer.scale(x, True)

            # remove extra shape
            x = x.reshape(x.shape[2:])

            data[group] = x

        return data

    def update_input(self, group: str):
        return self._cached_update_input(group)

    def get_update_input_cache_keys(self) -> tuple:
        return self.data_mode, self.temp_p_size, self.part_strat, self.dissim_measure, self.compare_all_via_seg_map

    def get_update_dissim_input_cache_keys(self) -> tuple:
        return self.get_update_input_cache_keys() + (self.data_red_mode, *self.seg_map_th)

    def get_update_dissim_mat_cache_keys(self) -> tuple:
        return self.get_update_dissim_input_cache_keys() + (self._use_fast_dtw, )

    def get_update_time_series_dissim_matrix_keys(self) -> tuple:
        return (self.data_mode, )

    def _cached_update_input(self, group):
        cache = self._cached_update_input_cache
        key = (self._cached_update_input.__name__, group) + self.get_update_input_cache_keys()
        # compare_all_via_seg_map = True  # debug

        do_not_use_cache = True  # don't use cache for input

        if cache.has_key(key) and not do_not_use_cache:
            v = cache[key]
        else:
            g_data = self.data[group]

            v = self._compute_input(g_data)

            if not do_not_use_cache:
                cache[key] = v

        self.encodings[group] = v

    def update_inputs(self):
        """
        Updates the encodings based on the current data configurations.
        :return:
        """
        t = time.time()
        if self.data_mode == self.DataMode.TimeSeriesData or self.data_mode == self.DataMode.TimeSeriesData_CombinedFeatures:
            for group in self.data:
                # special handling for time series data:
                # just copy time series data table into encodings

                # fake data for experiments
                if group.startswith("experiment"):
                    self.encodings[group] = self.data_time_series["austin"]
                else:
                    self.encodings[group] = self.data_time_series[group]
        else:
            data = self._original_data_sat_con.copy()

            # handle seg map here before normalization of encoder
            if self.compare_all_via_seg_map:
                data = {group: self._to_binary(data[group]) for group in data}

            if self.dissim_measure == self.DissimilarityMeasure.S4 or self.dissim_measure == self.DissimilarityMeasure.S4_Spatially_subdivided:
                self.data = self.prepare_data(data, *self.get_enc_norm_conf()[1:])
            else:
                self.data = data

            for group in self.data:
                self.update_input(group)

        group_beginnings = [(sum([len(self.encodings[k]) for k in self.data_keys[:i]])) for i, group in
                            enumerate(self.data_keys)]
        self.group_beginnings = group_beginnings
        self.update_enabled_group_beginnnings()
        print("update {} took: {} seconds".format("encodings", time.time() - t))

    @staticmethod
    def divide_into_small_patches_and_predict(X: np.array, encoder: keras.Model):
        """
        Given X of (num_patches, t, height, width, v), divides it into spatial patches based on encoder input_shape
            -> (num_patches, divided_height, divided_width, t, patch_height, patch_width, v) then predicts to
            -> (num_patches, divided_height, divided_width, features)
        :param X:
        :param encoder:
        :return:
        """
        enc_t, _, enc_h, enc_w = encoder.input_shape[1:5]
        x_h, x_w = X.shape[3:5]
        div_h, div_w = math.ceil(x_h / enc_h), math.ceil(x_w / enc_w)
        X_= np.empty_like(X, shape=(X.shape[0], div_h, div_w, enc_t, 1, enc_h, enc_w, X.shape[-1]))

        for p_i in range(X.shape[0]):
            for div_h_i in range(div_h):
                for div_w_i in range(div_w):
                    x_access_h = min(x_h - enc_h, div_h_i * enc_h)
                    x_access_w = min(x_w - enc_w, div_w_i * enc_w)
                    X_[p_i, div_h_i, div_w_i] = X[p_i, :enc_t, :, x_access_h:x_access_h+enc_h, x_access_w:x_access_w+enc_w]

        # print("encoder in shape:", encoder.input_shape)
        # print(f"X_.shape: {X_.shape}")
        # flatten first three dimensions (num_patches, div_h, div_w, *) -> (num_patches * div_h * div_w, *)
        s = X.shape[0] * div_h * div_w
        X_ = X_.reshape((s, enc_t, 1, enc_h, enc_w, X.shape[-1]))
        X_ = encoder.predict(X_)
        # reconstruct shape
        X_ = X_.reshape((X.shape[0], div_h, div_w, *X_.shape[1:]))

        return X_

    def _cached_update_dissimilarity_matrix(self):
        cache = self._cached_update_dissimilarity_matrix_cache

        key = (self._cached_update_dissimilarity_matrix.__name__, ) + self.get_update_dissim_mat_cache_keys()

        if cache.has_key(key):
            v = cache[key]
        else:
            encoder, _, _ = self.get_enc_norm_conf()

            if self.dissim_measure == self.DissimilarityMeasure.S4 or self.dissim_measure == self.DissimilarityMeasure.S4_Spatially_subdivided:
                if self.data_red_mode == self.DataReductionMode.GROUP:
                    # we use the encoder to reduce the input patch for each group to one feature vector
                    # self.dissim_input is here a list of lists with patches. shape: (groups, num_patches, patch_size_t=(3 or 1), 1, height, width, v)
                    X = self.dissim_input

                    # deal with smaller patch sizes when using S4
                    if self.temp_p_size == self.TemporalPatchSize.Full:
                        # do nothing
                        X = X
                    elif self.temp_p_size == self.TemporalPatchSize.Single:
                        # repeat single slice to amount of full patch size
                        X = list(map(lambda x: np.repeat(x, encoder.input_shape[1], 1), X))
                    else:
                        raise RuntimeError("unknown temporal patch size: {}".format(self.temp_p_size))

                    if self.dissim_measure == self.DissimilarityMeasure.S4_Spatially_subdivided:
                        precompute = lambda x: self.divide_into_small_patches_and_predict(x, encoder)
                    else:
                        precompute = lambda x: encoder.predict(x)

                elif self.data_red_mode == self.DataReductionMode.PATCH:
                    # we use the encoder on all input patches at once
                    # self.dissim_input is here a list of patches. shape: (patches, patch_size_t=(3 or 1), 1, height, width, v)
                    X = np.array(self.dissim_input)

                    # deal with smaller patch sizes when using S4
                    if self.temp_p_size == self.TemporalPatchSize.Full:
                        # do nothing
                        X = X
                    elif self.temp_p_size == self.TemporalPatchSize.Single:
                        # repeat single slice to amount of full patch size
                        X = np.repeat(X, encoder.input_shape[1], 1)
                    else:
                        raise RuntimeError("unknwon temporal patch size: {}".format(self.temp_p_size))

                    if self.dissim_measure == self.DissimilarityMeasure.S4:
                        X = encoder.predict(X)  # we can just use this since we use full spatial domain
                    elif self.dissim_measure == self.DissimilarityMeasure.S4_Spatially_subdivided:
                        # need to adjust prediction since we cannot feed full spatial domain in partially spatial enc.
                        X = self.divide_into_small_patches_and_predict(X, encoder)  # output will be of shape (patches, divided_height, divided_width, features)

                    # do nothing
                    precompute = lambda x: x
                else:
                    raise RuntimeError("unknown data red mode: {}".format(self.data_red_mode))

                metric = lambda x, y: np.linalg.norm(x - y, ord=1)

                if self.dissim_measure == self.DissimilarityMeasure.S4_Spatially_subdivided:
                    # wrap metric
                    tmp_metric = metric
                    metric = lambda x, y: tmp_metric(x.flatten(), y.flatten())

            elif self.dissim_measure == self.DissimilarityMeasure.Euclidean:
                X = self.dissim_input
                precompute = flatten_extra_dimensions  # keeps patch dimension and flattens others
                metric = lambda x, y: np.linalg.norm(x - y, ord=2)
            elif self.dissim_measure == self.DissimilarityMeasure.Manhattan:
                X = self.dissim_input
                precompute = flatten_extra_dimensions
                metric = lambda x, y: np.linalg.norm(x - y, ord=1)
            elif self.dissim_measure == self.DissimilarityMeasure.Wasserstein:
                def wasserstein_metric(a, b):
                    """
                    a and b are lists of histograms. for multivar data, len(a) = len(b) = 2 else 1.
                    computes the wasserstein distance between the histograms of provided patches a and b for sat and con (or just one of both)
                    if multivar, averages the wasserstein distance between sat and con
                    :param a:
                    :param b:
                    :return:
                    """
                    assert len(a) == len(b) and len(a) > 0
                    return sum(scipy.stats.wasserstein_distance(a[i], b[i], np.arange(256), np.arange(256))
                               for i in range(len(a))) / len(a)

                def prepare_histograms(x: np.array):
                    """
                    flattens the patches in the provided array and computes their histogram
                    :param x:
                    :return:
                    """
                    bins = 256
                    bin_range = [0, 255]  # here inclusive upper bound

                    take_only_first_frame = False

                    if self.data_red_mode == self.DataReductionMode.PATCH:
                        # each x is a single patch itself
                        if take_only_first_frame:
                            x = x[0, ...]
                        if self.data_mode == self.DataMode.Multivariate:
                            return tuple(
                                np.array(numpy.histogram(x[..., i].flat, bins=bins, range=bin_range)[0]) / x.size for
                                i in range(2))
                            # return [_histogram(x[..., i]) for i in range(2)]
                        else:
                            return (np.array(numpy.histogram(x.flat, bins=bins, range=bin_range)[0]) / x.size, )
                    elif self.data_red_mode == self.DataReductionMode.GROUP:
                        # each x is list of patches
                        if take_only_first_frame:
                            x = [x_[0, ...] for x_ in x]
                        if self.data_mode == self.DataMode.Multivariate:
                            return [tuple(
                                np.array(numpy.histogram(x_[..., i].flat, bins=bins, range=bin_range)[0]) / x_.size for
                                i in range(2)) for x_ in x]
                        else:
                            return [(np.array(numpy.histogram(x_.flat, bins=bins, range=bin_range)[0]) / x_.size, ) for x_
                                    in x]
                    else:
                        raise RuntimeError("unknown data red mode: {}".format(self.data_red_mode))

                X = self.dissim_input

                print(len(X))
                if hasattr(X, "shape"):
                    print(X.shape)

                # scale X to image values
                if self.data_mode == self.DataMode.Multivariate:
                    X = [interp_variables(x, xp=lambda x_, i: (self.min_max[0][i], self.min_max[1][i])).astype(np.uint8) for x in X]
                else:
                    if self.data_mode == self.DataMode.Saturation:
                        i = 0
                    elif self.data_mode == self.DataMode.Concentration:
                        i = 1
                    X = [interp_variables(x, xp=lambda x_, _: (self.min_max[0][i], self.min_max[1][i])).astype(np.uint8) for x in X]

                precompute = prepare_histograms
                metric = wasserstein_metric
            else:
                raise RuntimeError("unknown dissim measure: {}".format(self.dissim_measure))

            if self.data_red_mode == self.DataReductionMode.GROUP:
                if self._use_fast_dtw:
                    # wrap with dtw
                    metric_final = lambda a, b: fastdtw.fastdtw(a, b, dist=metric)[0]
                else:
                    # just sum up all the errors between the individual patches, assumes we always have the same amount of patches
                    metric_final = lambda a, b: np.sum(metric(a[i], b[i]) for i in range(min(len(a), len(b))))
            elif self.data_red_mode == self.DataReductionMode.PATCH:
                # do nothing
                metric_final = metric
            else:
                raise RuntimeError("unknown data red mode: {}".format(self.data_red_mode))

            if "S4" not in self.dissim_measure and not self.compare_all_via_seg_map and not "Wasserstein" in self.dissim_measure:
                # don't do this for S4 or Wasserstein distance since it does not make sense.
                # For S4, we already prepared the data, i.e., normalization etc. -> threshold does not apply to the corect values
                # For Wasserstein, we already computed the historgrams and applying the 'to_binary' stuff does not make sense (and probably even fails here)
                experiment_name = "experiment"
                n_v = 2 if self.data_mode == self.DataMode.Multivariate else 1

                def use_seg_for_experimental_data_metric_wrapper(a, b, a_idx, b_idx):
                    if self.data_red_mode == self.DataReductionMode.GROUP:
                        g_a = self.data_keys[a_idx]
                        g_b = self.data_keys[b_idx]
                    else:
                        g_a = self.data_keys[self.get_group_idx_of_dissimilarity_matrix_row(a_idx)]
                        g_b = self.data_keys[self.get_group_idx_of_dissimilarity_matrix_row(b_idx)]

                    # print("groups:", g_a, g_b, "indices:", a_idx, b_idx)

                    if g_b.startswith(experiment_name) and not g_a.startswith(experiment_name):
                        # print("change: ", g_a, g_b, a.shape, b.shape)
                        # fix flatten extra dimensions for this
                        if precompute == flatten_extra_dimensions:
                            a = a.reshape((len(a), int(a.size / (n_v * len(a))), n_v))
                            a = self._to_binary(a)
                            a = flatten_extra_dimensions(a)
                        else:
                            a = self._to_binary(a)

                    if g_a.startswith(experiment_name) and not g_b.startswith(experiment_name):
                        # print("change: ", g_b, g_a, b.shape, a.shape)
                        if precompute == flatten_extra_dimensions:
                            b = b.reshape((len(b), int(b.size / (n_v * len(b))), n_v))
                            b = self._to_binary(b)
                            b = flatten_extra_dimensions(b)
                        else:
                            b = self._to_binary(b)

                    return metric_final(a, b)

                metric_final_2 = use_seg_for_experimental_data_metric_wrapper
                metric_with_ensembe_index = True
            else:
                metric_final_2 = metric_final
                metric_with_ensembe_index = False

            v = pairwise_distance(X, precompute=precompute, metric=metric_final_2,
                                  metric_with_ensemble_index=metric_with_ensembe_index)
            cache[key] = v

        self.dissimilarity_matrix = v

    def _cached_time_series_dissim_matrix(self):
        cache = self._cached_update_dissimilarity_matrix_cache

        key = (self._cached_time_series_dissim_matrix.__name__, ) + self.get_update_time_series_dissim_matrix_keys()

        if cache.has_key(key):
            v = cache[key]
        else:
            if self.data_mode == self.DataMode.TimeSeriesData:
                X = self.dissim_input

                precompute = lambda x: x

                dissim_matrices = []

                if self._use_fast_dtw:
                    metric_ = lambda a, b: fastdtw.fastdtw(a, b)[0]
                else:
                    metric_ = lambda a, b: np.linalg.norm(a - b, ord=1)

                # compute dissim mat for each key
                for i in range(len(time_series_namings)):
                    x = [x_[i] for x_ in X]

                    dissim_mat = pairwise_distance(
                        x, precompute=precompute, metric=metric_
                    )

                    # normalize all dissim mats individually to 0 and 1
                    dissim_mat = np.interp(dissim_mat, (dissim_mat.min(), dissim_mat.max()), (0, 1))

                    dissim_matrices.append(dissim_mat)

                v = dissim_matrices
            else:
                X = self.dissim_input
                max_len = max(len(x) for x in X)
                print(max_len, min(len(x) for x in X))

                def metric_(a, b):
                    if self._use_fast_dtw:
                        return fastdtw.fastdtw(a, b)[0]
                    else:
                        return np.linalg.norm(a - b, ord=1)

                v = pairwise_distance(X, precompute=None, metric=metric_)
                assert v is not None
            cache[key] = v

        self.dissimilarity_matrix = v

    def update_dissimilarity_matrix(self):
        """
        Updates the similarity matrix based on the computed encodings.
        :return:
        """
        t = time.time()
        if self.data_mode == self.DataMode.TimeSeriesData or self.data_mode == self.DataMode.TimeSeriesData_CombinedFeatures:
            self._cached_time_series_dissim_matrix()
        else:
            self._cached_update_dissimilarity_matrix()

        print("update {} took: {} seconds".format("dissim mat", time.time() - t))

    def find_corners(self, projection: np.array):
        mins = np.min(projection, axis=0)  # x_min, y_min
        maxs = np.max(projection, axis=0)  # x_max, y_max

        if projection.shape[-1] == 2:
            dx, dy = maxs - mins  # dx, dy

            smallest_distance_i = np.argmin([dx, dy])

            o = np.abs((dx - dy) / 2)

            mins[smallest_distance_i] -= o
            maxs[smallest_distance_i] += o

        return mins, maxs

    def update_projection(self):
        """
        Update the projection based on the precomputed dissimilarity matrix.
        :return:
        """
        t = time.time()
        try:
            if self.data_mode == self.DataMode.TimeSeriesData:
                # update dissim matrix
                dissim_matrices = self.dissimilarity_matrix

                # add dissim mat by weighting
                dissim_mat = np.zeros((len(self.data_keys), len(self.data_keys)))
                s = 0
                for i, key in enumerate(time_series_namings_keys):
                    if self.enabled_times_series[key]:
                        dissim_mat += dissim_matrices[i]
                        s += 1
                if s == 0:
                    warnings.warn("no time series enabled; skip computing dissim matrix")
                    return
                dissim_mat = dissim_mat / s

                final_dissim_mat = dissim_mat
            else:
                final_dissim_mat = self.dissimilarity_matrix

            if self.data_red_mode == self.DataReductionMode.GROUP:
                # filter groups
                to_delete = [i for i, k in enumerate(self.data_keys) if not self.enabled_groups[k]]
                final_dissim_mat = np.delete(np.delete(final_dissim_mat, to_delete, 0), to_delete, 1)
            elif self.data_red_mode == self.DataReductionMode.PATCH:
                # filter patches from groups
                nek = self.get_not_enabled_data_keys()
                for k_i in reversed(range(len(self.data_keys))):
                    if self.data_keys[k_i] in nek:
                        end = None if k_i == len(self.data_keys) - 1 else self.group_beginnings[k_i + 1]
                        tod = slice(self.group_beginnings[k_i], end)
                        final_dissim_mat = np.delete(np.delete(final_dissim_mat, tod, 0), tod, 1)
            else:
                raise RuntimeError("unknown data red mode: {}".format(self.data_red_mode))

            self.projection = {
                alg_name: alg(final_dissim_mat) for alg_name, alg in self.get_proj_algorithm()
            }

            # find corners
            corners = {
                alg_name: self.find_corners(self.projection[alg_name]) for alg_name, _ in self.get_proj_algorithm()
            }

            # scale projection to [0, 1]
            self.projection = {
                alg_name: interp_variables(self.projection[alg_name], xp=lambda x, i: (corners[alg_name][0][i],
                                                                                       corners[alg_name][1][i]),
                                               fp=lambda x, i: (0, 1)) for alg_name in self.projection
            }

            if self.proj_dim == self.ProjectionDim.TwoD:
                return
            elif self.proj_dim == self.ProjectionDim.OneD:
                for alg_name in self.projection:
                    projection = self.projection[alg_name]
                    if self.data_red_mode == self.DataReductionMode.PATCH:
                        # use time as second dimension on 1D projections
                        # Note: width does vary depending on temporal patch size and if overlapping or not
                        new_projection = np.zeros((len(projection), 2), dtype=projection.dtype)

                        group_beginnings = self.enabled_group_beginnings

                        max_length = max([len(self.encodings[g]) for g in self.data_keys])

                        for i, p_i in enumerate(projection):
                            group_idx, group_beg_idx = self.get_group_idx_of_projection_row(i)

                            patch_num_in_group = i - group_beginnings[group_beg_idx]

                            new_projection[i] = (patch_num_in_group / max_length, p_i)

                        self.projection[alg_name] = new_projection
                        return
                    elif self.data_red_mode == self.DataReductionMode.GROUP:
                        # plot horizontally in 1D
                        new_projection = np.zeros((len(projection), 2), dtype=projection.dtype)

                        for i, p_i in enumerate(projection):
                            new_projection[i] = (0.4, p_i)

                        self.projection[alg_name] = new_projection
                        return
                    else:
                        raise RuntimeError("unknown data red mode: {}", self.data_red_mode)
            else:
                raise RuntimeError("unknown proj dim: {}".format(self.proj_dim))
        finally:
            print("update {} took: {} seconds".format("projection", time.time() - t))
