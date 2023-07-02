import json
import os.path
import random
import time

import numpy as np
import qtpex
from PySide6.QtCharts import QValueAxis
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import QMainWindow, QWidget, QToolButton, QSizePolicy, QFileDialog, QRadioButton, \
    QLabel, QButtonGroup, QCheckBox, QApplication
from pybrutil.np_util.data import interp_variables

from src.projection_widget import ProjectionWidget
from src.time_series_view import TimeSeriesViewWidget, time_series_namings_keys, time_series_namings
from src.ui_volume_viewer import Ui_VolumeViewer
from src.ui_mainwindow import Ui_MainWindow
from src.ui_volume_settings import Ui_volumeSettings
from PySide6.QtWidgets import QGridLayout
from PySide6.QtCore import Qt, Slot, QPoint
from src.util import resize_to_power_of_two_and_scale_to_uint8, make_custom_3d_volume_item_from_data, get_attributes

from qtpex.qt_widgets.transferfunction_widget import TransferFunctionWidget

from src.volume_renderer_view import VolumeRenderGraph
from src.constants import groups, screenshot_locations, screenshot_randint_max


class MainWindow(QMainWindow):
    def __init__(self, data: dict):
        # def __init__(self, volume_item_cache_left: dict, volume_item_cache_right: dict):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._allow_experimental_data_for_s4 = True

        # init container widget with content created by QT Designer
        self.ui.leftViewContainer.ui = Ui_VolumeViewer()
        self.leftUi = self.ui.leftViewContainer.ui
        self.leftUi.setupUi(self.ui.leftViewContainer)
        self.leftUi.loadDataButton.clicked.connect(self.update_load_data_tool_button_actions)

        # init container widget with content created by QT Designer
        self.ui.rightViewContainer.ui = Ui_VolumeViewer()
        self.rightUi = self.ui.rightViewContainer.ui
        self.rightUi.setupUi(self.ui.rightViewContainer)
        self.rightUi.loadDataButton.clicked.connect(self.update_load_data_tool_button_actions)

        # init settings ui
        self.ui.tab_left.ui = Ui_volumeSettings()
        self.ui.tab_left.ui.setupUi(self.ui.tab_left)
        self.tab_ui = self.ui.tab_left.ui

        self.ui.tab_right.ui = Ui_volumeSettings()
        self.ui.tab_right.ui.setupUi(self.ui.tab_right)
        self.tab_2_ui = self.ui.tab_right.ui

        # connect saturation concentration toggle button
        self.leftUi.switchSatCon.clicked.connect(self.toggle_saturation_concentration)
        self.rightUi.switchSatCon.clicked.connect(self.toggle_saturation_concentration)

        self.volume_renderer_widget_right = None
        self.volume_renderer_widget_left = None

        self.container_grid = QGridLayout()
        self.container_grid2 = QGridLayout()

        # self.setWindowState(Qt.WindowNoState)
        self.data = data

        for ui in [self.leftUi, self.rightUi]:
            ui.volume_item_cache = dict()

        self.showMaximized()

        # compute min per data
        self.min_ = [min([self.data[group]["sat_con"][:, :, :, i].min() for group in self.data if not group.startswith("experiment")]) for i in range(2)]

        # compute max p percent data value
        p = 0.98
        self.max_at_p_percent = [max([np.sort(self.data[group]["sat_con"][:, :, :, i].flatten())
                             [int(p * len(self.data[group]["sat_con"][:, :, :, i].flatten()))]
                             for group in self.data if not group.startswith("experiment")]) for i in range(2)]

        # scale experimental data to new min and max
        for group in self.data:
            if group.startswith("experiment"):
                self.data[group]["sat_con"] = interp_variables(self.data[group]["sat_con"].astype(float), fp=lambda _, i: (self.min_[i], self.max_at_p_percent[i]), in_place=False)

        for group in self.data:
            print(f"group: {group}")
            d = self.data[group]
            for v in ["t", "x", "y"]:
                if v in d:
                    dv = d[v]
                    dv = sorted(dv)
                    print("{} range: [{}, {}]".format(v, dv[0], dv[-1]))

            print("sat_con.shape: {}".format(d["sat_con"].shape))
            print("")

        self.customize_ui()

    def customize_ui(self):
        """
        The ui is the provided ui file from the QT-Designer.
        Since I would like to keep the original ui as it was created by QT-Designer and compiled with pyside6-uic,
            this function initializes and (possibly) modifies the placeholders of the original UI.
        :return:
        """
        # container_grid.addWidget(QWidget.createWindowContainer(VolumeRenderWidget()))
        self.leftUi.volume_renderer = VolumeRenderGraph()
        self.volume_renderer_widget_left = QWidget.createWindowContainer(self.leftUi.volume_renderer)

        self.rightUi.volume_renderer = VolumeRenderGraph()
        self.volume_renderer_widget_right = QWidget.createWindowContainer(self.rightUi.volume_renderer)

        # self.setCentralWidget(QWidget.createWindowContainer(self.window3d))
        self.container_grid.addWidget(self.volume_renderer_widget_left)
        self.container_grid2.addWidget(self.volume_renderer_widget_right)

        self.ui.leftViewContainer.ui.volumeViewContainer.setLayout(self.container_grid)
        self.ui.rightViewContainer.ui.volumeViewContainer.setLayout(self.container_grid2)

        self.leftUi.loadDataButton.click()
        # if len(self.ui.leftViewContainer.ui.loadDataButton.actions()) > 0:
        #     self.ui.leftViewContainer.ui.loadDataButton.actions()[0].trigger()

        self.rightUi.loadDataButton.click()
        # if len(self.ui.rightViewContainer.ui.loadDataButton.actions()) > 0:
        #     self.ui.rightViewContainer.ui.loadDataButton.actions()[0].trigger()

        self.left_transfer_function = TransferFunctionWidget("linear")  #, #x_range=(min(self.min_),
                                                                       #         max(self.max_at_p_percent)))
        self.tab_ui.transferFunctionContainer.addWidget(self.left_transfer_function)
        self.left_transfer_function.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_transfer_function.changed_signal.connect(lambda: self.update_color_table(
            self.left_transfer_function, self.leftUi.volume_renderer))
        self.tab_ui.applyTfToBoth.clicked.connect(lambda: self.right_transfer_function.copy_from_other_tf_widget(
            self.left_transfer_function))
        self.tab_ui.loadTF.clicked.connect(self.load_transferfunction)
        self.tab_ui.saveTF.clicked.connect(self.save_transferfunction)

        self.right_transfer_function = TransferFunctionWidget("linear")  #, x_range=((min(self.min_),
                                                                      #            max(self.max_at_p_percent))))
        self.tab_2_ui.transferFunctionContainer.addWidget(self.right_transfer_function)
        self.right_transfer_function.changed_signal.connect(lambda: self.update_color_table(
            self.right_transfer_function, self.rightUi.volume_renderer))
        self.tab_2_ui.applyTfToBoth.clicked.connect(lambda: self.left_transfer_function.copy_from_other_tf_widget(
            self.right_transfer_function))
        self.tab_2_ui.loadTF.clicked.connect(self.load_transferfunction)
        self.tab_2_ui.saveTF.clicked.connect(self.save_transferfunction)

        for tf in [self.left_transfer_function, self.right_transfer_function]:
            self.setup_transferfunctions(tf)

        # load default transferfunctions
        for tf in [self.left_transfer_function, self.right_transfer_function]:
            self._load_transferfunction_from_file(os.path.abspath(os.path.join("./transferfunctions", "default.json")), tf)

        ## projection container
        self.projection_widget = ProjectionWidget(self.data,
                                                  min_max=[self.min_, self.max_at_p_percent],
                                                  data_red_mode=ProjectionWidget.DataReductionMode.GROUP,
                                                  proj_alg=ProjectionWidget.ProjAlgorithm.MDS,
                                                  proj_dim=ProjectionWidget.ProjectionDim.TwoD,
                                                  dissim_measure=ProjectionWidget.DissimilarityMeasure.S4,
                                                  part_strat=ProjectionWidget.PartitionStrategy.NonOverlapping,
                                                  temp_p_size=ProjectionWidget.TemporalPatchSize.Full)
        self.projection_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.projection_widget.setFixedSize(550, 550)
        self.ui.projection_container.addWidget(self.projection_widget)

        self.projection_widget.group_clicked_signal.connect(self.group_clicked)
        self.projection_widget.group_patch_hovered_signal.connect(self.patch_hovered)

        # connect control widget to analysis widget
        self.init_projection_control_tool_buttons()

        # save pointers to active cameras
        self.left_camera = self.leftUi.volume_renderer.scene().activeCamera()
        self.right_camera = self.rightUi.volume_renderer.scene().activeCamera()

        self.couple_signals_to_slots_of_objects(self.left_camera, self.right_camera, "cameras_are_coupled",
                                                [
                                ("cameraPresetChanged", "setCameraPreset"),
                                ("maxZoomLevelChanged", "maxZoomLevelChanged"),
                                ("targetChanged", "setTarget"),
                                ("wrapXRotationChanged", "setWrapXRotation"),
                                ("wrapYRotationChanged", "setWrapYRotation"),
                                ("xRotationChanged", "setXRotation"),
                                ("yRotationChanged", "setYRotation"),
                                ("zoomLevelChanged", "setZoomLevel"),
                            ])

        self.couple_tfs(self.left_transfer_function, self.right_transfer_function)

        ## time_series containers
        self.time_series_view_widget_1 = TimeSeriesViewWidget(self.data)
        self.time_series_view_widget_2 = TimeSeriesViewWidget(self.data)

        self.init_time_series_radio_check_buttons_and_description(
            self.time_series_view_widget_1, self.time_series_view_widget_2)

        self.ui.timeSeriesViewContainer_1.addWidget(self.time_series_view_widget_1)
        self.ui.timeSeriesViewContainer_2.addWidget(self.time_series_view_widget_2)

        # init check radio button
        self._button_group1.buttons()[0].click()
        self._button_group2.buttons()[1].click()

        # connect time_series_view_mouse_move to patch_hovered
        self.couple_time_series_view_with_patch_hovered(self.time_series_view_widget_1, self.time_series_view_widget_2)

        # connect resetting views button
        self.ui.resetViewsButton.clicked.connect(self.resetViews)

        # connect zooming to boxes buttons
        self.connect_zooming_to_boxes_buttons()

        # self.ui.proj_alg_tb.clicked.connect(self.analysis_widget.set_proj_alg)
        # self.ui.proj_temp_size_tb.clicked.connect(self.analysis_widget.set_proj_temp_size)
        # self.ui.proj_dim_tb.clicked.connect(self.analysis_widget.set_proj_dim)
        # self.ui.proj_red_tb.clicked.connect(self.analysis_widget.set_red_mode)
        # self.ui.proj_dis_metric_tb.clicked.connect(self.analysis_widget.set_proj_dis_metric)
        # self.ui.proj_part_strat_tb.clicked.connect(self.analysis_widget.set_proj_part_strat)

        # connect use via seg map thingy button
        self.ui.compareAllViaSegMap.clicked.connect(lambda: self.projection_widget.set_compare_all_via_seg_map(
            self.ui.compareAllViaSegMap.isChecked()))

        # connect showing spline series or not
        self.ui.showSplinesBetweenTimeSteps.clicked.connect(lambda: self.projection_widget.set_show_spline_series(
            self.ui.showSplinesBetweenTimeSteps.isChecked()))

        # set a time range # NOTE: currently disabled since not implemented in projection widget
        # by default hide the widget and disable button # enabling it is handled when setting the reduction mode
        self.ui.selectedTimeRangeWidget.setVisible(False)
        self.ui.selectedTimeRangeWidget.setEnabled(False)

        # connect apply range and apply reset button
        self.ui.selectedTimeRangeButton.clicked.connect(lambda: self.projection_widget.set_patch_range(
            self.time_series_view_widget_1.patch_start,
            self.time_series_view_widget_1.patch_end
        ))
        self.ui.selectedTimeRangeResetButton.clicked.connect(lambda: self.projection_widget.set_patch_range(
            None, None
        ))

        # seg map threshold
        self.ui.segThresholdContainerWidget.setVisible(True)
        # set min max values of spin box
        step_size_spin_box = self.ui.saturationSegThresholdSpinBox.singleStep()

        minmin = 0.001

        self.ui.saturationSegThresholdSpinBox.setRange(minmin, self.max_at_p_percent[0] - step_size_spin_box)
        self.ui.concentrationSegThresholdSpinBox.setRange(minmin, self.max_at_p_percent[1] - step_size_spin_box)

        # connect apply and reset
        self.ui.applySegThreshold.clicked.connect(lambda: self.projection_widget.set_segmentation_threshold(
            [self.ui.saturationSegThresholdSpinBox.value(), self.ui.concentrationSegThresholdSpinBox.value()]
        ))
        self.ui.resetSegThreshold.clicked.connect(lambda: self.projection_widget.set_segmentation_threshold([self.ui.saturationSegThresholdSpinBox.minimum(), self.ui.concentrationSegThresholdSpinBox.minimum()]))

        # connect screenshot projection action
        self.ui.actionScreenshot.triggered.connect(lambda e: self.projection_widget.save_projection_as_screenshot(None))

        # connect screenshot volume views
        self.ui.actionScreenshot_Volume_Views.triggered.connect(lambda e: self.save_volume_views_as_screenshots(None))

        # connect time series screenshot
        self.ui.actionScreenshot_Time_Series.triggered.connect(lambda e: self.save_time_series_as_screenshot(None))

        # connect screenshot controls
        self.ui.actionScreenshot_Controls.triggered.connect(lambda e: self.save_screenshot_controls(None))

        # connect screenshot all individually
        self.ui.actionScreenshot_All_Individually.triggered.connect(lambda e: self.save_screenshot_all_individually(None))

        # connect screenshot all as one image
        self.ui.actionScreenshot_All_One_Image.triggered.connect(lambda e: self.save_screenshot_all_as_one_image(None))

        # connect run all buttons
        self.ui.actionRun_All.triggered.connect(lambda e: self.run_all_and_save_screenshots(False))
        self.ui.actionRun_All_with_Screenshots.triggered.connect(lambda e: self.run_all_and_save_screenshots(True))

        # connect show labels checkbox
        self.ui.showLabelsCheckbox.clicked.connect(lambda e: self.projection_widget.set_show_labels(self.ui.showLabelsCheckbox.isChecked()))

        # connect use fast dtw reduction checkbox
        self.ui.useFastDTWCheckbox.clicked.connect(lambda e: self.projection_widget.set_use_fastdtw_reduction(self.ui.useFastDTWCheckbox.isChecked()))

    def run_all_and_save_screenshots(self, with_screenshot: bool):
        """
        Runs all projection control combinations and saves screenshots for each of them.
        :return:
        """
        print(f"run all; with screenshots: {with_screenshot}")

        # get available attributes
        proj_alg = get_attributes(ProjectionWidget.ProjAlgorithm)
        proj_red = get_attributes(ProjectionWidget.DataReductionMode)
        proj_dim = get_attributes(ProjectionWidget.ProjectionDim)
        proj_met = get_attributes(ProjectionWidget.DissimilarityMeasure)
        proj_dat = get_attributes(ProjectionWidget.DataMode)

        proj_scp = [True, False]
        proj_cvs = [True, False]

        total_runs = 0

        self.projection_widget._is_full_run = True

        if not self.ui.showLabelsCheckbox.isChecked():
            self.ui.showLabelsCheckbox.click()

        always_enable_experimental = False

        try:
            # spatial data
            screenshots_per_configuration = 6
            randints = [random.randint(0, screenshot_randint_max) for i in range(screenshots_per_configuration)]
            sd_proj_dat = list(filter(lambda pd: "Time" not in pd, proj_dat))
            for palg in proj_alg:
                palg_action = list(filter(lambda action: action.text() == palg, self.ui.proj_alg_tb.actions()))[0]
                palg_action.trigger()
                QApplication.processEvents()
                for pred in proj_red:
                    a = list(filter(lambda action: action.text() == pred, self.ui.proj_red_tb.actions()))[0]
                    a.trigger()
                    QApplication.processEvents()
                    for pdim in proj_dim:
                        a = list(filter(lambda action: action.text() == pdim, self.ui.proj_dim_tb.actions()))[0]
                        a.trigger()
                        QApplication.processEvents()
                        for pmet in proj_met:
                            a = list(filter(lambda action: action.text() == pmet, self.ui.proj_dis_metric_tb.actions()))[0]
                            if always_enable_experimental and "S4" in pmet:
                                continue
                            a.trigger()
                            QApplication.processEvents()
                            for pdat in sd_proj_dat:
                                a = list(filter(lambda action: action.text() == pdat, self.ui.proj_data.actions()))[0]
                                a.trigger()
                                QApplication.processEvents()
                                # time.sleep(0.15)
                                for pcvs in proj_cvs:
                                    self.ui.compareAllViaSegMap.click()
                                    QApplication.processEvents()
                                    for pscp in proj_scp:
                                        if pred == self.projection_widget.DataReductionMode.PATCH:
                                            self.ui.showSplinesBetweenTimeSteps.click()  # click and continue with screenshot
                                        else:
                                            # it is group -> only click if it is enabled (to disable)
                                            if self.ui.showSplinesBetweenTimeSteps.isChecked():
                                                self.ui.showSplinesBetweenTimeSteps.click()  # if group and checked, click to uncheck
                                            else:
                                                # if group and not checked, continue with screenshot
                                                pass

                                        QApplication.processEvents()

                                        if always_enable_experimental:
                                            # always enable experimental groups if possible
                                            buttons = self._button_group4.buttons()
                                            experimental_buttons = [b for b in buttons if b.text().startswith("experiment")]
                                            experimental_enabled = experimental_buttons[0].isEnabled()

                                            if experimental_enabled:
                                                for exp_b in experimental_buttons:
                                                    if not exp_b.isChecked():
                                                        exp_b.click()
                                                        QApplication.processEvents()

                                        print("screenshot settings: {}".format((palg, pred, pdim, pmet, pdat, self.ui.showSplinesBetweenTimeSteps.isChecked(), self.ui.showSplinesBetweenTimeSteps.isChecked())))
                                        print(f"making screenshot for run w.r.t. spatial data metrics: {total_runs}")
                                        total_runs += 1

                                        # if total_runs > 30:
                                        #     print("early debug stop")
                                        #     return

                                        if with_screenshot:
                                            self.projection_widget.save_projection_as_screenshot(randints[0])
                                            for randint in randints[1:]:
                                                palg_action.trigger()
                                                QApplication.processEvents()
                                                self.projection_widget.save_projection_as_screenshot(randint)

                                        # if it was group and not showing splines: continue, don't need to do group with splines
                                        if pred == self.projection_widget.DataReductionMode.GROUP and self.ui.showSplinesBetweenTimeSteps.isChecked():
                                            continue

            # time series data
            td_proj_dat = list(filter(lambda pd: "Time" in pd, proj_dat))
            for palg in proj_alg:
                palg_action = list(filter(lambda action: action.text() == palg, self.ui.proj_alg_tb.actions()))[0]
                palg_action.trigger()
                QApplication.processEvents()
                for pdat in td_proj_dat:
                    a = list(filter(lambda action: action.text() == pdat, self.ui.proj_data.actions()))[0]
                    a.trigger()
                    QApplication.processEvents()
                    for pdim in proj_dim:
                        a = list(filter(lambda action: action.text() == pdim, self.ui.proj_dim_tb.actions()))[0]
                        a.trigger()
                        QApplication.processEvents()

                        print(f"making screenshot for run w.r.t. time series metrics: {total_runs}")
                        total_runs += 1

                        if with_screenshot:
                            self.projection_widget.save_projection_as_screenshot(randints[0])
                            for randint in randints[1:]:
                                palg_action.trigger()
                                QApplication.processEvents()
                                self.projection_widget.save_projection_as_screenshot(randint)

            print(f"finished total runs: {total_runs}")
        finally:
            self.projection_widget._is_full_run = False

    def save_screenshot_all_as_one_image(self, randint=None):
        if randint is None:
            randint = random.randint(0, screenshot_randint_max)

        pixmap = self.screen().grabWindow(0, self.x(), self.y(), self.width(), self.height())
        pixmap.save(f"{screenshot_locations}/vamled_{randint}.png")

    def save_screenshot_all_individually(self, randint=None):
        if randint is None:
            randint = random.randint(0, screenshot_randint_max)

        self.save_screenshot_controls(randint)
        self.save_time_series_as_screenshot(randint)
        self.save_volume_views_as_screenshots(randint)
        self.projection_widget.save_projection_as_screenshot(randint)

    def save_screenshot_controls(self, randint=None):
        cwc = self.ui.controlWidgetContainer

        top_left = cwc.mapToGlobal(QPoint(0, 0))

        pixmap = cwc.screen().grabWindow(0, top_left.x(), top_left.y(), cwc.width(), cwc.height())

        if randint is None:
            randint = random.randint(0, screenshot_randint_max)

        pixmap.save(f"{screenshot_locations}/controls_{randint}.png")

    def save_time_series_as_screenshot(self, randint=None):
        if randint is None:
            randint = random.randint(0, screenshot_randint_max)

        for tsvw in [self.time_series_view_widget_1, self.time_series_view_widget_2]:
            tsvw.save_screenshot(randint)

    def save_volume_views_as_screenshots(self, randint=None):
        if randint is None:
            randint = random.randint(0, screenshot_randint_max)

        for ui in [self.leftUi, self.rightUi]:
            vr = ui.volume_renderer
            vr.save_screen_shot(group=ui.selectDataLabel.text(), sat_or_con=ui.switchSatCon.text(), rand_int=randint)

    def connect_zooming_to_boxes_buttons(self):
        """
                x,  y,  w,  h
        Box A:  11, 0,  17, 6
        Box B:  0,  6,  11, 6
        Box C:  11, 1,  15, 3

        Note: multiply x,w-values by 5; y,h-values by 4 for correct ranges.
        :return:
        """
        def compute_mxw_myh(v: VolumeRenderGraph):
            """
            total boxes in models are w: 28, h: 15
            :param v:
            :return:
            """
            m_xw = v.actual_data_shape[2] / 28
            m_yh = v.actual_data_shape[1] / 15

            return m_xw, m_yh

        # m_xw = 10
        # m_yh = 8

        def zoom_to_box_a(v: VolumeRenderGraph, ui):
            m_xw, m_yh = compute_mxw_myh(v)

            v.axisX().setRange(11 * m_xw, (11 + 17) * m_xw)
            v.axisY().setRange(0 * m_yh, (0 + 6) * m_yh)

            ui.currentZoom.setText("Box A")

        def zoom_to_box_b(v: VolumeRenderGraph, ui):
            m_xw, m_yh = compute_mxw_myh(v)

            v.axisX().setRange(0 * m_xw, (0 + 11) * m_xw)
            v.axisY().setRange(6 * m_yh, (6 + 6) * m_yh)

            ui.currentZoom.setText("Box B")

        def zoom_to_box_c(v: VolumeRenderGraph, ui):
            m_xw, m_yh = compute_mxw_myh(v)

            v.axisX().setRange(11 * m_xw, (11 + 15) * m_xw)
            v.axisY().setRange(1 * m_yh, (1 + 3) * m_yh)

            ui.currentZoom.setText("Box C")

        def make_to_zoom_fn(view: VolumeRenderGraph, ui, fn):
            return lambda: fn(view, ui)

        for (ui, v_view) in [(self.leftUi, self.leftUi.volume_renderer), (self.rightUi, self.rightUi.volume_renderer)]:
            self.ui.zoomToBoxAButton.clicked.connect(make_to_zoom_fn(v_view, ui, zoom_to_box_a))
            self.ui.zoomToBoxBButton.clicked.connect(make_to_zoom_fn(v_view, ui, zoom_to_box_b))
            self.ui.zoomToBoxCButton.clicked.connect(make_to_zoom_fn(v_view, ui, zoom_to_box_c))

    def resetViews(self):
        for ui, v_view in [(self.leftUi, self.leftUi.volume_renderer), (self.rightUi, self.rightUi.volume_renderer)]:
            v_view.reset_all_axis()
            ui.currentZoom.setText("Full Volume")

    def couple_time_series_view_with_patch_hovered(self, tv1: TimeSeriesViewWidget, tv2: TimeSeriesViewWidget):
        for tv in [tv1, tv2]:
            tv.patch_hovered_signal.connect(self.time_series_view_hovered)

    @Slot(int, int)
    def time_series_view_hovered(self, patch_start, patch_end, drag: bool):
        if not self.ui.time_view_control_checkbox.isChecked():
            return

        if not self.ui.time_view_hover_control_checkbox.isChecked() and not drag:
            return

        for_groups = [self.leftUi.selectDataLabel.text(), self.rightUi.selectDataLabel.text()]

        for g in for_groups:
            self.patch_hovered(g, patch_start, patch_end)
            if self.ui.coupleTimeStep.isChecked():
                break # only do this once if it is coupled anyways

    @staticmethod
    def _init_time_series_radio_check_buttons(view: TimeSeriesViewWidget, grid: QGridLayout, column: int,
                                              bg: QButtonGroup):
        """
        Init the buttons and connect their listeners to the view widget.
        :param view:
        :return:
        """
        def make_set_time_series_data(k):
            return lambda: view.set_time_series_data(k)

        for i, key in enumerate(time_series_namings_keys):
            rb = QRadioButton()
            rb.clicked.connect(make_set_time_series_data(key))
            grid.addWidget(rb, i+1, column)
            bg.addButton(rb)

    def init_time_series_radio_check_buttons_and_description(self, view_1: TimeSeriesViewWidget,
                                                             view_2: TimeSeriesViewWidget):
        """
        Init the buttons and connect their listeners to the view widgets.
        :param view_1:
        :param view_2:
        :return:
        """
        self._button_group1 = QButtonGroup(self.ui.tsd_grid_layout)
        self._init_time_series_radio_check_buttons(view_1, self.ui.tsd_grid_layout, 0, self._button_group1)

        self._button_group2 = QButtonGroup(self.ui.tsd_grid_layout)
        self._init_time_series_radio_check_buttons(view_2, self.ui.tsd_grid_layout, 1, self._button_group2)

        self._button_group3 = QButtonGroup(self.ui.tsd_grid_layout)
        self._button_group3.setExclusive(False)

        def make_set_enabled_time_series_as_feature(k):
            # TODO: stop setting checked if it is last one
            return lambda state: self.projection_widget.update_enabled_times_series(k, state == Qt.Checked)

        # add checkbuttons
        for i, key in enumerate(time_series_namings_keys):
            cb = QCheckBox()
            cb.setChecked(True)  # checked by default
            cb.stateChanged.connect(make_set_enabled_time_series_as_feature(key))
            self._button_group3.addButton(cb)
            self.ui.tsd_grid_layout.addWidget(cb, i+1, 3)

        # init description
        for i, key in enumerate(time_series_namings_keys):
            label = QLabel(time_series_namings[key])
            self.ui.tsd_grid_layout.addWidget(label, i+1, 2)

        def make_set_enabled_group_as_features(group):
            def fn(state):
                self.projection_widget.update_enabled_group(group, state == Qt.Checked)
                self.time_series_view_widget_1.update_enabled_group(group, state == Qt.Checked)
                self.time_series_view_widget_2.update_enabled_group(group, state == Qt.Checked)
                # TODO: stop setting checked if group is last one that is checked
            return fn

        # init checkbuttons for group selection
        self._button_group4 = QButtonGroup(self.ui.groupSelectionGridLayout)
        self._button_group4.setExclusive(False)
        for i, group in enumerate(sorted(self.data.keys())):
            if group not in groups:
                continue
            cb = QCheckBox(group)
            cb.setChecked(not group.startswith("experiment"))  # checked by default  # experiment groups are not enabled by default
            cb.stateChanged.connect(make_set_enabled_group_as_features(group))
            cb.setEnabled(not group.startswith("experiment") or self._allow_experimental_data_for_s4)
            self._button_group4.addButton(cb)
            self.ui.groupSelectionGridLayout.addWidget(cb, i+1, 0)

    def setup_transferfunctions(self, tf: qtpex.qt_widgets.transferfunction_widget.TransferFunctionWidget):
        """Add axis description to tf"""
        tf.chart().axisX().setTitleText("[Upper scale: Saturation; Lower scale: Concentration] -> Color".format(
            #self.min_[0], self.max_at_p_percent[0], self.min_[1], self.max_at_p_percent[1]
        ))
        tf.chart().axisY().setTitleText("Opacity")

        tf.chart().axisX().setVisible(False)
        tf.chart().axisY().setVisible(False)

        opacity_axis = QValueAxis()
        opacity_axis.setRange(0, 1)
        opacity_axis.setTitleText("Opacity")
        tf.opacity_axis = opacity_axis

        tf.chart().addAxis(tf.opacity_axis, Qt.AlignLeft)

        tf.chart().axisX().setLabelsVisible(True)

        saturation_axis = QValueAxis()
        saturation_axis.setRange(self.min_[0], self.max_at_p_percent[0])
        saturation_axis.setTickCount(9)
        saturation_axis.setTitleText("[Upper scale: Saturation; Lower scale: Concentration] -> Color")
        saturation_axis.setTitleVisible(False)

        concentration_axis = QValueAxis()
        concentration_axis.setRange(self.min_[1], self.max_at_p_percent[1])
        concentration_axis.setTickCount(9)
        concentration_axis.setTitleText("[Upper scale: Saturation; Lower scale: Concentration] -> Color")

        tf.sat_axis = saturation_axis
        tf.con_axis = concentration_axis

        tf.chart().addAxis(saturation_axis, Qt.AlignBottom)
        tf.chart().addAxis(concentration_axis, Qt.AlignBottom)

        concentration_axis.setVisible(True)
        saturation_axis.setVisible(True)

    @property
    def tfs_are_coupled(self) -> bool:
        return self.ui.coupleTF.isChecked()

    @staticmethod
    def _load_transferfunction_from_file(filename, transferfunction_widget):
        if filename == "" or not os.path.exists(filename) or os.path.isdir(filename):
            print(f"filename is empty or does not exist or is a directory: '{filename}")
            return

        with open(filename, "rt") as f:
            content = f.read()
        # print(content)
        transferfunction_widget.tf_from_json(content)

    def load_transferfunction(self):
        filename = QFileDialog.getOpenFileName(
            self, "Load transferfunction", os.path.abspath("./transferfunctions"), "JSON Files (*.json)")

        # print(filename)

        filename = filename[0]
        tfw = self.left_transfer_function if self.is_left_tab(self.sender()) else self.right_transfer_function

        self._load_transferfunction_from_file(filename, tfw)

    def save_transferfunction(self):
        if not os.path.exists(os.path.abspath("./transferfunctions")):
            os.mkdir(os.path.abspath("./transferfunctions"))

        filename = QFileDialog.getSaveFileName(
            self, "Save transferfunction", os.path.abspath("./transferfunctions"), "JSON Files (*.json)")

        # print(filename)

        filename = filename[0]

        if filename == "" or os.path.isdir(filename) or not os.path.exists(os.path.dirname(filename)):
            print(f"filename is empty, it's parent dir does not exist, or it is a directory: '{filename}")
            return

        tfw = self.left_transfer_function if self.is_left_tab(self.sender()) else self.right_transfer_function

        with open(filename, "wt") as f:
            tfj = tfw.tf_as_json()
            # print("d", tfj)
            json.dump(json.loads(tfj), f)

    def couple_tfs(self, left_tf: TransferFunctionWidget, right_tf: TransferFunctionWidget):
        def block_signals_wrapper(obj, fn, *args, **kwargs):
            obj.blockSignals(True)
            res = fn(*args, **kwargs)
            obj.blockSignals(False)
            return res

        def check_coupled_and_prevent_loop_wrapper(obj, fn):
            def check_coupled(*args, **kwargs):
                if not self.tfs_are_coupled:
                    return
                else:
                    return block_signals_wrapper(obj, fn, *args, **kwargs)

            return check_coupled

        def handle_changed(src, tgt, vr):
            def copy_tf():
                tgt.copy_from_other_tf_widget(src)
                self.update_color_table(tgt, vr)  # need to call explicitly since we disabled the changed signal

            return copy_tf

        for (source, target, vr) in [(left_tf, right_tf, self.rightUi.volume_renderer),
                                 (right_tf, left_tf, self.leftUi.volume_renderer)]:
            source.changed_signal.connect(check_coupled_and_prevent_loop_wrapper(target, handle_changed(source, target, vr)))

    @property
    def cameras_are_coupled(self) -> bool:
        return self.ui.coupleCameras.isChecked()

    def couple_signals_to_slots_of_objects(self, c1, c2, prop_to_check: str, signals_to_slots: list):
        def block_signals_wrapper(obj, fn, *args, **kwargs):
            obj.blockSignals(True)
            res = fn(*args, **kwargs)
            obj.blockSignals(False)
            return res

        def check_coupled_and_prevent_loop_wrapper(obj, fn):
            def check_coupled(*args, **kwargs):
                if not getattr(self, prop_to_check):
                    return
                else:
                    return block_signals_wrapper(obj, fn, *args, **kwargs)

            return check_coupled

        for (source, target) in [(c1, c2), (c2, c1)]:
            for signal, slot in signals_to_slots:
                getattr(source, signal).connect(check_coupled_and_prevent_loop_wrapper(target, getattr(target, slot)))

            # source.cameraPresetChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setCameraPreset))
            # source.maxZoomLevelChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.maxZoomLevelChanged))
            # source.targetChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setTarget))
            # source.wrapXRotationChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setWrapXRotation))
            # source.wrapYRotationChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setWrapYRotation))
            # source.xRotationChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setXRotation))
            # source.yRotationChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setYRotation))
            # source.zoomLevelChanged.connect(check_coupled_and_prevent_loop_wrapper(target, target.setZoomLevel))

    def init_projection_control_tool_buttons(self):
        self.init_projection_control_tool_button_actions(self.ui.proj_alg_tb,
                                                         get_attributes(ProjectionWidget.ProjAlgorithm),
                                                         self.projection_widget.set_proj_alg)
        self.ui.proj_alg_tb.setText(self.projection_widget.proj_alg)

        # self.init_projection_control_tool_button_actions(self.ui.proj_temp_size_tb,
        #                                                  get_attributes(ProjectionWidget.TemporalPatchSize),
        #                                                  self.projection_widget.set_proj_temp_size)
        # self.ui.proj_temp_size_tb.setText(self.projection_widget.temp_p_size)

        self.init_projection_control_tool_button_actions(self.ui.proj_dim_tb,
                                                         get_attributes(ProjectionWidget.ProjectionDim),
                                                         self.projection_widget.set_proj_dim)
        self.ui.proj_dim_tb.setText(self.projection_widget.proj_dim)

        self.init_projection_control_tool_button_actions(self.ui.proj_red_tb,
                                                         get_attributes(ProjectionWidget.DataReductionMode),
                                                         self.handle_proj_red_mode)
        self.ui.proj_red_tb.setText(self.projection_widget.data_red_mode)

        self.init_projection_control_tool_button_actions(self.ui.proj_dis_metric_tb,
                                                         get_attributes(ProjectionWidget.DissimilarityMeasure),
                                                         self.handle_dis_metric_mode)
        self.ui.proj_dis_metric_tb.setText(self.projection_widget.dissim_measure)

        # self.init_projection_control_tool_button_actions(self.ui.proj_part_strat_tb,
        #                                                  get_attributes(ProjectionWidget.PartitionStrategy),
        #                                                  self.projection_widget.set_proj_part_strat)
        # self.ui.proj_part_strat_tb.setText(self.projection_widget.part_strat)

        self.init_projection_control_tool_button_actions(self.ui.proj_data,
                                                         get_attributes(ProjectionWidget.DataMode),
                                                         self.handle_time_series_data_mode)
        self.ui.proj_data.setText(self.projection_widget.data_mode)

        # Hide projection strat. and temp. size button
        # self.ui.proj_part_strat_tb.setVisible(False)
        # self.ui.proj_part_strat_label.setVisible(False)
#
        # self.ui.proj_temp_size_tb.setVisible(False)
        # self.ui.proj_temp_size_label.setVisible(False)

    @Slot(str, int, int, str)
    def group_clicked(self, group: str, time_step_from: int, time_step_to: int, btn: str):
        ui = self.leftUi if btn != "right" else self.rightUi
        self.set_fluidflower_data(group, ui, False)

        ui.volume_renderer.axisZ().setRange(time_step_from, time_step_to)

        for tv in [self.time_series_view_widget_1, self.time_series_view_widget_2]:
            tv.update_time_frame_sliders(time_step_from, time_step_to)

    @Slot(str, int, int)
    def patch_hovered(self, group: str, time_step_from: int, time_step_to: int):
        uis_to_use = []

        # only shows patch if it's dataset is current in the ui
        if self.leftUi.selectDataLabel.text() == group:
            uis_to_use.append(self.leftUi)
        elif self.rightUi.selectDataLabel.text() == group:
            uis_to_use.append(self.rightUi)
        else:
            return

        if self.ui.coupleTimeStep.isChecked():
            uis_to_use = [self.leftUi, self.rightUi]  # override uis to use with both

        # print(time_step_from, time_step_to)

        for ui in uis_to_use:
            ui.volume_renderer.axisZ().setRange(time_step_from, time_step_to)


        for tv in [self.time_series_view_widget_1, self.time_series_view_widget_2]:
            tv.update_time_frame_sliders(time_step_from, time_step_to)

        # select patch in view
        for halo_series in self.projection_widget.halo_serieses.values():
            halo_series.deselectAllPoints()
        for series in self.projection_widget.serieses.values():
            series.deselectAllPoints()
        groups_ = [ui.selectDataLabel.text() for ui in uis_to_use if ui.selectDataLabel.text() != "Default"]
        self.projection_widget.select_group_or_patch(groups_, time_step_from, time_step_to)

    @staticmethod
    def update_color_table(transferfunction, volume_renderer):
        if volume_renderer.volume_item is None:
            return
        volume_renderer.volume_item.setColorTable([c.rgba() for c in transferfunction.get_current_color_map()])

    def is_left_ui(self, w: QWidget):
        """
        Returns if w is a child of the left view container or not.
        :param w:
        :return: bool
        """
        return self.ui.leftViewContainer.isAncestorOf(w)

    def is_left_tab(self, w: QWidget):
        """
        Returns if w is a child of the left tab container or not.
        :param w:
        :return: bool
        """
        return self.ui.tab_left.isAncestorOf(w)

    def toggle_saturation_concentration(self):
        """
        Toggles the calling button to switch between Saturation and Concentration.
        :return:
        """
        if self.sender().text() == "Saturation":
            self.sender().setText("Concentration")
        else:
            self.sender().setText("Saturation")

        if self.ui.leftViewContainer.isAncestorOf(self.sender()):
            self.set_fluidflower_data(self.leftUi.selectDataLabel.text(), self.leftUi, True)
        else:
            self.set_fluidflower_data(self.rightUi.selectDataLabel.text(), self.rightUi, True)

    def scale_volume(self, x: np.array):
        # min_at_1_minus_p_percent = [min([np.sort(self.data[group]["sat_con"][:, :, :, i].flatten())
        #                         [int((1-p) * len(self.data[group]["sat_con"][:, :, :, i].flatten()))]
        #                         for group in self.data]) for i in range(2)]

        min_d = self.min_
        max_d = self.max_at_p_percent  # [max([self.data[group]["sat_con"][:, :, :, i].max() for group in self.data]) for i in range(2)]

        x_ = interp_variables(x.copy(), xp=lambda _, i: (min_d[i], max_d[i]), fp=lambda _, i: (0, 255))  # [:144]

        # scale with the same conditions
        # x_ = interp_variables(x.copy(), xp=lambda _, i: (min(min_d), max(max_d)), fp=lambda _, i: (0, 255))

        # clip to range
        return np.clip(x_, 0, 255)

    def set_fluidflower_data(self, group: str, ui: Ui_VolumeViewer, was_switched: bool = False):
        """
        Sets the fluidflower data to the correct volume widget depending on leftTrueRightFalse.
        Sets by default saturation.
        :param ui: the left view widget ui or the right view widget ui
        :param was_switched:
        :param group:
        :return:
        """
        label = ui.selectDataLabel

        # if group is already selected: do nothing
        if (label.text() == group and not was_switched) or group == "Default":
            return

        label.setText(group)

        vi = ui.volume_renderer  # is added in customize_ui

        volume_item_cache = ui.volume_item_cache  # is added in customize_ui

        # if a volume item is already set, save the current range to apply the same range to it after setting it
        if vi.volume_item is not None:
            set_range = True
            x_range = [vi.axisX().min(), vi.axisX().max()]
            y_range = [vi.axisY().min(), vi.axisY().max()]
            z_range = [vi.axisZ().min(), vi.axisZ().max()]
        else:
            set_range = False

        if group in volume_item_cache and ui.switchSatCon.text() in volume_item_cache[group]:
            # use cache
            vi.set_volume_item(*volume_item_cache[group][ui.switchSatCon.text()])
        else:
            dl, dl_shape = resize_to_power_of_two_and_scale_to_uint8(
                self.scale_volume(self.data[group]["sat_con"])[:, :, :, 0 if ui.switchSatCon.text() == "Saturation" else 1],
                False)
            v_item = make_custom_3d_volume_item_from_data(dl.astype(int))
            vi.set_volume_item(v_item, dl_shape)

            if group not in volume_item_cache:
                volume_item_cache[group] = dict()

            volume_item_cache[group][ui.switchSatCon.text()] = (v_item, dl_shape)

        if set_range:
            vi.axisX().setRange(*x_range)
            vi.axisY().setRange(*y_range)
            vi.axisZ().setRange(*z_range)

        self.update_color_table(self.right_transfer_function, self.rightUi.volume_renderer)
        self.update_color_table(self.left_transfer_function, self.leftUi.volume_renderer)

    @staticmethod
    def init_projection_control_tool_button_actions(tb: QToolButton, options: [str], fn):
        def make_parameter_fn(tb_: QToolButton, s: str):
            def _make_paramter_fn():
                tb_.setText(s)
                return fn(s)

            return _make_paramter_fn

        for o in options:
            tb.addAction(o, make_parameter_fn(tb, o))

    def handle_proj_red_mode(self, proj_red_mode: str):
        if proj_red_mode == self.projection_widget.DataReductionMode.GROUP:
            # hide button
            self.ui.selectedTimeRangeWidget.setVisible(False)
            self.ui.selectedTimeRangeButton.setEnabled(False)
        else:
            # show button # NOTE: currently disabled since it is not implemented in proj. widget
            # self.ui.selectedTimeRangeWidget.setVisible(True)
            # self.ui.selectedTimeRangeWidget.setEnabled(True)
            pass
        return self.projection_widget.set_red_mode(proj_red_mode)

    def handle_dis_metric_mode(self, dis_metric: str):
        # disable experimental groups for S4 metrics, enable for others
        enable_exp = not ((not self._allow_experimental_data_for_s4 and "S4" in dis_metric) or "Time" in self.projection_widget.data_mode)

        experimental_buttons = [btn for btn in self._button_group4.buttons() if "experiment" in btn.text()]

        # enable / disable buttons
        for btn in experimental_buttons:
            # set checked state but block signals to prevent recursion
            btn.setEnabled(enable_exp)

        # force uncheck on disable
        if not enable_exp:
            for btn in experimental_buttons:
                old_signal = btn.signalsBlocked()
                btn.blockSignals(True)
                btn.setChecked(enable_exp)
                btn.blockSignals(old_signal)

            self.projection_widget.update_enabled_groups([btn.text() for btn in experimental_buttons], False)

        return self.projection_widget.set_proj_dis_metric(dis_metric)

    def handle_time_series_data_mode(self, data_mode):
        tsd_incompatible_buttons = [self.ui.proj_red_tb, #self.ui.proj_temp_size_tb, self.ui.proj_part_strat_tb,
                                    self.ui.proj_dis_metric_tb]
        tsd_incompatible_labels = [self.ui.proj_red_label, #self.ui.proj_temp_size_label, self.ui.proj_part_strat_label,
                                   self.ui.proj_dis_metric_label]

        b = not ("Time" in data_mode)
        for btn in tsd_incompatible_buttons:
            btn.setVisible(b)
        for label in tsd_incompatible_labels:
            label.setVisible(b)

        b_exp = not ((not self._allow_experimental_data_for_s4 and "S4" in self.projection_widget.dissim_measure) or "Time" in data_mode)

        # disable experimental data if time series
        experimental_buttons = [btn for btn in self._button_group4.buttons() if "experiment" in btn.text()]

        # enable / disable buttons
        for btn in experimental_buttons:
            btn.setEnabled(b_exp)

        # force uncheck on disable
        if not b_exp:
            for btn in experimental_buttons:
                old_signal = btn.signalsBlocked()
                btn.blockSignals(True)
                btn.setChecked(b_exp)
                btn.blockSignals(old_signal)

            self.projection_widget.update_enabled_groups([btn.text() for btn in experimental_buttons], False)

        return self.projection_widget.set_selected_data_mode(data_mode)

    @Slot()
    def update_load_data_tool_button_actions(self):
        is_left_ui = self.is_left_ui(self.sender())
        ui = self.leftUi if is_left_ui else self.rightUi

        tool_button = ui.loadDataButton

        # clear actions
        [tool_button.removeAction(a) for a in tool_button.actions().copy()]  # need copy to not modify list

        def make_action_fn(grp):
            return lambda: self.set_fluidflower_data(grp, ui)

        label = ui.switchSatCon

        for group in self.data:
            if label.text():
                tool_button.addAction(group, make_action_fn(group))
