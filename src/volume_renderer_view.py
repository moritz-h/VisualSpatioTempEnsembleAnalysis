import random

import PySide6
from PySide6.QtDataVisualization import *
from PySide6.QtGui import (QGuiApplication, QQuaternion, QVector3D, QOpenGLContext, QFont)
from PySide6.QtWidgets import QMessageBox


# based on Qt data vis examples: https://code.qt.io/cgit/qt/qtdatavis3d.git/tree/examples/datavisualization/volumetric/
class VolumeRenderGraph(Q3DScatter):  # inheriting from Q3DScatter makes it a graph
    def __init__(self):
        super().__init__()

        self.volume_item = None
        self.actual_data_shape = (0, 0, 0)

        if not self.hasContext():
            msg_box = QMessageBox()
            msg_box.setText("Failed to initialize the OpenGL context.")
            msg_box.exec()
            exit(-1)

        # init graph
        self.activeTheme().setType(Q3DTheme.Theme.ThemeQt)
        self.setShadowQuality(QAbstract3DGraph.ShadowQuality.ShadowQualityNone)
        self.scene().activeCamera().setCameraPreset(Q3DCamera.CameraPreset.CameraPresetFront)

        self.setOrthoProjection(True)

        self.activeTheme().setBackgroundEnabled(False)

        # Only allow zooming at the center and limit the zoom to 200% to avoid clipping issues
        Q3DInputHandler(self.activeInputHandler()).setZoomAtTargetEnabled(False)
        self.scene().activeCamera().setMaxZoomLevel(180)

        # set titles
        for axis, title in [(self.axisX(), "width"), (self.axisY(), "height"), (self.axisZ(), "time")]:
            axis.setTitle(title)
            axis.setTitleVisible(True)

        # debug: only draw part of volume

        if QOpenGLContext.currentContext().isOpenGLES():
            # OpenGL ES2 doesn't support 3D textures, so show a warning label instead
            warning_label = QCustom3DLabel("QCustom3DVolume is not supported with OpenGL ES2",
                                           QFont(),
                                           QVector3D(0, 0.5, 0),
                                           QVector3D(1.5, 1.5, 0),
                                           QQuaternion())
            warning_label.setPositionAbsolute(True)
            warning_label.setFacingCamera(True)
            self.addCustomItem(warning_label)
        else:
            # Display empty data
            self.no_data_label = QCustom3DLabel("Select data to display",
                                           QFont(),
                                           QVector3D(0, 0.5, 0),
                                           QVector3D(1.5, 1.5, 0),
                                           QQuaternion())
            self.no_data_label.setPositionAbsolute(True)
            self.no_data_label.setFacingCamera(True)
            self.addCustomItem(self.no_data_label)

        # connect zooming event to adjust segment count:
        self.axisX().rangeChanged.connect(lambda: self.axis_xy_changed(self.axisX()))
        self.axisY().rangeChanged.connect(lambda: self.axis_xy_changed(self.axisY()))
        self.axisZ().rangeChanged.connect(lambda: self.axis_z_changed(self.axisZ()))

    def axis_z_changed(self, axis: QValue3DAxis):
        # adjust segment count to have one segment per 20 min time step
        # range
        r = [axis.min(), axis.max()]

        # length
        lgth = r[1] - r[0]

        # compute segment count
        sc = max(1, round(lgth / 12))  # each step is 10 min -> we want segments of approx. size 2h / 120min but minimum 1 segment

        # set subsegment count if necessary
        if sc == 1:
            ssc = max(1, round(lgth / 4))  # subsegments of 30 min if less than 1 hour is looked at
        else:
            ssc = 1

        axis.setSegmentCount(sc)
        axis.setSubSegmentCount(ssc)
        # print("scz:", sc)

    def axis_xy_changed(self, axis: QValue3DAxis):
        # adjust segment count to have one segment per 10 cm
        # range
        r = [axis.min(), axis.max()]

        # length
        lgth = r[1] - r[0]

        # compute segment count
        sc = round(lgth / 10)  # each step is 1cm -> we want segments of size 10cm

        if sc < 5:
            sc = max(1, round(lgth / 5))  # each step is 5cm

        if sc == 1:
            ssc = 2
        else:
            ssc = 1

        axis.setSegmentCount(sc)
        axis.setSubSegmentCount(ssc)
        # print("scxy:", sc)

    def set_volume_item(self, volume_item: QCustom3DVolume, actual_data_shape: tuple):
        # first make previous custom item invisible (removing it would destroy its resources)
        if self.volume_item is not None:
            self.volume_item.setVisible(False)
        else:
            self.reset_all_axis()

        self.actual_data_shape = actual_data_shape

        self.toggle_area_all(True, (volume_item.textureDepth(), volume_item.textureHeight(), volume_item.textureWidth()))
        self.volume_item = volume_item
        self.volume_item.setScaling(QVector3D(
            self.axisX().max() - self.axisX().min(),
            (self.axisY().max() - self.axisY().min()) * 0.91,
            self.axisZ().max() - self.axisZ().min()
        ))
        self.volume_item.setPosition(QVector3D(
            (self.axisX().max() + self.axisX().min()) / 2,
            -0.045 * (self.axisY().max() - self.axisY().min()) +
            (self.axisY().max() + self.axisY().min()) / 2.0,
            (self.axisZ().max() + self.axisZ().min()) / 2.0
        ))
        self.volume_item.setScalingAbsolute(False)

        # data = data[:, ::-1, :]
        # print(data.shape, "set to texture: width {}, height {}, depth {}".format(*data.shape))
        # self.volume_item.setTextureDimensions(*data.shape)
        # print(data.shape, "set to texture: width {}, height {}, depth {}".format(data.shape[2], data.shape[1], data.shape[0]))
        # self.volume_item.setTextureDimensions(data.shape[2], data.shape[1], data.shape[0])

        # print(data.shape, "before flattening")
        # make it visible instead of adding it if it already exists (e.g., when used from some cache)
        if self.volume_item in self.customItems():
            self.volume_item.setVisible(True)
        else:
            self.addCustomItem(self.volume_item)

        self.toggle_area_all(True, actual_data_shape)

        if self.no_data_label in self.customItems():
            self.removeCustomItem(self.no_data_label)

    def toggle_area_all(self, enabled: bool, data_shape):
        """
        data is of shape: depth, height, width
        :param enabled:
        :param data_shape:
        :return:
        """
        if enabled:
            self.axisX().setRange(0, data_shape[2])
            self.axisY().setRange(0, data_shape[1])
            self.axisZ().setRange(0, data_shape[0])

    def reset_all_axis(self):
        self.toggle_area_all(True, self.actual_data_shape)
        print(self.actual_data_shape)

    def mousePressEvent(self, event:PySide6.QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)

    def save_screen_shot(self, group: str, sat_or_con: str, rand_int: int):
        image = self.renderToImage(msaaSamples=2)
        image.save(f"screenshots/{group}_{sat_or_con}_{rand_int}.png")
        del image
