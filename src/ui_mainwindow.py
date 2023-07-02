# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QMainWindow, QMenu,
    QMenuBar, QPushButton, QScrollArea, QSizePolicy,
    QStatusBar, QTabWidget, QToolButton, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1284, 1203)
        self.actionScreenshot = QAction(MainWindow)
        self.actionScreenshot.setObjectName(u"actionScreenshot")
        self.actionScreenshot_Volume_Views = QAction(MainWindow)
        self.actionScreenshot_Volume_Views.setObjectName(u"actionScreenshot_Volume_Views")
        self.actionScreenshot_Time_Series = QAction(MainWindow)
        self.actionScreenshot_Time_Series.setObjectName(u"actionScreenshot_Time_Series")
        self.actionScreenshot_Controls = QAction(MainWindow)
        self.actionScreenshot_Controls.setObjectName(u"actionScreenshot_Controls")
        self.actionScreenshot_All_Individually = QAction(MainWindow)
        self.actionScreenshot_All_Individually.setObjectName(u"actionScreenshot_All_Individually")
        self.actionScreenshot_All_One_Image = QAction(MainWindow)
        self.actionScreenshot_All_One_Image.setObjectName(u"actionScreenshot_All_One_Image")
        self.actionRun_All_with_Screenshots = QAction(MainWindow)
        self.actionRun_All_with_Screenshots.setObjectName(u"actionRun_All_with_Screenshots")
        self.actionRun_All = QAction(MainWindow)
        self.actionRun_All.setObjectName(u"actionRun_All")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_8 = QGridLayout()
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.gridLayout_8.setContentsMargins(0, -1, -1, -1)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout_8.addWidget(self.line, 0, 1, 1, 1)

        self.widget_2 = QWidget(self.centralwidget)
        self.widget_2.setObjectName(u"widget_2")
        self.gridLayout_5 = QGridLayout(self.widget_2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.leftViewContainer = QWidget(self.widget_2)
        self.leftViewContainer.setObjectName(u"leftViewContainer")

        self.horizontalLayout.addWidget(self.leftViewContainer)

        self.line_2 = QFrame(self.widget_2)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout.addWidget(self.line_2)

        self.rightViewContainer = QWidget(self.widget_2)
        self.rightViewContainer.setObjectName(u"rightViewContainer")

        self.horizontalLayout.addWidget(self.rightViewContainer)


        self.gridLayout_7.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.analysis_container = QGridLayout()
        self.analysis_container.setObjectName(u"analysis_container")
        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.timeSeriesViewContainer_1 = QGridLayout()
        self.timeSeriesViewContainer_1.setObjectName(u"timeSeriesViewContainer_1")

        self.gridLayout_9.addLayout(self.timeSeriesViewContainer_1, 1, 2, 1, 1)

        self.timeSeriesViewContainer_2 = QGridLayout()
        self.timeSeriesViewContainer_2.setObjectName(u"timeSeriesViewContainer_2")

        self.gridLayout_9.addLayout(self.timeSeriesViewContainer_2, 2, 2, 1, 1)

        self.projection_container = QGridLayout()
        self.projection_container.setObjectName(u"projection_container")

        self.gridLayout_9.addLayout(self.projection_container, 1, 1, 2, 1)

        self.gridLayout_9.setRowStretch(1, 1)
        self.gridLayout_9.setRowStretch(2, 1)
        self.gridLayout_9.setColumnStretch(1, 1)
        self.gridLayout_9.setColumnStretch(2, 1)

        self.analysis_container.addLayout(self.gridLayout_9, 0, 0, 1, 1)


        self.gridLayout_7.addLayout(self.analysis_container, 2, 0, 1, 1)

        self.line_3 = QFrame(self.widget_2)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.gridLayout_7.addWidget(self.line_3, 1, 0, 1, 1)

        self.gridLayout_7.setRowStretch(0, 1)
        self.gridLayout_7.setRowStretch(2, 2)

        self.gridLayout.addLayout(self.gridLayout_7, 0, 0, 1, 1)


        self.gridLayout_5.addLayout(self.gridLayout, 0, 0, 1, 1)


        self.gridLayout_8.addWidget(self.widget_2, 0, 0, 1, 1)

        self.controlWidgetContainer = QWidget(self.centralwidget)
        self.controlWidgetContainer.setObjectName(u"controlWidgetContainer")
        self.gridLayout_16 = QGridLayout(self.controlWidgetContainer)
        self.gridLayout_16.setSpacing(0)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.gridLayout_16.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox_3 = QGroupBox(self.controlWidgetContainer)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_3 = QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.coupleTF = QCheckBox(self.groupBox_3)
        self.coupleTF.setObjectName(u"coupleTF")
        self.coupleTF.setChecked(True)

        self.verticalLayout_3.addWidget(self.coupleTF)

        self.coupleCameras = QCheckBox(self.groupBox_3)
        self.coupleCameras.setObjectName(u"coupleCameras")
        self.coupleCameras.setChecked(True)

        self.verticalLayout_3.addWidget(self.coupleCameras)

        self.coupleTimeStep = QCheckBox(self.groupBox_3)
        self.coupleTimeStep.setObjectName(u"coupleTimeStep")
        self.coupleTimeStep.setChecked(True)

        self.verticalLayout_3.addWidget(self.coupleTimeStep)

        self.time_view_control_checkbox = QCheckBox(self.groupBox_3)
        self.time_view_control_checkbox.setObjectName(u"time_view_control_checkbox")
        self.time_view_control_checkbox.setChecked(True)

        self.verticalLayout_3.addWidget(self.time_view_control_checkbox)

        self.time_view_hover_control_checkbox = QCheckBox(self.groupBox_3)
        self.time_view_hover_control_checkbox.setObjectName(u"time_view_hover_control_checkbox")

        self.verticalLayout_3.addWidget(self.time_view_hover_control_checkbox)


        self.horizontalLayout_3.addLayout(self.verticalLayout_3)

        self.line_4 = QFrame(self.groupBox_3)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.VLine)
        self.line_4.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_3.addWidget(self.line_4)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.zoomToBoxAButton = QPushButton(self.groupBox_3)
        self.zoomToBoxAButton.setObjectName(u"zoomToBoxAButton")

        self.verticalLayout_2.addWidget(self.zoomToBoxAButton)

        self.zoomToBoxBButton = QPushButton(self.groupBox_3)
        self.zoomToBoxBButton.setObjectName(u"zoomToBoxBButton")

        self.verticalLayout_2.addWidget(self.zoomToBoxBButton)

        self.zoomToBoxCButton = QPushButton(self.groupBox_3)
        self.zoomToBoxCButton.setObjectName(u"zoomToBoxCButton")

        self.verticalLayout_2.addWidget(self.zoomToBoxCButton)

        self.resetViewsButton = QPushButton(self.groupBox_3)
        self.resetViewsButton.setObjectName(u"resetViewsButton")

        self.verticalLayout_2.addWidget(self.resetViewsButton)


        self.horizontalLayout_3.addLayout(self.verticalLayout_2)


        self.gridLayout_3.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.groupBox_2 = QGroupBox(self.controlWidgetContainer)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setMinimumSize(QSize(0, 300))
        self.gridLayout_6 = QGridLayout(self.groupBox_2)
        self.gridLayout_6.setSpacing(4)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(4, 2, 4, 4)
        self.tabWidget = QTabWidget(self.groupBox_2)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_left = QWidget()
        self.tab_left.setObjectName(u"tab_left")
        self.tabWidget.addTab(self.tab_left, "")
        self.tab_right = QWidget()
        self.tab_right.setObjectName(u"tab_right")
        self.tabWidget.addTab(self.tab_right, "")

        self.gridLayout_6.addWidget(self.tabWidget, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.groupBox = QGroupBox(self.controlWidgetContainer)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_4 = QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.selectedTimeRangeWidget = QWidget(self.groupBox)
        self.selectedTimeRangeWidget.setObjectName(u"selectedTimeRangeWidget")
        self.gridLayout_13 = QGridLayout(self.selectedTimeRangeWidget)
        self.gridLayout_13.setSpacing(0)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.gridLayout_13.setContentsMargins(0, 0, 0, 0)
        self.selectedTimeRangeHorizLayout = QHBoxLayout()
        self.selectedTimeRangeHorizLayout.setObjectName(u"selectedTimeRangeHorizLayout")
        self.selectedTimeRangeLabel = QLabel(self.selectedTimeRangeWidget)
        self.selectedTimeRangeLabel.setObjectName(u"selectedTimeRangeLabel")

        self.selectedTimeRangeHorizLayout.addWidget(self.selectedTimeRangeLabel)

        self.selectedTimeRangeButton = QPushButton(self.selectedTimeRangeWidget)
        self.selectedTimeRangeButton.setObjectName(u"selectedTimeRangeButton")

        self.selectedTimeRangeHorizLayout.addWidget(self.selectedTimeRangeButton)

        self.selectedTimeRangeResetButton = QPushButton(self.selectedTimeRangeWidget)
        self.selectedTimeRangeResetButton.setObjectName(u"selectedTimeRangeResetButton")

        self.selectedTimeRangeHorizLayout.addWidget(self.selectedTimeRangeResetButton)


        self.gridLayout_13.addLayout(self.selectedTimeRangeHorizLayout, 0, 0, 1, 1)


        self.gridLayout_4.addWidget(self.selectedTimeRangeWidget, 2, 0, 1, 1)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.proj_alg_label = QLabel(self.groupBox)
        self.proj_alg_label.setObjectName(u"proj_alg_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.proj_alg_label)

        self.proj_alg_tb = QToolButton(self.groupBox)
        self.proj_alg_tb.setObjectName(u"proj_alg_tb")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.proj_alg_tb.sizePolicy().hasHeightForWidth())
        self.proj_alg_tb.setSizePolicy(sizePolicy1)
        self.proj_alg_tb.setPopupMode(QToolButton.InstantPopup)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.proj_alg_tb)

        self.proj_red_label = QLabel(self.groupBox)
        self.proj_red_label.setObjectName(u"proj_red_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.proj_red_label)

        self.proj_red_tb = QToolButton(self.groupBox)
        self.proj_red_tb.setObjectName(u"proj_red_tb")
        sizePolicy1.setHeightForWidth(self.proj_red_tb.sizePolicy().hasHeightForWidth())
        self.proj_red_tb.setSizePolicy(sizePolicy1)
        self.proj_red_tb.setPopupMode(QToolButton.InstantPopup)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.proj_red_tb)

        self.proj_dim_label = QLabel(self.groupBox)
        self.proj_dim_label.setObjectName(u"proj_dim_label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.proj_dim_label)

        self.proj_dim_tb = QToolButton(self.groupBox)
        self.proj_dim_tb.setObjectName(u"proj_dim_tb")
        sizePolicy1.setHeightForWidth(self.proj_dim_tb.sizePolicy().hasHeightForWidth())
        self.proj_dim_tb.setSizePolicy(sizePolicy1)
        self.proj_dim_tb.setPopupMode(QToolButton.InstantPopup)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.proj_dim_tb)

        self.proj_dis_metric_label = QLabel(self.groupBox)
        self.proj_dis_metric_label.setObjectName(u"proj_dis_metric_label")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.proj_dis_metric_label)

        self.proj_dis_metric_tb = QToolButton(self.groupBox)
        self.proj_dis_metric_tb.setObjectName(u"proj_dis_metric_tb")
        sizePolicy1.setHeightForWidth(self.proj_dis_metric_tb.sizePolicy().hasHeightForWidth())
        self.proj_dis_metric_tb.setSizePolicy(sizePolicy1)
        self.proj_dis_metric_tb.setPopupMode(QToolButton.InstantPopup)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.proj_dis_metric_tb)

        self.proj_data_label = QLabel(self.groupBox)
        self.proj_data_label.setObjectName(u"proj_data_label")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.proj_data_label)

        self.proj_data = QToolButton(self.groupBox)
        self.proj_data.setObjectName(u"proj_data")
        sizePolicy1.setHeightForWidth(self.proj_data.sizePolicy().hasHeightForWidth())
        self.proj_data.setSizePolicy(sizePolicy1)
        self.proj_data.setPopupMode(QToolButton.InstantPopup)

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.proj_data)


        self.gridLayout_4.addLayout(self.formLayout, 1, 0, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.showSplinesBetweenTimeSteps = QCheckBox(self.groupBox)
        self.showSplinesBetweenTimeSteps.setObjectName(u"showSplinesBetweenTimeSteps")

        self.verticalLayout_4.addWidget(self.showSplinesBetweenTimeSteps)

        self.compareAllViaSegMap = QCheckBox(self.groupBox)
        self.compareAllViaSegMap.setObjectName(u"compareAllViaSegMap")

        self.verticalLayout_4.addWidget(self.compareAllViaSegMap)

        self.showLabelsCheckbox = QCheckBox(self.groupBox)
        self.showLabelsCheckbox.setObjectName(u"showLabelsCheckbox")
        self.showLabelsCheckbox.setChecked(True)

        self.verticalLayout_4.addWidget(self.showLabelsCheckbox)


        self.horizontalLayout_5.addLayout(self.verticalLayout_4)

        self.line_5 = QFrame(self.groupBox)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.VLine)
        self.line_5.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_5.addWidget(self.line_5)

        self.segThresholdContainerWidget = QWidget(self.groupBox)
        self.segThresholdContainerWidget.setObjectName(u"segThresholdContainerWidget")
        self.gridLayout_14 = QGridLayout(self.segThresholdContainerWidget)
        self.gridLayout_14.setSpacing(0)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.gridLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.useFastDTWCheckbox = QCheckBox(self.segThresholdContainerWidget)
        self.useFastDTWCheckbox.setObjectName(u"useFastDTWCheckbox")
        self.useFastDTWCheckbox.setChecked(True)

        self.verticalLayout_5.addWidget(self.useFastDTWCheckbox)

        self.line_7 = QFrame(self.segThresholdContainerWidget)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.HLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_5.addWidget(self.line_7)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.applySegThreshold = QPushButton(self.segThresholdContainerWidget)
        self.applySegThreshold.setObjectName(u"applySegThreshold")

        self.horizontalLayout_7.addWidget(self.applySegThreshold)

        self.resetSegThreshold = QPushButton(self.segThresholdContainerWidget)
        self.resetSegThreshold.setObjectName(u"resetSegThreshold")

        self.horizontalLayout_7.addWidget(self.resetSegThreshold)


        self.verticalLayout_5.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_6 = QLabel(self.segThresholdContainerWidget)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_6.addWidget(self.label_6)

        self.saturationSegThresholdSpinBox = QDoubleSpinBox(self.segThresholdContainerWidget)
        self.saturationSegThresholdSpinBox.setObjectName(u"saturationSegThresholdSpinBox")
        self.saturationSegThresholdSpinBox.setDecimals(3)
        self.saturationSegThresholdSpinBox.setSingleStep(0.001000000000000)

        self.horizontalLayout_6.addWidget(self.saturationSegThresholdSpinBox)

        self.label_7 = QLabel(self.segThresholdContainerWidget)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_6.addWidget(self.label_7)

        self.concentrationSegThresholdSpinBox = QDoubleSpinBox(self.segThresholdContainerWidget)
        self.concentrationSegThresholdSpinBox.setObjectName(u"concentrationSegThresholdSpinBox")
        self.concentrationSegThresholdSpinBox.setDecimals(3)
        self.concentrationSegThresholdSpinBox.setSingleStep(0.001000000000000)

        self.horizontalLayout_6.addWidget(self.concentrationSegThresholdSpinBox)


        self.verticalLayout_5.addLayout(self.horizontalLayout_6)


        self.gridLayout_14.addLayout(self.verticalLayout_5, 0, 0, 1, 1)


        self.horizontalLayout_5.addWidget(self.segThresholdContainerWidget)


        self.gridLayout_4.addLayout(self.horizontalLayout_5, 6, 0, 1, 1)

        self.line_6 = QFrame(self.groupBox)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShape(QFrame.HLine)
        self.line_6.setFrameShadow(QFrame.Sunken)

        self.gridLayout_4.addWidget(self.line_6, 3, 0, 1, 1)


        self.verticalLayout.addWidget(self.groupBox)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.groupBox_4 = QGroupBox(self.controlWidgetContainer)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_15 = QGridLayout(self.groupBox_4)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.scrollArea = QScrollArea(self.groupBox_4)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 233, 196))
        self.gridLayout_10 = QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.tsd_grid_layout = QGridLayout()
        self.tsd_grid_layout.setObjectName(u"tsd_grid_layout")
        self.label_3 = QLabel(self.scrollAreaWidgetContents)
        self.label_3.setObjectName(u"label_3")

        self.tsd_grid_layout.addWidget(self.label_3, 0, 1, 1, 1)

        self.label_2 = QLabel(self.scrollAreaWidgetContents)
        self.label_2.setObjectName(u"label_2")

        self.tsd_grid_layout.addWidget(self.label_2, 0, 0, 1, 1)

        self.label_4 = QLabel(self.scrollAreaWidgetContents)
        self.label_4.setObjectName(u"label_4")

        self.tsd_grid_layout.addWidget(self.label_4, 0, 2, 1, 1)

        self.label_8 = QLabel(self.scrollAreaWidgetContents)
        self.label_8.setObjectName(u"label_8")

        self.tsd_grid_layout.addWidget(self.label_8, 0, 3, 1, 1)

        self.tsd_grid_layout.setColumnStretch(3, 1)

        self.gridLayout_10.addLayout(self.tsd_grid_layout, 0, 0, 1, 1)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout_15.addWidget(self.scrollArea, 0, 0, 1, 1)


        self.horizontalLayout_2.addWidget(self.groupBox_4)

        self.groupBox_5 = QGroupBox(self.controlWidgetContainer)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.gridLayout_12 = QGridLayout(self.groupBox_5)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.scrollArea_2 = QScrollArea(self.groupBox_5)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 103, 213))
        self.gridLayout_11 = QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.groupSelectionGridLayout = QGridLayout()
        self.groupSelectionGridLayout.setObjectName(u"groupSelectionGridLayout")
        self.label_5 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_5.setObjectName(u"label_5")

        self.groupSelectionGridLayout.addWidget(self.label_5, 0, 0, 1, 1)


        self.gridLayout_11.addLayout(self.groupSelectionGridLayout, 0, 0, 1, 1)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.gridLayout_12.addWidget(self.scrollArea_2, 0, 0, 1, 1)


        self.horizontalLayout_2.addWidget(self.groupBox_5)

        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(3, 1)

        self.gridLayout_16.addLayout(self.verticalLayout, 0, 0, 1, 1)


        self.gridLayout_8.addWidget(self.controlWidgetContainer, 0, 2, 1, 1)

        self.gridLayout_8.setColumnStretch(0, 3)

        self.gridLayout_2.addLayout(self.gridLayout_8, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1284, 22))
        self.menu_Screenshots = QMenu(self.menubar)
        self.menu_Screenshots.setObjectName(u"menu_Screenshots")
        self.menuRun = QMenu(self.menubar)
        self.menuRun.setObjectName(u"menuRun")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menu_Screenshots.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menu_Screenshots.addAction(self.actionScreenshot)
        self.menu_Screenshots.addAction(self.actionScreenshot_Volume_Views)
        self.menu_Screenshots.addAction(self.actionScreenshot_Time_Series)
        self.menu_Screenshots.addAction(self.actionScreenshot_Controls)
        self.menu_Screenshots.addAction(self.actionScreenshot_All_Individually)
        self.menu_Screenshots.addAction(self.actionScreenshot_All_One_Image)
        self.menuRun.addAction(self.actionRun_All_with_Screenshots)
        self.menuRun.addAction(self.actionRun_All)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Vamled", None))
        self.actionScreenshot.setText(QCoreApplication.translate("MainWindow", u"Screenshot (Projection)", None))
        self.actionScreenshot_Volume_Views.setText(QCoreApplication.translate("MainWindow", u"Screenshot Volume Views", None))
        self.actionScreenshot_Time_Series.setText(QCoreApplication.translate("MainWindow", u"Screenshot Time Series", None))
        self.actionScreenshot_Controls.setText(QCoreApplication.translate("MainWindow", u"Screenshot Controls", None))
        self.actionScreenshot_All_Individually.setText(QCoreApplication.translate("MainWindow", u"Screenshot All Individually", None))
        self.actionScreenshot_All_One_Image.setText(QCoreApplication.translate("MainWindow", u"Screenshot All One Image", None))
        self.actionRun_All_with_Screenshots.setText(QCoreApplication.translate("MainWindow", u"Run All (with Screenshots)", None))
        self.actionRun_All.setText(QCoreApplication.translate("MainWindow", u"Run All", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Viewer Settings and Controls", None))
        self.coupleTF.setText(QCoreApplication.translate("MainWindow", u"Couple Transfer Functions", None))
        self.coupleCameras.setText(QCoreApplication.translate("MainWindow", u"Couple Camera Control", None))
        self.coupleTimeStep.setText(QCoreApplication.translate("MainWindow", u"Couple Time Step Selection", None))
        self.time_view_control_checkbox.setText(QCoreApplication.translate("MainWindow", u"Control Time with Time View on Drag", None))
        self.time_view_hover_control_checkbox.setText(QCoreApplication.translate("MainWindow", u"Show Time with Time View on Hover", None))
        self.zoomToBoxAButton.setText(QCoreApplication.translate("MainWindow", u"Zoom to Box A", None))
        self.zoomToBoxBButton.setText(QCoreApplication.translate("MainWindow", u"Zoom to Box B", None))
        self.zoomToBoxCButton.setText(QCoreApplication.translate("MainWindow", u"Zoom to Box C", None))
        self.resetViewsButton.setText(QCoreApplication.translate("MainWindow", u"Reset Views", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Transferfunctions", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_left), QCoreApplication.translate("MainWindow", u"Left Config", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_right), QCoreApplication.translate("MainWindow", u"Right Config", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Projection Controls", None))
        self.selectedTimeRangeLabel.setText(QCoreApplication.translate("MainWindow", u"Use patches in time range", None))
        self.selectedTimeRangeButton.setText(QCoreApplication.translate("MainWindow", u"Apply Range", None))
        self.selectedTimeRangeResetButton.setText(QCoreApplication.translate("MainWindow", u"Apply Reset", None))
#if QT_CONFIG(tooltip)
        self.proj_alg_label.setToolTip(QCoreApplication.translate("MainWindow", u"Projection Algorithm", None))
#endif // QT_CONFIG(tooltip)
        self.proj_alg_label.setText(QCoreApplication.translate("MainWindow", u"Projection Algorithm", None))
        self.proj_alg_tb.setText(QCoreApplication.translate("MainWindow", u"None", None))
#if QT_CONFIG(tooltip)
        self.proj_red_label.setToolTip(QCoreApplication.translate("MainWindow", u"Data Reduction Mode", None))
#endif // QT_CONFIG(tooltip)
        self.proj_red_label.setText(QCoreApplication.translate("MainWindow", u"Reduction (Group/Patch)", None))
        self.proj_red_tb.setText(QCoreApplication.translate("MainWindow", u"None", None))
#if QT_CONFIG(tooltip)
        self.proj_dim_label.setToolTip(QCoreApplication.translate("MainWindow", u"Projection Dimension", None))
#endif // QT_CONFIG(tooltip)
        self.proj_dim_label.setText(QCoreApplication.translate("MainWindow", u"Output Dimension", None))
        self.proj_dim_tb.setText(QCoreApplication.translate("MainWindow", u"None", None))
#if QT_CONFIG(tooltip)
        self.proj_dis_metric_label.setToolTip(QCoreApplication.translate("MainWindow", u"Dissimilarity Measure", None))
#endif // QT_CONFIG(tooltip)
        self.proj_dis_metric_label.setText(QCoreApplication.translate("MainWindow", u"Distance Metric", None))
        self.proj_dis_metric_tb.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.proj_data_label.setText(QCoreApplication.translate("MainWindow", u"Data", None))
        self.proj_data.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.showSplinesBetweenTimeSteps.setText(QCoreApplication.translate("MainWindow", u"Show Connected Patches", None))
        self.compareAllViaSegMap.setText(QCoreApplication.translate("MainWindow", u"Compare Via Segmentation", None))
        self.showLabelsCheckbox.setText(QCoreApplication.translate("MainWindow", u"Show Labels", None))
        self.useFastDTWCheckbox.setText(QCoreApplication.translate("MainWindow", u"Use fast dtw", None))
#if QT_CONFIG(tooltip)
        self.applySegThreshold.setToolTip(QCoreApplication.translate("MainWindow", u"Requires full recomputation for every new value! Apply the threshold to use for the segmentation decision. 1 if value is > threshold else 0.", None))
#endif // QT_CONFIG(tooltip)
        self.applySegThreshold.setText(QCoreApplication.translate("MainWindow", u"Apply Th.", None))
#if QT_CONFIG(tooltip)
        self.resetSegThreshold.setToolTip(QCoreApplication.translate("MainWindow", u"Use the minimum as minmal value instead of the threshold", None))
#endif // QT_CONFIG(tooltip)
        self.resetSegThreshold.setText(QCoreApplication.translate("MainWindow", u"Use min.", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Sat.:", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Con:", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Time Series Data View +  Selection", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"View2", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"View1", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Description", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Use as feature", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Group Selection", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Group Enabled", None))
        self.menu_Screenshots.setTitle(QCoreApplication.translate("MainWindow", u"&Screenshots", None))
        self.menuRun.setTitle(QCoreApplication.translate("MainWindow", u"Runs", None))
    # retranslateUi

