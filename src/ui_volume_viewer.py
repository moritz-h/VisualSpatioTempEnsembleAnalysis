# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'volume_viewer.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget)

class Ui_VolumeViewer(object):
    def setupUi(self, VolumeViewer):
        if not VolumeViewer.objectName():
            VolumeViewer.setObjectName(u"VolumeViewer")
        VolumeViewer.resize(465, 257)
        self.gridLayout = QGridLayout(VolumeViewer)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.loadDataButton = QToolButton(VolumeViewer)
        self.loadDataButton.setObjectName(u"loadDataButton")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadDataButton.sizePolicy().hasHeightForWidth())
        self.loadDataButton.setSizePolicy(sizePolicy)
        self.loadDataButton.setLayoutDirection(Qt.RightToLeft)
        self.loadDataButton.setCheckable(False)
        self.loadDataButton.setPopupMode(QToolButton.InstantPopup)
        self.loadDataButton.setToolButtonStyle(Qt.ToolButtonIconOnly)

        self.horizontalLayout_4.addWidget(self.loadDataButton)

        self.switchSatCon = QPushButton(VolumeViewer)
        self.switchSatCon.setObjectName(u"switchSatCon")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.switchSatCon.sizePolicy().hasHeightForWidth())
        self.switchSatCon.setSizePolicy(sizePolicy1)

        self.horizontalLayout_4.addWidget(self.switchSatCon)

        self.currentZoom = QLabel(VolumeViewer)
        self.currentZoom.setObjectName(u"currentZoom")

        self.horizontalLayout_4.addWidget(self.currentZoom)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(VolumeViewer)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.selectDataLabel = QLabel(VolumeViewer)
        self.selectDataLabel.setObjectName(u"selectDataLabel")
        self.selectDataLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.selectDataLabel)

        self.horizontalLayout_5.setStretch(1, 1)

        self.horizontalLayout_4.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_4.setStretch(0, 3)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 3)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.volumeViewContainer = QWidget(VolumeViewer)
        self.volumeViewContainer.setObjectName(u"volumeViewContainer")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.volumeViewContainer.sizePolicy().hasHeightForWidth())
        self.volumeViewContainer.setSizePolicy(sizePolicy2)
        self.volumeViewContainer.setContextMenuPolicy(Qt.CustomContextMenu)

        self.verticalLayout_2.addWidget(self.volumeViewContainer)


        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)


        self.retranslateUi(VolumeViewer)

        QMetaObject.connectSlotsByName(VolumeViewer)
    # setupUi

    def retranslateUi(self, VolumeViewer):
        VolumeViewer.setWindowTitle(QCoreApplication.translate("VolumeViewer", u"Form", None))
        self.loadDataButton.setText(QCoreApplication.translate("VolumeViewer", u"Load Data", None))
        self.switchSatCon.setText(QCoreApplication.translate("VolumeViewer", u"Saturation", None))
        self.currentZoom.setText(QCoreApplication.translate("VolumeViewer", u"Full Volume", None))
        self.label_3.setText(QCoreApplication.translate("VolumeViewer", u"Data:", None))
        self.selectDataLabel.setText(QCoreApplication.translate("VolumeViewer", u"Default", None))
    # retranslateUi

