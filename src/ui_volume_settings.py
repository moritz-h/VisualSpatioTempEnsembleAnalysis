# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'volume_settings.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_volumeSettings(object):
    def setupUi(self, volumeSettings):
        if not volumeSettings.objectName():
            volumeSettings.setObjectName(u"volumeSettings")
        volumeSettings.resize(277, 608)
        self.gridLayout = QGridLayout(volumeSettings)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.transferFunctionContainer = QGridLayout()
        self.transferFunctionContainer.setObjectName(u"transferFunctionContainer")

        self.verticalLayout.addLayout(self.transferFunctionContainer)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.loadTF = QPushButton(volumeSettings)
        self.loadTF.setObjectName(u"loadTF")

        self.horizontalLayout.addWidget(self.loadTF)

        self.saveTF = QPushButton(volumeSettings)
        self.saveTF.setObjectName(u"saveTF")

        self.horizontalLayout.addWidget(self.saveTF)

        self.applyTfToBoth = QPushButton(volumeSettings)
        self.applyTfToBoth.setObjectName(u"applyTfToBoth")

        self.horizontalLayout.addWidget(self.applyTfToBoth)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.verticalLayout.setStretch(0, 1)

        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)


        self.retranslateUi(volumeSettings)

        QMetaObject.connectSlotsByName(volumeSettings)
    # setupUi

    def retranslateUi(self, volumeSettings):
        volumeSettings.setWindowTitle(QCoreApplication.translate("volumeSettings", u"Form", None))
        self.loadTF.setText(QCoreApplication.translate("volumeSettings", u"Load", None))
        self.saveTF.setText(QCoreApplication.translate("volumeSettings", u"Save", None))
        self.applyTfToBoth.setText(QCoreApplication.translate("volumeSettings", u"Apply To Both", None))
    # retranslateUi

