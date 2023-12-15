import os
import sys
import numpy as np
import wfdb
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QUrl, QBuffer, QIODevice
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QFileDialog, QPushButton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa as librosa
from scipy.fft import fft
from cine_canvas import MplCanvas
import scipy.io.wavfile as wavf
# Increase the threshold for the warning
plt.rcParams['figure.max_open_warning'] = 50  # Set it to a value higher than 20

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1113, 859)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_17 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setElideMode(QtCore.Qt.ElideLeft)
        self.tabWidget.setObjectName("tabWidget")
        self.smoothingWindowTab = QtWidgets.QWidget()
        self.smoothingWindowTab.setObjectName("smoothingWindowTab")
        self.gridLayout_18 = QtWidgets.QGridLayout(self.smoothingWindowTab)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.frame_2 = QtWidgets.QFrame(self.smoothingWindowTab)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_7)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.smoothingWindowLayout = QtWidgets.QVBoxLayout()
        self.smoothingWindowLayout.setObjectName("smoothingWindowLayout")
        self.gridLayout_3.addLayout(self.smoothingWindowLayout, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame_7, 0, 0, 1, 1)
        self.frame_8 = QtWidgets.QFrame(self.frame_2)
        self.frame_8.setMaximumSize(QtCore.QSize(16777215, 200))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.frame_8)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.label_8 = QtWidgets.QLabel(self.frame_8)
        self.label_8.setObjectName("label_8")
        self.gridLayout_16.addWidget(self.label_8, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame_8)
        self.label.setMaximumSize(QtCore.QSize(150, 20))
        self.label.setObjectName("label")
        self.gridLayout_16.addWidget(self.label, 2, 0, 1, 1)
        self.smothingComboBox = QtWidgets.QComboBox(self.frame_8)
        self.smothingComboBox.setMaximumSize(QtCore.QSize(500, 25))
        self.smothingComboBox.setObjectName("smothingComboBox")
        self.smothingComboBox.addItem("")
        self.smothingComboBox.addItem("")
        self.smothingComboBox.addItem("")
        self.smothingComboBox.addItem("")
        self.gridLayout_16.addWidget(self.smothingComboBox, 2, 1, 1, 1)
        self.stdSlider = QtWidgets.QSlider(self.frame_8)
        self.stdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stdSlider.setObjectName("stdSlider")
        self.gridLayout_16.addWidget(self.stdSlider, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.frame_8, 1, 0, 1, 1)
        self.gridLayout_18.addWidget(self.frame_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.smoothingWindowTab, "")
        self.unifromRangeTab = QtWidgets.QWidget()
        self.unifromRangeTab.setObjectName("unifromRangeTab")
        self.gridLayout_22 = QtWidgets.QGridLayout(self.unifromRangeTab)
        self.gridLayout_22.setObjectName("gridLayout_22")
        self.frame_3 = QtWidgets.QFrame(self.unifromRangeTab)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_9 = QtWidgets.QFrame(self.frame_3)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.frame_9)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.uniformTimeLayout = QtWidgets.QVBoxLayout()
        self.uniformTimeLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.uniformTimeLayout.setObjectName("uniformTimeLayout")
        self.gridLayout_10.addLayout(self.uniformTimeLayout, 0, 0, 1, 1)
        self.uniformFrequencyLayout = QtWidgets.QVBoxLayout()
        self.uniformFrequencyLayout.setObjectName("uniformFrequencyLayout")
        self.gridLayout_10.addLayout(self.uniformFrequencyLayout, 0, 1, 1, 1)
        self.gridLayout_5.addWidget(self.frame_9, 0, 0, 1, 1)
        self.frame_13 = QtWidgets.QFrame(self.frame_3)
        self.frame_13.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_13)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.zoomInButton_1 = QtWidgets.QPushButton(self.frame_13)
        self.zoomInButton_1.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomInButton_1.setObjectName("zoomInButton_1")
        self.gridLayout_6.addWidget(self.zoomInButton_1, 0, 0, 1, 1)
        self.zoomOutButton_1 = QtWidgets.QPushButton(self.frame_13)
        self.zoomOutButton_1.setMaximumSize(QtCore.QSize(100, 50))
        self.zoomOutButton_1.setObjectName("zoomOutButton_1")
        self.gridLayout_6.addWidget(self.zoomOutButton_1, 0, 1, 1, 1)
        self.pausePlayButton_1 = QtWidgets.QPushButton(self.frame_13)
        self.pausePlayButton_1.setMinimumSize(QtCore.QSize(400, 0))
        self.pausePlayButton_1.setMaximumSize(QtCore.QSize(400, 16777215))
        self.pausePlayButton_1.setObjectName("pausePlayButton_1")
        self.gridLayout_6.addWidget(self.pausePlayButton_1, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame_13)
        self.label_5.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_5.setObjectName("label_5")
        self.gridLayout_6.addWidget(self.label_5, 0, 4, 1, 1)
        self.rewindButton_1 = QtWidgets.QPushButton(self.frame_13)
        self.rewindButton_1.setMaximumSize(QtCore.QSize(75, 16777215))
        self.rewindButton_1.setObjectName("rewindButton_1")
        self.gridLayout_6.addWidget(self.rewindButton_1, 0, 3, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.uniforRangeSlider_1 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_1.setObjectName("uniforRangeSlider_1")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_1)
        self.uniforRangeSlider_2 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_2.setObjectName("uniforRangeSlider_2")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_2)
        self.uniforRangeSlider_3 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_3.setObjectName("uniforRangeSlider_3")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_3)
        self.uniforRangeSlider_4 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_4.setObjectName("uniforRangeSlider_4")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_4)
        self.uniforRangeSlider_5 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_5.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_5.setObjectName("uniforRangeSlider_5")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_5)
        self.uniforRangeSlider_6 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_6.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_6.setObjectName("uniforRangeSlider_6")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_6)
        self.uniforRangeSlider_7 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_7.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_7.setObjectName("uniforRangeSlider_7")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_7)
        self.uniforRangeSlider_8 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_8.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_8.setObjectName("uniforRangeSlider_8")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_8)
        self.uniforRangeSlider_9 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_9.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_9.setObjectName("uniforRangeSlider_9")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_9)
        self.uniforRangeSlider_10 = QtWidgets.QSlider(self.frame_13)
        self.uniforRangeSlider_10.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_10.setObjectName("uniforRangeSlider_10")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_10)
        self.gridLayout_6.addLayout(self.horizontalLayout, 2, 0, 1, 7)
        self.spinBox = QtWidgets.QSpinBox(self.frame_13)
        self.spinBox.setMinimumSize(QtCore.QSize(200, 0))
        self.spinBox.setObjectName("spinBox")
        self.gridLayout_6.addWidget(self.spinBox, 0, 5, 1, 1)
        self.gridLayout_5.addWidget(self.frame_13, 1, 0, 1, 1)
        self.gridLayout_22.addWidget(self.frame_3, 0, 0, 1, 1)
        self.checkBox_1 = QtWidgets.QCheckBox(self.unifromRangeTab)
        self.checkBox_1.setMaximumSize(QtCore.QSize(150, 16777215))
        self.checkBox_1.setObjectName("checkBox_1")
        self.gridLayout_22.addWidget(self.checkBox_1, 1, 0, 1, 1)
        self.tabWidget.addTab(self.unifromRangeTab, "")
        self.animalTab = QtWidgets.QWidget()
        self.animalTab.setObjectName("animalTab")
        self.gridLayout_21 = QtWidgets.QGridLayout(self.animalTab)
        self.gridLayout_21.setObjectName("gridLayout_21")
        self.frame_4 = QtWidgets.QFrame(self.animalTab)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.frame_10 = QtWidgets.QFrame(self.frame_4)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame_10)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.animalTimeLayout = QtWidgets.QVBoxLayout()
        self.animalTimeLayout.setObjectName("animalTimeLayout")
        self.gridLayout_8.addLayout(self.animalTimeLayout, 0, 0, 1, 1)
        self.animalFrequencyLayout = QtWidgets.QVBoxLayout()
        self.animalFrequencyLayout.setObjectName("animalFrequencyLayout")
        self.gridLayout_8.addLayout(self.animalFrequencyLayout, 0, 1, 1, 1)
        self.gridLayout_7.addWidget(self.frame_10, 0, 0, 1, 1)
        self.frame_14 = QtWidgets.QFrame(self.frame_4)
        self.frame_14.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.frame_14)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.gridLayout_34 = QtWidgets.QGridLayout()
        self.gridLayout_34.setObjectName("gridLayout_34")
        self.gridLayout_23 = QtWidgets.QGridLayout()
        self.gridLayout_23.setObjectName("gridLayout_23")
        self.label_12 = QtWidgets.QLabel(self.frame_14)
        self.label_12.setObjectName("label_12")
        self.gridLayout_23.addWidget(self.label_12, 2, 0, 1, 1)
        self.animalSlider_1 = QtWidgets.QSlider(self.frame_14)
        self.animalSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.animalSlider_1.setObjectName("animalSlider_1")
        self.gridLayout_23.addWidget(self.animalSlider_1, 1, 0, 1, 1)
        self.gridLayout_34.addLayout(self.gridLayout_23, 0, 0, 1, 1)
        self.gridLayout_24 = QtWidgets.QGridLayout()
        self.gridLayout_24.setObjectName("gridLayout_24")
        self.animalSlider_4 = QtWidgets.QSlider(self.frame_14)
        self.animalSlider_4.setMaximumSize(QtCore.QSize(20, 16777215))
        self.animalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.animalSlider_4.setObjectName("animalSlider_4")
        self.gridLayout_24.addWidget(self.animalSlider_4, 0, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.frame_14)
        self.label_13.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_13.setObjectName("label_13")
        self.gridLayout_24.addWidget(self.label_13, 1, 0, 1, 1)
        self.gridLayout_34.addLayout(self.gridLayout_24, 0, 1, 1, 1)
        self.gridLayout_25 = QtWidgets.QGridLayout()
        self.gridLayout_25.setObjectName("gridLayout_25")
        self.animalSlider_3 = QtWidgets.QSlider(self.frame_14)
        self.animalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.animalSlider_3.setObjectName("animalSlider_3")
        self.gridLayout_25.addWidget(self.animalSlider_3, 0, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.frame_14)
        self.label_15.setObjectName("label_15")
        self.gridLayout_25.addWidget(self.label_15, 1, 0, 1, 1)
        self.gridLayout_34.addLayout(self.gridLayout_25, 0, 2, 1, 1)
        self.gridLayout_26 = QtWidgets.QGridLayout()
        self.gridLayout_26.setObjectName("gridLayout_26")
        self.animalSlider_2 = QtWidgets.QSlider(self.frame_14)
        self.animalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.animalSlider_2.setObjectName("animalSlider_2")
        self.gridLayout_26.addWidget(self.animalSlider_2, 0, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.frame_14)
        self.label_16.setObjectName("label_16")
        self.gridLayout_26.addWidget(self.label_16, 1, 0, 1, 1)
        self.gridLayout_34.addLayout(self.gridLayout_26, 0, 3, 1, 1)
        self.gridLayout_13.addLayout(self.gridLayout_34, 1, 0, 1, 7)
        self.zoomInButton_2 = QtWidgets.QPushButton(self.frame_14)
        self.zoomInButton_2.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomInButton_2.setObjectName("zoomInButton_2")
        self.gridLayout_13.addWidget(self.zoomInButton_2, 0, 0, 1, 1)
        self.checkBox_2 = QtWidgets.QCheckBox(self.frame_14)
        self.checkBox_2.setObjectName("checkBox_2")
        self.gridLayout_13.addWidget(self.checkBox_2, 2, 0, 1, 1)
        self.zoomOutButton_2 = QtWidgets.QPushButton(self.frame_14)
        self.zoomOutButton_2.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomOutButton_2.setObjectName("zoomOutButton_2")
        self.gridLayout_13.addWidget(self.zoomOutButton_2, 0, 1, 1, 1)
        self.spinBox_2 = QtWidgets.QSpinBox(self.frame_14)
        self.spinBox_2.setMinimumSize(QtCore.QSize(300, 0))
        self.spinBox_2.setObjectName("spinBox_2")
        self.gridLayout_13.addWidget(self.spinBox_2, 0, 6, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame_14)
        self.label_6.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_6.setObjectName("label_6")
        self.gridLayout_13.addWidget(self.label_6, 0, 5, 1, 1)
        self.rewindButton_2 = QtWidgets.QPushButton(self.frame_14)
        self.rewindButton_2.setMaximumSize(QtCore.QSize(75, 16777215))
        self.rewindButton_2.setObjectName("rewindButton_2")
        self.gridLayout_13.addWidget(self.rewindButton_2, 0, 4, 1, 1)
        self.pausePlayButton_2 = QtWidgets.QPushButton(self.frame_14)
        self.pausePlayButton_2.setMinimumSize(QtCore.QSize(300, 0))
        self.pausePlayButton_2.setMaximumSize(QtCore.QSize(700, 16777215))
        self.pausePlayButton_2.setObjectName("pausePlayButton_2")
        self.gridLayout_13.addWidget(self.pausePlayButton_2, 0, 3, 1, 1)
        self.gridLayout_7.addWidget(self.frame_14, 1, 0, 1, 1)
        self.gridLayout_21.addWidget(self.frame_4, 0, 0, 1, 1)
        self.tabWidget.addTab(self.animalTab, "")
        self.musicTab = QtWidgets.QWidget()
        self.musicTab.setObjectName("musicTab")
        self.gridLayout_20 = QtWidgets.QGridLayout(self.musicTab)
        self.gridLayout_20.setObjectName("gridLayout_20")
        self.frame_5 = QtWidgets.QFrame(self.musicTab)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.frame_11 = QtWidgets.QFrame(self.frame_5)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.frame_11)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.musicalTimeLayout = QtWidgets.QVBoxLayout()
        self.musicalTimeLayout.setObjectName("musicalTimeLayout")
        self.gridLayout_12.addLayout(self.musicalTimeLayout, 0, 0, 1, 1)
        self.musicalFrequencyLayout = QtWidgets.QVBoxLayout()
        self.musicalFrequencyLayout.setObjectName("musicalFrequencyLayout")
        self.gridLayout_12.addLayout(self.musicalFrequencyLayout, 0, 1, 1, 1)
        self.gridLayout_11.addWidget(self.frame_11, 0, 0, 1, 1)
        self.frame_15 = QtWidgets.QFrame(self.frame_5)
        self.frame_15.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.frame_15)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.gridLayout_33 = QtWidgets.QGridLayout()
        self.gridLayout_33.setObjectName("gridLayout_33")
        self.gridLayout_28 = QtWidgets.QGridLayout()
        self.gridLayout_28.setObjectName("gridLayout_28")
        self.label_18 = QtWidgets.QLabel(self.frame_15)
        self.label_18.setObjectName("label_18")
        self.gridLayout_28.addWidget(self.label_18, 2, 0, 1, 1)
        self.musicSlider_1 = QtWidgets.QSlider(self.frame_15)
        self.musicSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.musicSlider_1.setObjectName("musicSlider_1")
        self.gridLayout_28.addWidget(self.musicSlider_1, 1, 0, 1, 1)
        self.gridLayout_33.addLayout(self.gridLayout_28, 0, 0, 1, 1)
        self.gridLayout_29 = QtWidgets.QGridLayout()
        self.gridLayout_29.setObjectName("gridLayout_29")
        self.musicSlider_2 = QtWidgets.QSlider(self.frame_15)
        self.musicSlider_2.setMaximumSize(QtCore.QSize(20, 16777215))
        self.musicSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.musicSlider_2.setObjectName("musicSlider_2")
        self.gridLayout_29.addWidget(self.musicSlider_2, 0, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.frame_15)
        self.label_20.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_20.setObjectName("label_20")
        self.gridLayout_29.addWidget(self.label_20, 1, 0, 1, 1)
        self.gridLayout_33.addLayout(self.gridLayout_29, 0, 1, 1, 1)
        self.gridLayout_30 = QtWidgets.QGridLayout()
        self.gridLayout_30.setObjectName("gridLayout_30")
        self.musicSlider_3 = QtWidgets.QSlider(self.frame_15)
        self.musicSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.musicSlider_3.setObjectName("musicSlider_3")
        self.gridLayout_30.addWidget(self.musicSlider_3, 0, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.frame_15)
        self.label_21.setObjectName("label_21")
        self.gridLayout_30.addWidget(self.label_21, 1, 0, 1, 1)
        self.gridLayout_33.addLayout(self.gridLayout_30, 0, 2, 1, 1)
        self.gridLayout_31 = QtWidgets.QGridLayout()
        self.gridLayout_31.setObjectName("gridLayout_31")
        self.musicSlider_4 = QtWidgets.QSlider(self.frame_15)
        self.musicSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.musicSlider_4.setObjectName("musicSlider_4")
        self.gridLayout_31.addWidget(self.musicSlider_4, 0, 0, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.frame_15)
        self.label_22.setObjectName("label_22")
        self.gridLayout_31.addWidget(self.label_22, 1, 0, 1, 1)
        self.gridLayout_33.addLayout(self.gridLayout_31, 0, 3, 1, 1)
        self.gridLayout_9.addLayout(self.gridLayout_33, 3, 0, 1, 11)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem, 0, 3, 1, 1)
        self.checkBox_3 = QtWidgets.QCheckBox(self.frame_15)
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout_9.addWidget(self.checkBox_3, 4, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame_15)
        self.label_14.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_14.setObjectName("label_14")
        self.gridLayout_9.addWidget(self.label_14, 0, 7, 1, 1)
        self.zoomInButton_3 = QtWidgets.QPushButton(self.frame_15)
        self.zoomInButton_3.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomInButton_3.setObjectName("zoomInButton_3")
        self.gridLayout_9.addWidget(self.zoomInButton_3, 0, 0, 1, 1)
        self.rewindButton_3 = QtWidgets.QPushButton(self.frame_15)
        self.rewindButton_3.setMaximumSize(QtCore.QSize(75, 16777215))
        self.rewindButton_3.setObjectName("rewindButton_3")
        self.gridLayout_9.addWidget(self.rewindButton_3, 0, 6, 1, 1)
        self.zoomOutButton_3 = QtWidgets.QPushButton(self.frame_15)
        self.zoomOutButton_3.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomOutButton_3.setObjectName("zoomOutButton_3")
        self.gridLayout_9.addWidget(self.zoomOutButton_3, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem1, 0, 2, 1, 1)
        self.spinBox_3 = QtWidgets.QSpinBox(self.frame_15)
        self.spinBox_3.setMinimumSize(QtCore.QSize(300, 0))
        self.spinBox_3.setObjectName("spinBox_3")
        self.gridLayout_9.addWidget(self.spinBox_3, 0, 8, 1, 3)
        self.pausePlayButton_3 = QtWidgets.QPushButton(self.frame_15)
        self.pausePlayButton_3.setMinimumSize(QtCore.QSize(300, 0))
        self.pausePlayButton_3.setMaximumSize(QtCore.QSize(700, 16777215))
        self.pausePlayButton_3.setObjectName("pausePlayButton_3")
        self.gridLayout_9.addWidget(self.pausePlayButton_3, 0, 4, 1, 2)
        self.gridLayout_11.addWidget(self.frame_15, 1, 0, 1, 1)
        self.gridLayout_20.addWidget(self.frame_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.musicTab, "")
        self.ecgTab = QtWidgets.QWidget()
        self.ecgTab.setObjectName("ecgTab")
        self.gridLayout_19 = QtWidgets.QGridLayout(self.ecgTab)
        self.gridLayout_19.setObjectName("gridLayout_19")
        self.frame_6 = QtWidgets.QFrame(self.ecgTab)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.frame_12 = QtWidgets.QFrame(self.frame_6)
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.frame_12)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.ecgTimeLayout = QtWidgets.QVBoxLayout()
        self.ecgTimeLayout.setObjectName("ecgTimeLayout")
        self.gridLayout_15.addLayout(self.ecgTimeLayout, 0, 0, 1, 1)
        self.ecgFrequencyLayout = QtWidgets.QVBoxLayout()
        self.ecgFrequencyLayout.setObjectName("ecgFrequencyLayout")
        self.gridLayout_15.addLayout(self.ecgFrequencyLayout, 0, 1, 1, 1)
        self.gridLayout_14.addWidget(self.frame_12, 0, 0, 1, 1)
        self.frame_16 = QtWidgets.QFrame(self.frame_6)
        self.frame_16.setMaximumSize(QtCore.QSize(16777215, 250))
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.gridLayout_32 = QtWidgets.QGridLayout(self.frame_16)
        self.gridLayout_32.setObjectName("gridLayout_32")
        self.zoomOutButton_4 = QtWidgets.QPushButton(self.frame_16)
        self.zoomOutButton_4.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomOutButton_4.setObjectName("zoomOutButton_4")
        self.gridLayout_32.addWidget(self.zoomOutButton_4, 0, 2, 1, 1)
        self.zoomInButton_4 = QtWidgets.QPushButton(self.frame_16)
        self.zoomInButton_4.setMaximumSize(QtCore.QSize(100, 16777215))
        self.zoomInButton_4.setObjectName("zoomInButton_4")
        self.gridLayout_32.addWidget(self.zoomInButton_4, 0, 1, 1, 1)
        self.checkBox_4 = QtWidgets.QCheckBox(self.frame_16)
        self.checkBox_4.setObjectName("checkBox_4")
        self.gridLayout_32.addWidget(self.checkBox_4, 3, 1, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_36 = QtWidgets.QGridLayout()
        self.gridLayout_36.setObjectName("gridLayout_36")
        self.ecgSlider_1 = QtWidgets.QSlider(self.frame_16)
        self.ecgSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.ecgSlider_1.setObjectName("ecgSlider_1")
        self.gridLayout_36.addWidget(self.ecgSlider_1, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame_16)
        self.label_4.setObjectName("label_4")
        self.gridLayout_36.addWidget(self.label_4, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_36, 0, 0, 1, 1)
        self.gridLayout_37 = QtWidgets.QGridLayout()
        self.gridLayout_37.setObjectName("gridLayout_37")
        self.ecgSlider_3 = QtWidgets.QSlider(self.frame_16)
        self.ecgSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.ecgSlider_3.setObjectName("ecgSlider_3")
        self.gridLayout_37.addWidget(self.ecgSlider_3, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame_16)
        self.label_2.setObjectName("label_2")
        self.gridLayout_37.addWidget(self.label_2, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_37, 0, 2, 1, 1)
        self.gridLayout_38 = QtWidgets.QGridLayout()
        self.gridLayout_38.setObjectName("gridLayout_38")
        self.ecgSlider_4 = QtWidgets.QSlider(self.frame_16)
        self.ecgSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.ecgSlider_4.setObjectName("ecgSlider_4")
        self.gridLayout_38.addWidget(self.ecgSlider_4, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_16)
        self.label_3.setObjectName("label_3")
        self.gridLayout_38.addWidget(self.label_3, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_38, 0, 3, 1, 1)
        self.gridLayout_35 = QtWidgets.QGridLayout()
        self.gridLayout_35.setObjectName("gridLayout_35")
        self.ecgSlider_2 = QtWidgets.QSlider(self.frame_16)
        self.ecgSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.ecgSlider_2.setObjectName("ecgSlider_2")
        self.gridLayout_35.addWidget(self.ecgSlider_2, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.frame_16)
        self.label_7.setObjectName("label_7")
        self.gridLayout_35.addWidget(self.label_7, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_35, 0, 1, 1, 1)
        self.gridLayout_32.addLayout(self.gridLayout_4, 1, 1, 1, 7)
        self.spinBox_4 = QtWidgets.QSpinBox(self.frame_16)
        self.spinBox_4.setMinimumSize(QtCore.QSize(300, 0))
        self.spinBox_4.setObjectName("spinBox_4")
        self.gridLayout_32.addWidget(self.spinBox_4, 0, 7, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.frame_16)
        self.label_19.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_19.setObjectName("label_19")
        self.gridLayout_32.addWidget(self.label_19, 0, 6, 1, 1)
        self.rewindButton_4 = QtWidgets.QPushButton(self.frame_16)
        self.rewindButton_4.setMaximumSize(QtCore.QSize(75, 16777215))
        self.rewindButton_4.setObjectName("rewindButton_4")
        self.gridLayout_32.addWidget(self.rewindButton_4, 0, 5, 1, 1)
        self.pausePlayButton_4 = QtWidgets.QPushButton(self.frame_16)
        self.pausePlayButton_4.setMinimumSize(QtCore.QSize(300, 0))
        self.pausePlayButton_4.setMaximumSize(QtCore.QSize(700, 16777215))
        self.pausePlayButton_4.setObjectName("pausePlayButton_4")
        self.gridLayout_32.addWidget(self.pausePlayButton_4, 0, 4, 1, 1)
        self.gridLayout_14.addWidget(self.frame_16, 1, 0, 1, 1)
        self.gridLayout_19.addWidget(self.frame_6, 0, 0, 1, 1)
        self.tabWidget.addTab(self.ecgTab, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.gridLayout_17.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1113, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # create canvas
        self.smoothingWindowCanvas = FigureCanvas(plt.figure(figsize=(4, 3)))

        self.unifromTimeInputCanvas = MplCanvas(MainWindow,1,1)
        self.unifromTimeOutputCanvas = MplCanvas(MainWindow,1,1)
        self.unifromFrequencyCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.unifromInputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.unifromOutputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))

        self.animalTimeInputCanvas = MplCanvas(MainWindow,1,1)
        self.animalTimeOutputCanvas = MplCanvas(MainWindow,1,1)
        self.animalFrequencyCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.animalInputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.animalOutputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))

        self.musicTimeInputCanvas =  MplCanvas(MainWindow,1,1)
        self.musicTimeOutputCanvas = MplCanvas(MainWindow,1,1)
        self.musicFrequencyCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.musicInputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.musicOutputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))

        self.ecgTimeInputCanvas = MplCanvas(MainWindow,1,1)
        self.ecgTimeOutputCanvas = MplCanvas(MainWindow,1,1)
        self.ecgFrequencyCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.ecgInputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))
        self.ecgOutputSpectrogramCanvas = FigureCanvas(plt.figure(figsize=(1,1)))

        # initialize empty canvases
        self.init_empty_canvases()


        for i in range(1, 11):
            slider_name1 = f"uniforRangeSlider_{i}"
            slider_instance1 = getattr(self, slider_name1, None)
            if slider_instance1 is not None:
                slider_instance1.setRange(0,10)
                slider_instance1.setValue(10)
                slider_instance1.valueChanged.connect(lambda value, range=i-1: self.uniformfrecuencyupdate(value, range))
        
        for i in range(1, 5):
            slider_name2 = f"animalSlider_{i}"
            slider_instance2 = getattr(self, slider_name2, None)
            if slider_instance2 is not None:
                slider_instance2.setRange(0,10)
                slider_instance2.setValue(10)
                slider_instance2.valueChanged.connect(lambda value, range=i: self.animalfrequencycomp(value, range))

        for i in range(1, 5):
            slider_name3 = f"musicSlider_{i}"
            slider_instance3 = getattr(self, slider_name3, None)
            if slider_instance3 is not None:
                slider_instance3.setRange(0,10)
                slider_instance3.setValue(10)
                slider_instance3.valueChanged.connect(lambda value, range=i: self.musicfrequencycomp(value, range))

        for i in range(1, 5):
            slider_name4 = f"ecgSlider_{i}"
            slider_instance4 = getattr(self, slider_name4, None)
            if slider_instance4 is not None:
                slider_instance4.setRange(0,10)
                slider_instance4.setValue(10)
                slider_instance4.valueChanged.connect(lambda value, range=i: self.ecgfrequencycomp(value, range))
            
        self.checkBox_1.stateChanged.connect(lambda state, canvas=self.unifromInputSpectrogramCanvas,canvas2=self.unifromOutputSpectrogramCanvas: self.toggle_spectrogram(state, canvas,canvas2))
        self.checkBox_2.stateChanged.connect(lambda state, canvas=self.animalInputSpectrogramCanvas ,canvas2=self.animalOutputSpectrogramCanvas: self.toggle_spectrogram(state, canvas,canvas2))
        self.checkBox_3.stateChanged.connect(lambda state, canvas=self.musicInputSpectrogramCanvas , canvas2= self.musicOutputSpectrogramCanvas: self.toggle_spectrogram(state, canvas,canvas2))
        self.checkBox_4.stateChanged.connect(lambda state, canvas=self.ecgInputSpectrogramCanvas,canvas2=self.ecgOutputSpectrogramCanvas: self.toggle_spectrogram(state, canvas,canvas2))    



        self.actionOpen.triggered.connect(self.loadFile)  # Connect to loadWavFile method

        self.pausePlayButton_3.clicked.connect(self.playPauseLoadedSound)  # Connect to playPauseLoadedSound method
        self.pausePlayButton_3.clicked.connect(self.musicTimeInputCanvas.play_or_pause) 
        self.pausePlayButton_2.clicked.connect(self.playPauseLoadedSound)  # Connect to playPauseLoadedSound method
        self.pausePlayButton_2.clicked.connect(self.animalTimeInputCanvas.play_or_pause) 
        self.pausePlayButton_1.clicked.connect(self.playPauseLoadedSound)  # Connect to playPauseLoadedSound method
        self.pausePlayButton_1.clicked.connect(self.unifromTimeInputCanvas.play_or_pause) 
        self.pausePlayButton_4.clicked.connect(self.playPauseLoadedSound)  # Connect to playPauseLoadedSound method
        self.pausePlayButton_4.clicked.connect(self.ecgTimeInputCanvas.play_or_pause) 

        self.rewindButton_2.clicked.connect(self.rewindLoadedSound) 
        self.rewindButton_2.clicked.connect(self.animalTimeInputCanvas.rewind) 
        self.rewindButton_3.clicked.connect(self.rewindLoadedSound)
        self.rewindButton_3.clicked.connect(self.musicTimeInputCanvas.rewind)
        self.rewindButton_4.clicked.connect(self.ecgTimeInputCanvas.rewind)
        

        self.smothingComboBox.currentIndexChanged.connect(self.chooseSmoothingWindow)

        self.stdSlider.setMinimum(1)
        self.stdSlider.setMaximum(20)
        self.stdSlider.setValue(5)  # Initial value of sigma
        self.stdSlider.valueChanged.connect(self.updateGaussian)

        self.sigma = self.stdSlider.value()



        self.zoomInButton_1.clicked.connect(self.unifromTimeInputCanvas.zoom_in)
        self.zoomOutButton_1.clicked.connect(self.unifromTimeInputCanvas.zoom_out)
        self.zoomInButton_2.clicked.connect(self.animalTimeInputCanvas.zoom_in)
        self.zoomOutButton_2.clicked.connect(self.animalTimeInputCanvas.zoom_out)
        self.zoomInButton_3.clicked.connect(self.musicTimeInputCanvas.zoom_in)
        self.zoomOutButton_3.clicked.connect(self.musicTimeInputCanvas.zoom_out)
        self.zoomInButton_4.clicked.connect(self.ecgTimeInputCanvas.zoom_in)
        self.zoomOutButton_4.clicked.connect(self.ecgTimeInputCanvas.zoom_out)


        #self.speedSlider_1.valueChanged.connect(self.unifromTimeInputCanvas.control_speed)
        #self.speedSlider_2.valueChanged.connect(self.animalTimeInputCanvas.control_speed)
        #self.speedSlider_3.valueChanged.connect(self.musicTimeInputCanvas.control_speed)
        #self.speedSlider_2.valueChanged.connect(self.ecgTimeInputCanvas.control_speed)
        self.number_of_output_file=0
        self.tabWidget.currentChanged.connect(self.resetNumber_of_output_file)
        self.name_of_output=""


    def init_empty_canvases(self):
            # Create an empty subplot for each canvas
        window_type = self.smothingComboBox.currentText()

        ax = self.smoothingWindowCanvas.figure.add_subplot(111)
        ax.plot(np.ones(100))
        ax.set_title(f"{window_type} Window")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        self.smoothingWindowCanvas.draw()

        self.unifromFrequencyCanvas.figure.add_subplot(111)
        self.unifromInputSpectrogramCanvas.figure.add_subplot(111)
        self.unifromOutputSpectrogramCanvas.figure.add_subplot(111)

        self.animalFrequencyCanvas.figure.add_subplot(111)
        self.animalInputSpectrogramCanvas.figure.add_subplot(111)
        self.animalOutputSpectrogramCanvas.figure.add_subplot(111)

        self.musicFrequencyCanvas.figure.add_subplot(111)
        self.musicInputSpectrogramCanvas.figure.add_subplot(111)
        self.musicOutputSpectrogramCanvas.figure.add_subplot(111)

        self.ecgFrequencyCanvas.figure.add_subplot(111)
        self.ecgInputSpectrogramCanvas.figure.add_subplot(111)
        self.ecgOutputSpectrogramCanvas.figure.add_subplot(111)

            # Add the empty canvases to the layouts
        self.smoothingWindowLayout.layout().addWidget(self.smoothingWindowCanvas)

        self.uniformTimeLayout.layout().addWidget(self.unifromTimeInputCanvas)
        self.uniformTimeLayout.layout().addWidget(self.unifromTimeOutputCanvas)
        self.uniformFrequencyLayout.layout().addWidget(self.unifromFrequencyCanvas)
        self.uniformFrequencyLayout.layout().addWidget(self.unifromInputSpectrogramCanvas)
        self.uniformFrequencyLayout.layout().addWidget(self.unifromOutputSpectrogramCanvas)

        self.animalTimeLayout.layout().addWidget(self.animalTimeInputCanvas)
        self.animalTimeLayout.layout().addWidget(self.animalTimeOutputCanvas)
        self.animalFrequencyLayout.layout().addWidget(self.animalFrequencyCanvas)
        self.animalFrequencyLayout.layout().addWidget(self.animalInputSpectrogramCanvas)
        self.animalFrequencyLayout.layout().addWidget(self.animalOutputSpectrogramCanvas)

        self.musicalTimeLayout.layout().addWidget(self.musicTimeInputCanvas)
        self.musicalTimeLayout.layout().addWidget(self.musicTimeOutputCanvas)
        self.musicalFrequencyLayout.layout().addWidget(self.musicFrequencyCanvas)
        self.musicalFrequencyLayout.layout().addWidget(self.musicInputSpectrogramCanvas)
        self.musicalFrequencyLayout.layout().addWidget(self.musicOutputSpectrogramCanvas)

        self.ecgTimeLayout.layout().addWidget(self.ecgTimeInputCanvas)
        self.ecgTimeLayout.layout().addWidget(self.ecgTimeOutputCanvas)
        self.ecgFrequencyLayout.layout().addWidget(self.ecgFrequencyCanvas)
        self.ecgFrequencyLayout.layout().addWidget(self.ecgInputSpectrogramCanvas)
        self.ecgFrequencyLayout.layout().addWidget(self.ecgOutputSpectrogramCanvas)

    def chooseSmoothingWindow(self):
        window_type = self.smothingComboBox.currentText()

        # Clear the existing plot
        self.smoothingWindowCanvas.figure.clf()

        # Create a new subplot
        ax = self.smoothingWindowCanvas.figure.add_subplot(111)

        # Plot the selected window
        if window_type == "Rectangle":
            window = np.ones(100)  # Replace with your desired length
        elif window_type == "Hamming":
            window = np.hamming(100)  # Replace with your desired length
        elif window_type == "Hanning":
            window = np.hanning(100)  # Replace with your desired length
        elif window_type == "Gaussian":
            # You can customize the parameters of the Gaussian window
            center = 50
            window = np.exp(-(np.arange(100) - center) ** 2 / (2 * self.sigma ** 2))
        # Add more conditions for other window types if needed
        ax.plot(window)
        ax.set_title(f"{window_type} Window")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        # Redraw the canvas
        self.smoothingWindowCanvas.draw()

    def updateGaussian(self):
        self.sigma=self.stdSlider.value()
        self.chooseSmoothingWindow()

    def loadFile(self):

        #check current tab
        currentTabindex=self.tabWidget.currentIndex()

        if currentTabindex == 1 or currentTabindex == 2 or currentTabindex == 3:    
        # Open a file dialog and get the selected file name
            fileName, _ = QFileDialog.getOpenFileName(
                None, "Load WAV File", "", "WAV Files (*.wav);;All Files (*)"
            )
        

            if fileName:
                self.file_path = fileName  # Store the file path
                print(fileName)
                self.media_player = QMediaPlayer()

                # Create a QUrl from the file name
                url = QtCore.QUrl.fromLocalFile(fileName)

                # Create QMediaContent from the QUrl
                content = QMediaContent(url)

                # Set the media content for the media player
                self.media_player.setMedia(content)

                # Reading file data
                yaxis, self.sample_rate = librosa.load(self.file_path)


                # Time domain coordinates
                self.time_domain_X_coordinates = np.arange(len(yaxis)) / self.sample_rate
                self.time_domain_Y_coordinates = yaxis

        else :
                file_path, _ = QFileDialog.getOpenFileName(None, "Open ECG File", "", "ECG Files (*.hea)")

                if file_path:
                    # Read the record
                    record = wfdb.rdrecord(file_path[:-4])
                    self.sample_rate = record.fs

                    # Extract time-domain values
                    time_values = record.p_signal[:, 0]  # Assuming the first column represents time-domain values
                    
                    # Time domain coordinates
                    self.time_domain_Y_coordinates = time_values
                    self.time_domain_X_coordinates = np.arange(len(self.time_domain_Y_coordinates)) / record.fs

        # Store x and y coordinates as a list of tuples
        self.xy_coordinates = list(zip(self.time_domain_X_coordinates, self.time_domain_Y_coordinates))

        # FFT coordinates
        self.fft_result = fft(self.time_domain_Y_coordinates)
        self.frequencies = np.fft.fftfreq(len(self.fft_result), 1 / self.sample_rate)

        #check current tab
        currentTabindex=self.tabWidget.currentIndex()
        self.output = False
        
        if currentTabindex == 1:
            self.plotTimeDomain(self.unifromTimeInputCanvas,self.xy_coordinates)
            self.plotFrequencyDomain(self.unifromFrequencyCanvas,self.frequencies,np.abs(self.fft_result))
            
            self.temparray1=self.fft_result.copy()
            self. plotSpectrogram( self.unifromInputSpectrogramCanvas,yaxis,self.sample_rate)

        elif currentTabindex == 2:
            self.plotTimeDomain(self.animalTimeInputCanvas, self.xy_coordinates)
            self.plotFrequencyDomain(self.animalFrequencyCanvas,self.frequencies,np.abs(self.fft_result))
            self.temparray2=self.fft_result.copy()
            self. plotSpectrogram( self.animalInputSpectrogramCanvas,yaxis,self.sample_rate)


        elif currentTabindex == 3:
            self.plotTimeDomain(self.musicTimeInputCanvas, self.xy_coordinates)
            self.plotFrequencyDomain(self.musicFrequencyCanvas,self.frequencies,np.abs(self.fft_result))
            self.temparray3=self.fft_result.copy()
            self. plotSpectrogram( self.musicInputSpectrogramCanvas,yaxis,self.sample_rate)

        elif currentTabindex == 4:
            self.plotTimeDomain(self.ecgTimeInputCanvas, self.xy_coordinates)
            self.temparray4 = self.fft_result.copy()
            self.plotFrequencyDomain(self.ecgFrequencyCanvas, self.frequencies, np.abs(self.fft_result))
            self.plotSpectrogram(self.ecgInputSpectrogramCanvas, self.time_domain_Y_coordinates, record.fs)

    def plotTimeDomain(self, canvas, xy_coordinates):
        x, y = zip(*xy_coordinates)
        canvas.upload_data(x,y)

    def plotFrequencyDomain(self, canvas,x,y):
        ax = canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.plot(np.abs(x),y)
        ax.set_title("Frequency Domain Plot")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        canvas.draw()   

    def uniformfrecuencyupdate(self,value,customrange):
        max_frequency = np.max(self.frequencies)
        sliderrange =int(max_frequency/10)
        lower_bound = customrange * sliderrange
        upper_bound = (customrange + 1) * sliderrange 

        for i, frequency in enumerate(self.frequencies):

                if lower_bound <= np.abs(frequency) <= upper_bound:
                    self.temparray1[i] = self.fft_result[i].copy()
                    self.temparray1[i] = self.temparray1[i]*(value / 10)

        self.plotFrequencyDomain(self.unifromFrequencyCanvas,self.frequencies, np.abs(self.temparray1))  
        self.add_shaded_region(self.unifromFrequencyCanvas, self.frequencies, np.abs(self.fft_result), np.abs(self.temparray1)) 
        self.output = True 
        self.output_time_domain_Y_coordinates=self.calcAndPlotIfft(self.temparray1,self.unifromTimeOutputCanvas,self.time_domain_X_coordinates,self.unifromTimeInputCanvas)  
        self.plotSpectrogram(self.unifromOutputSpectrogramCanvas,self.output_time_domain_Y_coordinates,self.sample_rate) 
    
    def animalfrequencycomp(self, value, range):

        for i, frequency in enumerate(self.frequencies):
                if (
                        (range == 1 and 2500 > np.abs(frequency) > 500) or
                        (range == 2 and (1000 > np.abs(frequency) > 125)) or
                        (range == 3 and (4000 >np.abs(frequency) > 1000)) or
                        (range == 4 and (16000 > np.abs(frequency) > 4000))
                    ):
                        self.temparray2[i] = self.fft_result[i].copy()
                        self.temparray2[i] = self.temparray2[i]*(value / 10)

        self.plotFrequencyDomain(self.animalFrequencyCanvas,self.frequencies, np.abs(self.temparray2))   
        self.add_shaded_region(self.animalFrequencyCanvas, self.frequencies, np.abs(self.fft_result), np.abs(self.temparray2))  
        self.output = True    
        self.output_time_domain_Y_coordinates=self.calcAndPlotIfft(self.temparray2,self.animalTimeOutputCanvas,self.time_domain_X_coordinates,self.animalTimeInputCanvas)  
        self.plotSpectrogram(self.animalOutputSpectrogramCanvas,self.output_time_domain_Y_coordinates,self.sample_rate)     

    def musicfrequencycomp(self, value, range):
        for i, frequency in enumerate(self.frequencies):
                if (
                        (range == 1 and 1000 > np.abs(frequency) > 0) or
                        (range == 2 and (7000 > np.abs(frequency) > 500)) or
                        (range == 3 and (4000 >np.abs(frequency) > 180)) or
                        (range == 4 and (15000 > np.abs(frequency) > 2000))
                    ):
                        self.temparray3[i] = self.fft_result[i].copy()
                        self.temparray3[i] = self.temparray3[i]*(value / 10)
            
        self.plotFrequencyDomain(self.musicFrequencyCanvas,self.frequencies, np.abs(self.temparray3)) 
        self.add_shaded_region(self.musicFrequencyCanvas, self.frequencies, np.abs(self.fft_result), np.abs(self.temparray3))
        self.output = True  
        self.output_time_domain_Y_coordinates=self.calcAndPlotIfft(self.temparray3,self.musicTimeOutputCanvas,self.time_domain_X_coordinates,self.musicTimeInputCanvas)  
        self.plotSpectrogram(self.musicOutputSpectrogramCanvas,self.output_time_domain_Y_coordinates,self.sample_rate)

    def ecgfrequencycomp(self, value, range):
        for i, frequency in enumerate(self.frequencies):
                if (
                        (range == 1 and(50 > np.abs(frequency) > 0)) or
                        (range == 2 and (100 > np.abs(frequency) > 50)) or
                        (range == 3 and (450 >np.abs(frequency) > 50)) or
                        (range == 4 and (8 > np.abs(frequency) > 0))
                    ):
                        self.temparray4[i] = self.fft_result[i].copy()
                        self.temparray4[i] = self.temparray4[i]*(value / 10)
            
        self.plotFrequencyDomain(self.ecgFrequencyCanvas,self.frequencies, np.abs(self.temparray4)) 
        self.add_shaded_region(self.ecgFrequencyCanvas, self.frequencies, np.abs(self.fft_result), np.abs(self.temparray4)) 
        self.output = True 
        self.output_time_domain_Y_coordinates=self.calcAndPlotIfft(self.temparray4,self.ecgTimeOutputCanvas,self.time_domain_X_coordinates,self.ecgTimeInputCanvas)  
        self.plotSpectrogram(self.ecgOutputSpectrogramCanvas,self.output_time_domain_Y_coordinates,self.sample_rate)   
    
    def add_shaded_region(self, canvas, x_values, y_values_original, y_values_modified):
        # Assuming canvas is a matplotlib axes
        ax = canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)    
        # Plot the original data with a specific color
        ax.plot(np.abs(x_values), y_values_original, label='Original Data', color='red')
        # Plot the modified data with a different color
        ax.plot(np.abs(x_values), y_values_modified, label='Modified Data', color='blue')
        ax.set_title("Frequency Domain Plot with Shaded Region")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
        canvas.draw()

    def playPauseLoadedSound(self):
        if hasattr(self, 'file_path') and self.file_path:
            if not hasattr(self, 'media_player') and self.number_of_output_file == 0:
                media_content = QMediaContent(QtCore.QUrl.fromLocalFile(self.file_path))
                self.media_player = QMediaPlayer()
                self.media_player.setMedia(media_content)
            if self.media_player.state() == QMediaPlayer.PlayingState:
                self.media_player.pause()
            else:
                self.media_player.play()

    def rewindLoadedSound(self):
        if hasattr(self, 'file_path') and self.file_path:
            if not hasattr(self, 'media_player') and self.number_of_output_file == 0:
                media_content = QMediaContent(QtCore.QUrl.fromLocalFile(self.file_path))
                self.media_player = QMediaPlayer()
                self.media_player.setMedia(media_content)
            print("Before setPosition:", self.media_player.state())
            self.media_player.setPosition(0)              
            print("Before setPosition:", self.media_player.state())

    def calcAndPlotIfft(self,freq_mag,canvas,time,input_canvas):
        #y=fft.ifft2(freq_mag)
        y=np.fft.ifft(freq_mag)
        y=y.astype(np.float32)
        xy_coordinates = list(zip(time, y))
        self.plotTimeDomain(canvas, xy_coordinates) 
        self.convertToWavFile(y,self.sample_rate)
        input_canvas.link_with_me(canvas)
        input_canvas.rewind()
        self.rewindLoadedSound()
        self.playPauseLoadedSound()
        return y

    def  plotSpectrogram(self, canvas,y,sr):
        ax = canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        hl = 512 # number of samples per time-step in spectrogram
        hi = 128 # Height of image
        wi = 384 # Width of image
        max_frequency = np.max(self.frequencies)
        window = y[0:wi*hl]
        S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=hi, fmax = max_frequency,hop_length=hl)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax = max_frequency, ax=ax)
        canvas.draw()

    def toggle_spectrogram(self, state, canvas,canvas2):
        # Toggle the visibility of the stored AxesImage object based on checkbox state
            if state == Qt.Checked:
                # Clear the canvas
                canvas.figure.clear()
                canvas.figure.add_subplot(111)
                canvas.draw()

                canvas2.figure.clear()
                canvas2.figure.add_subplot(111)
                canvas2.draw()
            else:
                # Plot the data again
                self.plotSpectrogram(canvas, self.time_domain_Y_coordinates,self.sample_rate)
                if self.output:
                    self.plotSpectrogram(canvas2, self.output_time_domain_Y_coordinates ,self.sample_rate)

    def convertToWavFile(self,data,fs):
        self.number_of_output_file=self.number_of_output_file+1
        self.name_of_output="out"+str(self.number_of_output_file)+".wav"
        wavf.write(self.name_of_output, fs, data)
        if self.number_of_output_file>1:
             os.remove("out" + str(self.number_of_output_file - 1) + ".wav")
        
    def rewindLoadedSound(self):       
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # Construct the file path
        file_path= os.path.join(current_directory, self.name_of_output)
        # Create the QMediaContent object
        media_content = QMediaContent(QtCore.QUrl.fromLocalFile(file_path))
        self.media_player = QMediaPlayer()
        self.media_player.setMedia(media_content)
        self.media_player.setPosition(0)
    
    def resetNumber_of_output_file(self):
        self.number_of_output_file=0

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_8.setText(_translate("MainWindow", "Gaussian STD"))
        self.label.setText(_translate("MainWindow", "Smoothing Window"))
        self.smothingComboBox.setItemText(0, _translate("MainWindow", "Rectangle"))
        self.smothingComboBox.setItemText(1, _translate("MainWindow", "Gaussian"))
        self.smothingComboBox.setItemText(2, _translate("MainWindow", "Hamming"))
        self.smothingComboBox.setItemText(3, _translate("MainWindow", "Hanning"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.smoothingWindowTab), _translate("MainWindow", "Smoothing Window"))
        self.zoomInButton_1.setText(_translate("MainWindow", "Zoom In"))
        self.zoomOutButton_1.setText(_translate("MainWindow", "Zoom Out"))
        self.pausePlayButton_1.setText(_translate("MainWindow", "Pause/Play"))
        self.label_5.setText(_translate("MainWindow", "Speed"))
        self.rewindButton_1.setText(_translate("MainWindow", "Rewind"))
        self.checkBox_1.setText(_translate("MainWindow", "Hide Spectrogram"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.unifromRangeTab), _translate("MainWindow", "Uniform Range"))
        self.label_12.setText(_translate("MainWindow", "Elephant"))
        self.label_13.setText(_translate("MainWindow", "Bat"))
        self.label_15.setText(_translate("MainWindow", "Eagle"))
        self.label_16.setText(_translate("MainWindow", "Whale"))
        self.zoomInButton_2.setText(_translate("MainWindow", "Zoom In"))
        self.checkBox_2.setText(_translate("MainWindow", "Hide Spectrogram"))
        self.zoomOutButton_2.setText(_translate("MainWindow", "Zoom Out"))
        self.label_6.setText(_translate("MainWindow", "Speed"))
        self.rewindButton_2.setText(_translate("MainWindow", "Rewind"))
        self.pausePlayButton_2.setText(_translate("MainWindow", "Pause/Play"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.animalTab), _translate("MainWindow", "Animal Sounds"))
        self.label_18.setText(_translate("MainWindow", "Drums"))
        self.label_20.setText(_translate("MainWindow", "Xylophone"))
        self.label_21.setText(_translate("MainWindow", "Occordion"))
        self.label_22.setText(_translate("MainWindow", "Cymbal"))
        self.checkBox_3.setText(_translate("MainWindow", "Hide Spectrogram"))
        self.label_14.setText(_translate("MainWindow", "Speed"))
        self.zoomInButton_3.setText(_translate("MainWindow", "Zoom In"))
        self.rewindButton_3.setText(_translate("MainWindow", "Rewind"))
        self.zoomOutButton_3.setText(_translate("MainWindow", "Zoom Out"))
        self.pausePlayButton_3.setText(_translate("MainWindow", "Pause/Play"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.musicTab), _translate("MainWindow", "Musical Instruments"))
        self.zoomOutButton_4.setText(_translate("MainWindow", "Zoom Out"))
        self.zoomInButton_4.setText(_translate("MainWindow", "Zoom In"))
        self.checkBox_4.setText(_translate("MainWindow", "Hide Spectrogram"))
        self.label_4.setText(_translate("MainWindow", "Normal ECG"))
        self.label_2.setText(_translate("MainWindow", "Sinus Rhythm"))
        self.label_3.setText(_translate("MainWindow", "Atrial"))
        self.label_7.setText(_translate("MainWindow", "Myocardial"))
        self.label_19.setText(_translate("MainWindow", "Speed"))
        self.rewindButton_4.setText(_translate("MainWindow", "Rewind"))
        self.pausePlayButton_4.setText(_translate("MainWindow", "Pause/Play"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ecgTab), _translate("MainWindow", "ECG"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))


def main():
    app = QtWidgets.QApplication(sys.argv)  # Create the application instance
    MainWindow = QtWidgets.QMainWindow()  # Create the main window
    ui = Ui_MainWindow()  # Create an instance of the UI class
    ui.setupUi(MainWindow)  # Set up the UI for the main window
    MainWindow.show()  # Display the main window
    sys.exit(app.exec_())  # Run the application event loop


main()