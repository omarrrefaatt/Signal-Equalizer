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
        MainWindow.resize(1117, 815)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_smoothing = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_smoothing.sizePolicy().hasHeightForWidth())
        self.frame_smoothing.setSizePolicy(sizePolicy)
        self.frame_smoothing.setMaximumSize(QtCore.QSize(16777215, 200))
        self.frame_smoothing.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_smoothing.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_smoothing.setObjectName("frame_smoothing")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_smoothing)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.stdSlider = QtWidgets.QSlider(self.frame_smoothing)
        self.stdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.stdSlider.setObjectName("stdSlider")
        self.gridLayout_3.addWidget(self.stdSlider, 2, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame_smoothing)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 2, 0, 1, 1)
        self.smothingComboBox = QtWidgets.QComboBox(self.frame_smoothing)
        self.smothingComboBox.setMaximumSize(QtCore.QSize(500, 25))
        self.smothingComboBox.setObjectName("smothingComboBox")
        self.smothingComboBox.addItem("")
        self.smothingComboBox.addItem("")
        self.smothingComboBox.addItem("")
        self.smothingComboBox.addItem("")
        self.gridLayout_3.addWidget(self.smothingComboBox, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.frame_smoothing)
        self.label.setMaximumSize(QtCore.QSize(150, 20))
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_smoothing, 2, 2, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.zoom_in_button = QtWidgets.QPushButton(self.frame_2)
        self.zoom_in_button.setMaximumSize(QtCore.QSize(60, 16777215))
        self.zoom_in_button.setObjectName("zoom_in_button")
        self.gridLayout_2.addWidget(self.zoom_in_button, 0, 1, 1, 1)
        self.spectro_checkBox = QtWidgets.QCheckBox(self.frame_2)
        self.spectro_checkBox.setMaximumSize(QtCore.QSize(140, 16777215))
        self.spectro_checkBox.setObjectName("spectro_checkBox")
        self.gridLayout_2.addWidget(self.spectro_checkBox, 0, 3, 1, 1)
        self.zoom_out_button = QtWidgets.QPushButton(self.frame_2)
        self.zoom_out_button.setMaximumSize(QtCore.QSize(60, 16777215))
        self.zoom_out_button.setObjectName("zoom_out_button")
        self.gridLayout_2.addWidget(self.zoom_out_button, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 4, 1, 1)
        self.pause_play_button = QtWidgets.QPushButton(self.frame_2)
        self.pause_play_button.setMaximumSize(QtCore.QSize(400, 16777215))
        self.pause_play_button.setObjectName("pause_play_button")
        self.gridLayout_2.addWidget(self.pause_play_button, 0, 6, 1, 1)
        self.rewind_button = QtWidgets.QPushButton(self.frame_2)
        self.rewind_button.setMaximumSize(QtCore.QSize(80, 16777215))
        self.rewind_button.setObjectName("rewind_button")
        self.gridLayout_2.addWidget(self.rewind_button, 0, 9, 1, 1)
        self.speed_up_button = QtWidgets.QPushButton(self.frame_2)
        self.speed_up_button.setMaximumSize(QtCore.QSize(60, 16777215))
        self.speed_up_button.setObjectName("speed_up_button")
        self.gridLayout_2.addWidget(self.speed_up_button, 0, 11, 1, 1)
        self.speed_down_button = QtWidgets.QPushButton(self.frame_2)
        self.speed_down_button.setMaximumSize(QtCore.QSize(60, 16777215))
        self.speed_down_button.setObjectName("speed_down_button")
        self.gridLayout_2.addWidget(self.speed_down_button, 0, 12, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 10, 1, 1)
        self.gridLayout.addWidget(self.frame_2, 1, 2, 1, 3)
        self.frame_normal = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_normal.sizePolicy().hasHeightForWidth())
        self.frame_normal.setSizePolicy(sizePolicy)
        self.frame_normal.setMaximumSize(QtCore.QSize(16777215, 200))
        self.frame_normal.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_normal.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_normal.setObjectName("frame_normal")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_normal)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.modeSlider_4 = QtWidgets.QSlider(self.frame_normal)
        self.modeSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.modeSlider_4.setObjectName("modeSlider_4")
        self.gridLayout_9.addWidget(self.modeSlider_4, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame_normal)
        self.label_4.setObjectName("label_4")
        self.gridLayout_9.addWidget(self.label_4, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_9, 0, 3, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.modeSlider_2 = QtWidgets.QSlider(self.frame_normal)
        self.modeSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.modeSlider_2.setObjectName("modeSlider_2")
        self.gridLayout_7.addWidget(self.modeSlider_2, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame_normal)
        self.label_2.setObjectName("label_2")
        self.gridLayout_7.addWidget(self.label_2, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_7, 0, 1, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.modeSlider_3 = QtWidgets.QSlider(self.frame_normal)
        self.modeSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.modeSlider_3.setObjectName("modeSlider_3")
        self.gridLayout_8.addWidget(self.modeSlider_3, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_normal)
        self.label_3.setObjectName("label_3")
        self.gridLayout_8.addWidget(self.label_3, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_8, 0, 2, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.modeSlider_1 = QtWidgets.QSlider(self.frame_normal)
        self.modeSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.modeSlider_1.setObjectName("modeSlider_1")
        self.gridLayout_6.addWidget(self.modeSlider_1, 0, 0, 1, 1)
        self.label_1 = QtWidgets.QLabel(self.frame_normal)
        self.label_1.setMaximumSize(QtCore.QSize(123234, 1233))
        self.label_1.setObjectName("label_1")
        self.gridLayout_6.addWidget(self.label_1, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_normal, 2, 4, 1, 1)
        self.frame_uni = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_uni.sizePolicy().hasHeightForWidth())
        self.frame_uni.setSizePolicy(sizePolicy)
        self.frame_uni.setMaximumSize(QtCore.QSize(16777215, 200))
        self.frame_uni.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_uni.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_uni.setObjectName("frame_uni")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_uni)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.uniforRangeSlider_1 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_1.setObjectName("uniforRangeSlider_1")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_1)
        self.uniforRangeSlider_2 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_2.setObjectName("uniforRangeSlider_2")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_2)
        self.uniforRangeSlider_3 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_3.setObjectName("uniforRangeSlider_3")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_3)
        self.uniforRangeSlider_4 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_4.setObjectName("uniforRangeSlider_4")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_4)
        self.uniforRangeSlider_5 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_5.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_5.setObjectName("uniforRangeSlider_5")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_5)
        self.uniforRangeSlider_6 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_6.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_6.setObjectName("uniforRangeSlider_6")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_6)
        self.uniforRangeSlider_7 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_7.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_7.setObjectName("uniforRangeSlider_7")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_7)
        self.uniforRangeSlider_8 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_8.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_8.setObjectName("uniforRangeSlider_8")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_8)
        self.uniforRangeSlider_9 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_9.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_9.setObjectName("uniforRangeSlider_9")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_9)
        self.uniforRangeSlider_10 = QtWidgets.QSlider(self.frame_uni)
        self.uniforRangeSlider_10.setOrientation(QtCore.Qt.Vertical)
        self.uniforRangeSlider_10.setObjectName("uniforRangeSlider_10")
        self.horizontalLayout.addWidget(self.uniforRangeSlider_10)
        self.gridLayout.addWidget(self.frame_uni, 2, 3, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setMaximumSize(QtCore.QSize(16777215, 500))
        self.tabWidget.setObjectName("tabWidget")
        self.smoothingWindowTab = QtWidgets.QWidget()
        self.smoothingWindowTab.setObjectName("smoothingWindowTab")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.smoothingWindowTab)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.smoothingWindowLayout = QtWidgets.QVBoxLayout()
        self.smoothingWindowLayout.setObjectName("smoothingWindowLayout")
        self.gridLayout_10.addLayout(self.smoothingWindowLayout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.smoothingWindowTab, "")
        self.unifromRangeTab = QtWidgets.QWidget()
        self.unifromRangeTab.setObjectName("unifromRangeTab")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.unifromRangeTab)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.uniformTimeLayout = QtWidgets.QVBoxLayout()
        self.uniformTimeLayout.setObjectName("uniformTimeLayout")
        self.gridLayout_11.addLayout(self.uniformTimeLayout, 0, 0, 1, 1)
        self.uniformFrequencyLayout = QtWidgets.QVBoxLayout()
        self.uniformFrequencyLayout.setObjectName("uniformFrequencyLayout")
        self.gridLayout_11.addLayout(self.uniformFrequencyLayout, 0, 1, 1, 1)
        self.tabWidget.addTab(self.unifromRangeTab, "")
        self.animalTab = QtWidgets.QWidget()
        self.animalTab.setObjectName("animalTab")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.animalTab)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.animalTimeLayout = QtWidgets.QVBoxLayout()
        self.animalTimeLayout.setObjectName("animalTimeLayout")
        self.gridLayout_12.addLayout(self.animalTimeLayout, 0, 0, 1, 1)
        self.animalFrequencyLayout = QtWidgets.QVBoxLayout()
        self.animalFrequencyLayout.setObjectName("animalFrequencyLayout")
        self.gridLayout_12.addLayout(self.animalFrequencyLayout, 0, 1, 1, 1)
        self.tabWidget.addTab(self.animalTab, "")
        self.musicTab = QtWidgets.QWidget()
        self.musicTab.setObjectName("musicTab")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.musicTab)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.musicalTimeLayout = QtWidgets.QVBoxLayout()
        self.musicalTimeLayout.setObjectName("musicalTimeLayout")
        self.gridLayout_13.addLayout(self.musicalTimeLayout, 0, 0, 1, 1)
        self.musicalFrequencyLayout = QtWidgets.QVBoxLayout()
        self.musicalFrequencyLayout.setObjectName("musicalFrequencyLayout")
        self.gridLayout_13.addLayout(self.musicalFrequencyLayout, 0, 1, 1, 1)
        self.tabWidget.addTab(self.musicTab, "")
        self.ecgTab = QtWidgets.QWidget()
        self.ecgTab.setObjectName("ecgTab")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.ecgTab)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.ecgTimeLayout = QtWidgets.QVBoxLayout()
        self.ecgTimeLayout.setObjectName("ecgTimeLayout")
        self.gridLayout_14.addLayout(self.ecgTimeLayout, 0, 0, 1, 1)
        self.ecgFrequencyLayout = QtWidgets.QVBoxLayout()
        self.ecgFrequencyLayout.setObjectName("ecgFrequencyLayout")
        self.gridLayout_14.addLayout(self.ecgFrequencyLayout, 0, 1, 1, 1)
        self.tabWidget.addTab(self.ecgTab, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 5)
        self.gridLayout_5.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1117, 26))
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
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.frame_smoothing.setVisible(True)
        self.frame_normal.setVisible(False)
        self.frame_uni.setVisible(False)
        
        self.tabWidget.currentChanged.connect(self.change_appearance)
        self.tabWidget.currentChanged.connect(self.change_text_lable)


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
            
        self.spectro_checkBox.stateChanged.connect(lambda state, canvas=self.unifromInputSpectrogramCanvas,canvas2=self.unifromOutputSpectrogramCanvas: self.toggle_spectrogram(state, canvas,canvas2))
        
        self.actionOpen.triggered.connect(self.loadFile)  # Connect to loadWavFile method

        self.pause_play_button.clicked.connect(self.playPauseLoadedSound)  # Connect to playPauseLoadedSound method
        self.pause_play_button.clicked.connect(self.musicTimeInputCanvas.play_or_pause) 


        self.rewind_button.clicked.connect(self.rewindLoadedSound) 
        self.rewind_button.clicked.connect(self.animalTimeInputCanvas.rewind)
        
        

        self.smothingComboBox.currentIndexChanged.connect(self.chooseSmoothingWindow)

        self.stdSlider.setMinimum(1)
        self.stdSlider.setMaximum(20)
        self.stdSlider.setValue(5)  # Initial value of sigma
        self.stdSlider.valueChanged.connect(self.updateGaussian)

        self.sigma = self.stdSlider.value()


        self.zoom_in_button.clicked.connect(self.unifromTimeInputCanvas.zoom_in)
        self.zoom_out_button.clicked.connect(self.unifromTimeInputCanvas.zoom_out)
    
        self.number_of_output_file=0
        self.tabWidget.currentChanged.connect(self.resetNumber_of_output_file)
        self.name_of_output=""

    def change_appearance(self):
            self.current_tab_index = self.tabWidget.currentIndex()
            if self.current_tab_index == 0:
                self.frame_smoothing.setVisible(True)
                self.frame_normal.setVisible(False)
                self.frame_uni.setVisible(False)
            elif self.current_tab_index == 1:
                self.frame_smoothing.setVisible(False)
                self.frame_normal.setVisible(False)
                self.frame_uni.setVisible(True)
            elif self.current_tab_index == 2:
                self.frame_smoothing.setVisible(False)
                self.frame_normal.setVisible(True)
                self.frame_uni.setVisible(False)
            elif self.current_tab_index == 3:
                self.frame_smoothing.setVisible(False)
                self.frame_normal.setVisible(True)
                self.frame_uni.setVisible(False)
            elif self.current_tab_index == 4:
                self.frame_smoothing.setVisible(False)
                self.frame_normal.setVisible(True)
                self.frame_uni.setVisible(False)

    def change_text_lable(self):
        self.current_tab_index = self.tabWidget.currentIndex()
        if self.current_tab_index == 2:
            self.label_1.setText("Animal 1")
            self.label_2.setText("Animal 2")
            self.label_3.setText("Animal 3")
            self.label_4.setText("Animal 4")
        elif self.current_tab_index == 3:
            self.label_1.setText("Music 1")
            self.label_2.setText("Music 2")
            self.label_3.setText("Music 3")
            self.label_4.setText("Music 4")
        elif self.current_tab_index == 4:
            self.label_1.setText("ECG 1")
            self.label_2.setText("ECG 2")
            self.label_3.setText("ECG 3")
            self.label_4.setText("ECG 4")

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
                        (range == 1 and 250 > np.abs(frequency) > 0) or
                        (range == 2 and (1000 > np.abs(frequency) > 250)) or
                        (range == 3 and (4000 >np.abs(frequency) > 1000)) or
                        (range == 4 and (16000 > np.abs(frequency) > 4000))
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
            self.media_player.setPosition(0)
            self.media_player.play()
            
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
                canvas.hide()

                canvas2.hide()
            else:
                # Plot the data again
                canvas.show()
                canvas2.show()

    def convertToWavFile(self,data,fs):
        self.number_of_output_file=self.number_of_output_file+1
        self.name_of_output="out"+str(self.number_of_output_file)+".wav"
        wavf.write(self.name_of_output, fs, data)
        if self.number_of_output_file>1:
             os.remove("out" + str(self.number_of_output_file - 1) + ".wav")
        
    def resetNumber_of_output_file(self):
        self.number_of_output_file=0

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.smoothingWindowTab), _translate("MainWindow", "Smoothing Window"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.unifromRangeTab), _translate("MainWindow", "Uniform Range"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.animalTab), _translate("MainWindow", "Animal Sounds"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.musicTab), _translate("MainWindow", "Musical Instruments"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.ecgTab), _translate("MainWindow", "ECG"))
        self.label_8.setText(_translate("MainWindow", "Gaussian STD"))
        self.smothingComboBox.setItemText(0, _translate("MainWindow", "Rectangle"))
        self.smothingComboBox.setItemText(1, _translate("MainWindow", "Gaussian"))
        self.smothingComboBox.setItemText(2, _translate("MainWindow", "Hamming"))
        self.smothingComboBox.setItemText(3, _translate("MainWindow", "Hanning"))
        self.label.setText(_translate("MainWindow", "Smoothing Window"))
        self.zoom_in_button.setText(_translate("MainWindow", "Zoom +"))
        self.spectro_checkBox.setText(_translate("MainWindow", "Hide Spectrogram"))
        self.zoom_out_button.setText(_translate("MainWindow", "Zoom -"))
        self.pause_play_button.setText(_translate("MainWindow", "PAUSE/PLAY"))
        self.rewind_button.setText(_translate("MainWindow", "Rewind"))
        self.speed_up_button.setText(_translate("MainWindow", "Speed +"))
        self.speed_down_button.setText(_translate("MainWindow", "Speed -"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_1.setText(_translate("MainWindow", "TextLabel"))
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