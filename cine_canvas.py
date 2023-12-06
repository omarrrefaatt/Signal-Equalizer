import sys
import matplotlib
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout,QHBoxLayout
import pandas as pd
import tkinter as tk
import math
matplotlib.use('Qt5Agg')
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.x_data=list()
        self.y_data=list()
        self.data_size=0
        self.window_size=1000
        self.data_plotted=0
        self.y_min = 0
        self.y_max = 0
        self.y_min_original = 0
        self.y_max_original = 0
        self.shift_amount=50
        self.time_of_drawing=1
        self.timer =  QtCore.QTimer()
    def upload_data(self,x_data,y_data):
        self.x_data=x_data
        self.y_data=y_data
        self.data_size=len(y_data)
        #print(self.y_data)
        self.y_min = min(self.y_data)
        self.y_max = max(self.y_data)
        self.y_min_original = float(min(self.y_data))
        self.y_max_original =  float(max(self.y_data))
        self.data_plotted=0
        self.timer.timeout.connect(self.dynamic_plot)
        self.timer.start(self.time_of_drawing)
    def dynamic_plot(self):
        if(self.window_size+self.shift_amount > self.data_size):
            self.data_plotted=0
        y_draw=self.y_data[self.data_plotted:self.window_size+self.data_plotted]
        x_draw=self.x_data[self.data_plotted:self.window_size+self.data_plotted]
        self.data_plotted=self.data_plotted+self.shift_amount
        self.axes.clear()
        self.axes.set_ylim(self.y_min, self.y_max)
        line,=self.axes.plot(x_draw, y_draw, 'b')
        self.draw()
        
            


