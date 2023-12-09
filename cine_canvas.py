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
        self.y_min = 1
        self.y_max = 1
        self.y_min_original = 0
        self.y_max_original = 0
        self.shift_amount=50
        self.time_of_drawing=1
        self.timer =  QtCore.QTimer()
        self.zoomed_by=1
        self.is_played=True
        self.linked_canvas=self
        self.linked=False
        
    def upload_data(self,x_data,y_data):
        self.x_data=x_data
        self.y_data=y_data
        self.data_size=len(y_data)
        self.y_min = min(self.y_data)
        self.y_max = max(self.y_data)
        self.y_min_original = float(min(self.y_data))
        self.y_max_original =  float(max(self.y_data))
        self.data_plotted=0
        self.timer.timeout.connect(self.dynamic_plot)
        self.timer.start(self.time_of_drawing)
        self.is_played=True

    def dynamic_plot(self):
        if(not self.is_played):
            return
        if(self.window_size+self.shift_amount > self.data_size):
            self.data_plotted=0
        y_draw=self.y_data[self.data_plotted:self.window_size+self.data_plotted]
        x_draw=self.x_data[self.data_plotted:self.window_size+self.data_plotted]
        self.data_plotted=self.data_plotted+self.shift_amount
        self.axes.clear()
        self.axes.set_ylim(self.y_min, self.y_max)
        line,=self.axes.plot(x_draw, y_draw, 'b')
        self.draw()
        
    def link_with_me(self,canvas):
        canvas.y_min=self.y_min
        canvas.y_max=self.y_max
        canvas.data_plotted=self.data_plotted
        canvas.shift_amount=self.shift_amount
        canvas.time_of_drawing=self.time_of_drawing
        canvas.window_size=self.window_size
        canvas.is_played=self.is_played
        canvas.zoomed_by=self.zoomed_by
        self.linked_canvas=canvas
        self.linked=True

    def reset(self):
        self.x_data.clear()
        self.y_data.clear()
        self.timer.stop()
        self.axes.clear()
        if(self.linked):
            self.linked_canvas.reset()

    def zoom_in(self):
        self.y_min*=0.9
        self.y_max*=0.9
        #self.window_size=int(self.window_size*0.95)
        self.zoomed_by=float(self.zoomed_by*0.9)
        if(self.linked):
            self.linked_canvas.zoom_in()

    def zoom_out(self):
        self.y_min = self.y_min*1.1
        self.y_max = self.y_max*1.1
        #self.window_size=int(self.window_size/0.95)
        self.zoomed_by=float(self.zoomed_by*1.1)
        if(self.linked):
            self.linked_canvas.zoom_out()

    def increase_speed(self):
        if(self.linked):
            self.linked_canvas.increase_speed()
        self.time_of_drawing = float(self.time_of_drawing * 0.8)
        self.shift_amount=2*self.shift_amount
        if(self.is_played):
            self.timer.start(self.time_of_drawing)
        
    def decrease_speed(self):
        if(self.linked):
            self.linked_canvas.decrease_speed()
        self.time_of_drawing = float(self.time_of_drawing * 1/0.8)
        self.shift_amount=int(0.5*self.shift_amount)
        if(self.is_played):
            self.timer.start(self.time_of_drawing)

    def control_speed(self,x):
        if(self.linked):
            self.linked_canvas.control_speed(x)
        self.time_of_drawing=5-(x/10)
        if(self.is_played):
            self.timer.start(self.time_of_drawing)
            self.timer.timeout.connect(self.dynamic_plot)

    def pause(self):
        self.is_played=False
        self.timer.stop()
        if(self.linked):
            self.linked_canvas.pause()

    def play(self):
        self.is_played=True
        self.timer.start(self.time_of_drawing)
        if(self.linked):
            self.linked_canvas.play()

    def play_or_pause(self):
        if(self.is_played):
            self.pause()
        else:
            self.play()



            


