import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow,QVBoxLayout, QWidget
import sys
import os

class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(25, 25, 1800, 1000)
        self.setWindowTitle("EnsemblePursuit")

    def set_layout(self):
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(1000,600)

    def plot_U(self):
        path='/home/maria/Documents/EnsemblePursuit/SAND9/experiments/natimg2800_M170717_MP034_2017-09-11.mat_U_ep_pytorch.npy'
        data=np.load(path)[:,0]
        plot_u = self.win.addPlot(title="U",col=0,row=1)
        print(data)
        print(plot_u)
        plot_u.plot(data)
        self.win.show()

app=QApplication(sys.argv)
window=MainW()
window.set_layout()
window.plot_U()
window.show()
sys.exit(app.exec_())
