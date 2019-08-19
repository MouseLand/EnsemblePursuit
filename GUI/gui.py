import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow,QVBoxLayout, QWidget
import sys
import os
from scipy import io

class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(25, 25, 1800, 1000)

        self.path_U = '/home/maria/Documents/EnsemblePursuit/SAND9/experiments/natimg2800_M170717_MP034_2017-09-11.mat_U_ep_pytorch.npy'
        self.path_V = '/home/maria/Documents/EnsemblePursuit/SAND9/experiments/natimg2800_M170717_MP034_2017-09-11.mat_V_ep_pytorch.npy'
        self.path_X = '/home/maria/Documents/EnsemblePursuit/SAND9/data/natimg2800_M170717_MP034_2017-09-11.mat'

        cwidget = QtGui.QWidget()
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        self.win = pg.GraphicsLayoutWidget()
        # self.win.move(600, 0)
        # self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 0, 0, 50, 30)
        layout = self.win.ci.layout

        #V plot
        self.pfull = self.win.addPlot(title="FULL VIEW", row=0, col=2, rowspan=1, colspan=3)
        self.pfull.setMouseEnabled(x=False, y=False)
        self.V = pg.ImageItem(autoDownsample=True)
        self.pfull.addItem(self.V)
        self.pfull.hideAxis('left')

        #U plot
        self.p0 = self.win.addPlot(titile="U",row=1, col=0, rowspan=2, colspan=1, lockAspect=True)
        self.U = pg.ImageItem(autoDownsample=True)
        self.p0.addItem(self.U)
        self.p0.setAspectLocked(ratio=1)
        self.plot_U()

        #Selected X's
        self.p1 = self.win.addPlot(title="X",row=1, col=2, colspan=3,
                                   rowspan=3, invertY=True, padding=0)
        self.p1.setMouseEnabled(x=False, y=False)
        self.X = pg.ImageItem(autoDownsample=False)
        self.p1.hideAxis('left')
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.p1.addItem(self.X)
        self.plot_X()
        print(self.path_X)

    def plot_V(self):
        data = np.load(self.path_V)[:1000, :1000].T
        self.V.setImage(data)
        self.show()
        self.win.show()

    def plot_U(self):
        U=np.load(self.path_U)[:100,:100]
        self.U.setImage(U)


    def plot_X(self):
        data = io.loadmat(self.path_X)
        X = data['stim'][0]['resp'][0][:1000,:1000]
        self.X.setImage(X)
        return 0

app=QApplication(sys.argv)
window=MainW()
window.plot_V()
sys.exit(app.exec_())

