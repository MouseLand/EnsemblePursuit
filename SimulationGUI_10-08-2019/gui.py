import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QVBoxLayout, QWidget
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

        # V plot
        self.pfull = self.win.addPlot(title="FULL VIEW", row=0, col=2, rowspan=1, colspan=3)
        self.pfull.setMouseEnabled(x=False, y=False)
        self.V = pg.ImageItem(autoDownsample=True)
        self.pfull.addItem(self.V)
        self.pfull.hideAxis('left')

        # U plot
        self.p0 = self.win.addPlot(titile="U", row=1, col=0, rowspan=2, colspan=1, lockAspect=True)
        self.U = pg.ImageItem(autoDownsample=True)
        self.p0.addItem(self.U)
        self.p0.setAspectLocked(ratio=1)
        self.plot_U()
        self.U_ROI = pg.InfiniteLine(movable=True)
        self.p0.addItem(self.U_ROI)

        def getcoordinates(roi):
            val = roi.value()
            print(val)
            print(np.floor(val))
            non_z_U = np.nonzero(self.U_dat[int(np.floor(val))].flatten())
            print(non_z_U)
            X = self.X_dat[non_z_U[0], :]
            print(X)
            self.X.setImage(X)
            self.pfull.plot(self.V_dat[int(np.floor(val)), :].flatten())

        self.U_ROI.sigPositionChangeFinished.connect(getcoordinates)

        # Selected X's
        self.p1 = self.win.addPlot(title="X", row=1, col=2, colspan=3,
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
        self.V_dat = np.load(self.path_V).T
        print(self.V_dat.shape)
        self.pfull.plot(self.V_dat[0, :])
        self.show()
        self.win.show()

    def plot_U(self):
        self.U_dat = np.load(self.path_U)[:100, :100]
        self.U.setImage(self.U_dat)

    def plot_X(self):
        data = io.loadmat(self.path_X)
        self.X_dat = data['stim'][0]['resp'][0][:1000, :1000]
        self.X.setImage(self.X_dat)
        return 0


app = QApplication(sys.argv)
window = MainW()
window.plot_V()
sys.exit(app.exec_())
