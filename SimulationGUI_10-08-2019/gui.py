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
        self.view=pg.GraphicsView()
        self.path='/home/maria/Documents/EnsemblePursuit/SAND9/experiments/natimg2800_M170717_MP034_2017-09-11.mat_U_ep_pytorch.npy'

    def set_layout(self):
        self.l0 = QtGui.QGridLayout()
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(500,500)
        self.l0.addWidget(self.win, 0, 0, 50, 30)


    def plot_U(self):
        data=np.load(self.path)[:,0]
        self.plot_u = self.win.addPlot(row=1, col=0, rowspan=2, colspan=1, lockAspect=True)
        print(data)
        print(self.plot_u)
        self.plot_u.plot(data)
        #self.win.scene()
        #pg.show()
        self.view.setCentralItem(self.plot_u)
        self.setCentralWidget(self.view)
        self.show()
        #self.win.show()

    def plot_U_im(self):
        data=np.load(self.path)[:1000,:1000].T
        print(data.shape)
        imv = pg.ImageView()
        self.setCentralWidget(imv)
        imv.setImage(data)
        self.show()
        #self.l0.addWidget(self.image_view, 0, 0)
        #self.show()

    def plot_square(self):
        data=np.load(self.path)[:10000,0]
        print(data.shape)
        data=data.reshape((100,100))
        imv = pg.ImageView()
        self.setCentralWidget(imv)
        imv.setImage(data)
        self.show()

    def plot_squares_scatter(self):
        data=np.load(self.path).T
        view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        self.setCentralWidget(view)
        w = view.addPlot()
        s = pg.ScatterPlotItem(pxMode=False)   ## Set pxMode=False to allow spots to transform with the view
        squares = []
        for i in range(10):
            for j in range(10):
                squares.append({'pos': (1e-6*i, 1e-6*j), 'size': 1e-6, 'pen': {'color': 'w', 'width': 2}, 'brush':pg.intColor(i*10+j, 100),'data':data[i,:][:10000].reshape((100,100))})
        s.addPoints(squares)
        w.addItem(s)
        self.show()

    def plot_squares_layout(self):
        data=np.load(self.path).T
        view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        self.setCentralWidget(view)
        for i in range(10):
            for j in range(10):
                imv = pg.ImageView()
                imv.setImage(data[i,:][:10000].reshape((100,100)))
                view.addItem(imv)
                view.nextRow()
                view.nextColumn()
        self.show()




app=QApplication(sys.argv)
window=MainW()
window.set_layout()
#window.plot_U()
#window.plot_U_im()
#window.plot_square()
#window.plot_squares_scatter()
window.plot_squares_layout()
sys.exit(app.exec_())
