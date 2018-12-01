from PyQt5.QtWidgets import QVBoxLayout, QWidget, QTabWidget, QSizePolicy
import cv2, random
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar, FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
class MyTabWidget(QWidget):
    """
    This class creates a tab layout with the provided tab size, tab names(list) and tab layouts(list)
    """
    def __init__(self, parent, tab_width, tab_height, tab_names, tab_layouts):
        super(QWidget, self).__init__(parent)
        self.tab_width = tab_width
        self.tab_height = tab_height
        self.tab_names = tab_names
        self.tab_layouts = tab_layouts
        self.parent = parent

        self.layout = QVBoxLayout(self)
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab_w = [QWidget() for _ in range(len(tab_names))]

        self.tabs.resize(300, 200)

        # Add tabs
        for i in range(len(self.tab_names)):
            self.tabs.addTab(self.tab_w[i], tab_names[i])

        # Create first tab
        for i in range(len(self.tab_names)):
            self.tab_w[i].layout = tab_layouts[i]
            self.tab_w[i].setLayout(self.tab_w[i].layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

class ImageDisplayWidget(QtWidgets.QWidget):
    def __init__(self, parent, default_image_path, display_image_width, display_image_height):
        super().__init__(parent)
        self.default_img = cv2.imread(default_image_path)
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.img_width = display_image_width
        self.img_height = display_image_height

        img_np = np.array(self.default_img)
        if img_np.shape[0] != self.img_width or img_np.shape[1] != self.img_height:
            img_np = cv2.resize(img_np, (self.img_width, self.img_height))
        self.image = self.get_qimage(img_np)

    def get_qimage_rgb(self, image: np.ndarray):
        height, width, colors = image.shape
        #bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data, width, height, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def get_qimage_gray(self, image: np.ndarray):
        height, width = image.shape
        #bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data, width, height, QImage.Format_RGB32)

        image = image.rgbSwapped()
        return image
    def get_qimage(self, image:np.ndarray):
        if image.shape[0] != self.img_width or image.shape[1] != self.img_height:
            image = cv2.resize(image, (self.img_width, self.img_height))
        if len(image.shape) == 3:
            return self.get_qimage_rgb(image)
        elif len(image.shape) == 2:
            image_rgb = np.repeat(image[:,:,np.newaxis], 3, axis=2)
            return self.get_qimage_rgb(np.asarray(image_rgb))
        else:
            return None
    def update_image(self, image_data):
        self.image = self.get_qimage(image_data)
        #print("Current image size: " + str(self.image.size()))
        #print("Current frame size: " + str(self.size()))
        if self.size() != self.image.size():
            self.setFixedSize(self.image.size())
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

class GraphDisplayWidget:
    def __init__(self, parent, fig_num, default_image_path=None):
        default_img_path = default_image_path
        self.fig = plt.figure(num=fig_num)

        self.canvas = FigureCanvas(self.fig)


        self.toolbar = NavigationToolbar(self.canvas, parent)
        if not (default_image_path is None):
            self.plot([np.array(cv2.imread(default_image_path))])


    def get_figure_canvas(self):
        return self.canvas
    def get_figure_toolbar(self):
        return self.toolbar

    def plot(self, images, histogram=False):
        self.fig.clear()
        if not histogram:
            for i in range(len(images)):
                ax = self.fig.add_subplot(1, len(images), i+1)
                if (not (images[i] is None)) and (images[i].size > 0):
                    ax.imshow(images[i])
        else:
            for i in range(len(images)):
                ax = self.fig.add_subplot(1, len(images), i+1)
                if not (images[i] is None):
                    ax.hist(images[i].ravel(), 256, [0, 256])

        self.canvas.draw()
