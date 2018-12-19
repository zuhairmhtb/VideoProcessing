from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2, random
import numpy as np


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

class ImageDisplayWidget(QWidget):
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

        image = QImage(image.data, width, height, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def get_qimage_gray(self, image: np.ndarray):
        height, width = image.shape
        #bytesPerLine = 3 * width


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
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

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
class AnnotationViewCanvas(QGraphicsView):
    def __init__(self, mouse_press_fn=None, mouse_move_fn=None):
        QGraphicsView.__init__(self, None)
        self.mouse_press_event = mouse_press_fn
        self.mouse_move_event = mouse_move_fn
    def mouseMoveEvent(self, event):
        if not (self.mouse_move_event is None):
            self.mouse_move_event(event)
        super(AnnotationViewCanvas, self).mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if not (self.mouse_press_event is None):
            self.mouse_press_event(event)
        super(AnnotationViewCanvas, self).mousePressEvent(event)

class AnnotationWindow(QWidget):
    def __init__(self, parent=None, image_path="D:\\pic.jpg"):
        QWidget.__init__(self, parent)
        self.image_path = image_path
        self.raw_img = cv2.imread(self.image_path)

        self.reset_window_btn = QPushButton("Reset Window")
        self.reset_window_btn.clicked.connect(self.reset_window)

        self.mark_options = ['point', 'line']
        self.save_as_options = ['Whole Image', 'Masked Image', 'Individual']
        self.mark_option_wid = QComboBox()
        for k in self.mark_options:
            self.mark_option_wid.addItem(k)
        self.create_segment_btn = QPushButton('Create Segment')
        self.create_segment_btn.clicked.connect(self.create_segment)

        self.info_display_container = QGroupBox('Info')
        self.info_display_lay = QFormLayout()
        self.info_display_container.setLayout(self.info_display_lay)

        self.total_segments_created = QLabel('0')
        self.info_display_lay.addRow(QLabel('Segments Created'), self.total_segments_created)

        self.select_segment_combobox = QComboBox()
        self.select_segment_combobox.addItem('All')

        self.view_selected_segment_btn = QPushButton('View Segment')
        self.view_selected_segment_btn.clicked.connect(self.view_segment)
        self.delete_selected_segment_btn = QPushButton('Delete Segment')
        self.delete_selected_segment_btn.clicked.connect(self.delete_segment)
        self.save_as_optn_combobox = QComboBox()
        for k in self.save_as_options:
            self.save_as_optn_combobox.addItem(k)


        self.segment_edit_container = QGroupBox('Edit Segment')
        self.segment_edit_container_lay = QHBoxLayout()
        self.segment_edit_container.setLayout(self.segment_edit_container_lay)
        self.segment_edit_container_lay.addWidget(self.view_selected_segment_btn)
        self.segment_edit_container_lay.addWidget(self.delete_selected_segment_btn)
        self.segment_edit_container_lay.addWidget(self.save_as_optn_combobox)

        self.segment_name_inp = QLabel('N/A')
        self.segment_label_inp = QTextEdit()
        self.occluded_checkbox = QCheckBox("Occluded")
        self.update_label_btn = QPushButton('Update')
        self.update_label_btn.clicked.connect(self.update_label_fn)


        self.label_container = QGroupBox("Information")
        self.label_lay = QFormLayout()
        self.label_container.setLayout(self.label_lay)

        self.label_lay.addRow(QLabel('Name of the Segment:'), self.segment_name_inp)
        self.label_lay.addRow(QLabel('Attributes/Labels:'), self.segment_label_inp)
        self.label_lay.addRow(self.occluded_checkbox, self.update_label_btn)



        self.btn_lay = QVBoxLayout()
        self.btn_wid = QGroupBox('Controls')
        self.btn_wid.setLayout(self.btn_lay)
        self.btn_lay.addWidget(self.reset_window_btn)
        self.btn_lay.addWidget(self.mark_option_wid)
        self.btn_lay.addWidget(self.create_segment_btn)
        self.btn_lay.addWidget(self.info_display_container)
        self.btn_lay.addWidget(self.select_segment_combobox)
        self.btn_lay.addWidget(self.segment_edit_container)
        self.btn_lay.addWidget(self.label_container)
        #self.btn_lay.addWidget(self.occluded_checkbox)


        self.gv = AnnotationViewCanvas(mouse_move_fn=self.mouse_move_event, mouse_press_fn=self.mouse_press_event)
        self.gv.setFixedWidth(self.raw_img.shape[1])
        self.gv.setFixedHeight(self.raw_img.shape[0])

        self.scene = QGraphicsScene(self)

        self.gv.setScene(self.scene)
        self.gv.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        lay = QHBoxLayout(self)
        lay.addWidget(self.btn_wid)
        lay.addWidget(self.gv)

        self.reset_window()

        #self.gv.setMouseTracking(True)

    def mouse_move_event(self, event):
        pass
    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            position = event.pos()
            self.set_point(position)
            if str(self.mark_option_wid.currentText()).lower() == self.mark_options[0]:
                self.draw_point(coordinate=position)
            elif str(self.mark_option_wid.currentText()).lower() == self.mark_options[1] and len(self.current_point_list) >= 2:
                self.draw_line(self.current_point_list[-2], self.current_point_list[-1])

    def reset_image_display(self):
        img_arr = self.raw_img.copy()
        img_qimage = QImage(img_arr.data, img_arr.shape[1], img_arr.shape[0], img_arr.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        self.display_image = QPixmap(img_qimage)
        self.scene.clear()
        self.p_item = self.scene.addPixmap(self.display_image)
    def reset_window(self):
        self.reset_image_display()
        self.reset_points()
        self.reset_available_segment_box()
        self.reset_label_info_view()
    def reset_points(self):
        self.current_point_list = []
        self.all_segment_list = []

        self.total_segments_created.setText('0')

    def reset_label_info_view(self):
        self.segment_label_inp.setText("")
        self.segment_name_inp.setText("")
        self.occluded_checkbox.setChecked(False)
    def reset_available_segment_box(self):
        # Reset ComboBox for available segments
        self.select_segment_combobox.clear()
        self.select_segment_combobox.addItem('All')
        for i in range(len(self.all_segment_list)):
            self.select_segment_combobox.addItem('Segment ' + str(i))
    def set_point(self, position):
        self.current_point_list.append(position)


    def draw_point(self, coordinate, color=QColor(255, 255, 255)):
        if not (coordinate is None):
            point_obj = QPointF(coordinate)
            current_image = self.display_image.toImage()
            current_image.setPixelColor(point_obj.x(), point_obj.y(), color)

            self.display_image = QPixmap(current_image)
            self.scene.clear()
            self.p_item = self.scene.addPixmap(self.display_image)
    def draw_line(self, x, y, color=QColor(255, 255, 255)):
        line = QGraphicsLineItem(QLineF(x, y), self.p_item)
        line.setPen(QPen(color, 1))
        #line.setFlag(QGraphicsItem.ItemIsMovable, True)
        #line.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.gv.fitInView(self.scene.sceneRect())

    def draw_segment(self, points):
        if len(points) > 1:
            for i in range(len(points)):
                if i != len(points) - 1:
                    self.draw_line(points[i], points[i+1])
                else:
                    self.draw_line(points[i], points[0])
    def create_segment(self):
        if len(self.current_point_list) > 1:
            self.draw_segment(self.current_point_list)
            default_segment_name = "Segment " + str(len(self.all_segment_list))
            default_label = ['Unknown']
            default_occluded = False
            self.all_segment_list.append([self.current_point_list.copy(), default_segment_name, default_label, default_occluded])
            self.current_point_list = []
            self.total_segments_created.setText(str(len(self.all_segment_list)))
            self.reset_available_segment_box()

    def view_segment(self):
        self.reset_image_display()
        self.reset_label_info_view()
        if self.select_segment_combobox.currentText().lower() == 'all':
            for i in range(len(self.all_segment_list)):
                self.draw_segment(self.all_segment_list[i][0])
        else:
            segment_no = int(self.select_segment_combobox.currentText().split(" ")[1])
            self.draw_segment(self.all_segment_list[segment_no][0])
            self.segment_name_inp.setText(self.all_segment_list[segment_no][1])
            labels_str = ''
            for i in range(len(self.all_segment_list[segment_no][2])):
                labels_str += str(self.all_segment_list[segment_no][2][i])
                if i < len(self.all_segment_list[segment_no][2]) - 1:
                    labels_str += ','
            self.segment_label_inp.setText(labels_str)
            self.occluded_checkbox.setChecked(self.all_segment_list[segment_no][3])

    def delete_segment(self):
        if self.select_segment_combobox.currentText().lower() == 'all':
            self.reset_window()

        else:
            segment_no = int(self.select_segment_combobox.currentText().split(" ")[1])
            self.all_segment_list.pop(segment_no)
        self.reset_available_segment_box()

    def update_label_fn(self):
        current_name = str(self.segment_name_inp.text())
        current_labels_str = str(self.segment_label_inp.toPlainText())
        current_occluded = self.occluded_checkbox.isChecked()
        print('Current name: ' + current_name + ', Labels: ' + str(current_labels_str) + ', Occluded: ' + str(current_occluded))
        if 'segment' in current_name.lower() and len(current_labels_str) > 0:
            segment_no = int(current_name.split(" ")[1])
            print('Segment No. ' + str(segment_no))
            self.all_segment_list[segment_no][3] = current_occluded
            self.all_segment_list[segment_no][2] = current_labels_str.split(",")

    def set_annotation_image(self, img):
        self.raw_img = img.copy()
        self.gv.setFixedWidth(self.raw_img.shape[1])
        self.gv.setFixedHeight(self.raw_img.shape[0])
        self.reset_window()
