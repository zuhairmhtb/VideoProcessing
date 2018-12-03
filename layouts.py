from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QScrollArea,\
    QWidget, QAction, QTabWidget, QLabel, QSlider, QCheckBox, QTextEdit, QLineEdit, QComboBox, QFormLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
import cv2
from collections import OrderedDict
from MyNet.image_segmentation.video_processing_software.widgets import ImageDisplayWidget, GraphDisplayWidget


class CommonLayout(QHBoxLayout):
    def __init__(self, parent):
        super(QHBoxLayout, self).__init__(parent)
        self.parent = parent
        self.control_widget = QGroupBox("Control Panel", self.parent)
        self.control_widget_scrollbar = QScrollArea()

        self.control_widget_layout = QVBoxLayout(self.parent)
        self.control_widget.setLayout(self.control_widget_layout)
        self.control_widget_scrollbar.setWidget(self.control_widget)
        self.control_widget_scrollbar.setWidgetResizable(True)

        self.addWidget(self.control_widget_scrollbar)


        self.view_widget = QGroupBox("View Panel", self.parent)
        self.view_widget_scrollbar = QScrollArea()
        self.view_widget_layout = QVBoxLayout(self.parent)
        self.view_widget.setLayout(self.view_widget_layout)
        self.view_widget_scrollbar.setWidget(self.view_widget)
        self.view_widget_scrollbar.setWidgetResizable(True)
        self.addWidget(self.view_widget_scrollbar)
class WebcamPlayerLayout(CommonLayout):
    def __init__(self, parent, default_frame_width=200, default_frame_height=200):
        super(WebcamPlayerLayout, self).__init__(parent)
        self.my_parent = parent
        self.default_frame_image_path = "D:\\default.png"
        self.default_frame_width = default_frame_width
        self.default_frame_height = default_frame_height
        self.stream_button_widget, self.stream_button_widget_layout, self.stream_buttons = self.create_stream_button_widget()
        self.brightness_ctrl_widget, self.brightness_ctrl_widget_layout, self.brightness_control_containers, self.brightness_control_sliders = self.create_brightness_control_widget()
        self.auto_adjustment_widget, self.auto_adjustment_layout, self.auto_adjustment_containers, self.auto_adjustment_control_input = self.create_auto_adjustment_params()
        self.frame_process_widget, self.frame_process_widget_layout, self.frame_process_containers, self.frame_process_control_inputs = self.create_frame_processing_params()
        self.edge_detect_widget, self.edge_detect_layout, self.edge_detect_containers, self.edge_detect_controls = self.create_edge_detection_params()
        self.display_frame_widget, self.display_frame_layout, self.display_frame_containers, self.display_frame_controls = self.create_view_frames()

    def create_stream_button_widget(self):
        """
        This function generates the neccessary buttons required for controlling webcam stream
        :param button_name: The name of available buttons(list)
        :return: The button widget, layout and the buttons
        """
        stream_button_widgets = QGroupBox("Stream options", self.my_parent)
        stream_button_widgets_layout = QHBoxLayout(self.my_parent)
        stream_button_widgets.setLayout(stream_button_widgets_layout)
        buttons = {}
        for b in ["play", "pause", "stop"]:
            but = QPushButton(b.upper(), self.my_parent)
            buttons[b] = but
            stream_button_widgets_layout.addWidget(but)
        self.control_widget_layout.addWidget(stream_button_widgets)
        return stream_button_widgets, stream_button_widgets_layout, buttons
    def create_brightness_control_widget(self):
        """
        This function creates the options for controlling brightness, hue, contrast and saturation of the webcam feed
        :return: The widgets containing parameters, the layout, the parameter widgets and the parameters
        """
        brightness_control_widget = QGroupBox("Brightness Control", self.my_parent)
        brightness_control_widget_layout = QHBoxLayout(self.my_parent)
        brightness_control_widget.setLayout(brightness_control_widget_layout)
        controls = {}
        control_containers = {}
        for b in ["brightness", "contrast", "sharpness"]:
            slider_container = QGroupBox(b.upper(), self.my_parent)
            slider_container_layout = QHBoxLayout(self.my_parent)
            slider_container.setLayout(slider_container_layout)
            ctrl = QSlider(Qt.Horizontal)
            ctrl.setFocusPolicy(Qt.StrongFocus)
            ctrl.setTickPosition(QSlider.TicksBothSides)
            ctrl.setTickInterval(10)
            ctrl.setSingleStep(1)
            slider_container_layout.addWidget(ctrl)
            controls[b] = ctrl
            control_containers[b] = [slider_container, slider_container_layout]
            brightness_control_widget_layout.addWidget(slider_container)
        self.control_widget_layout.addWidget(brightness_control_widget)
        return brightness_control_widget, brightness_control_widget_layout, control_containers, controls

    def create_auto_adjustment_params(self):
        """
        This function creates the auto adjustment parameters for controlling dynamic preprocessing of image
        :return: The widget containing parameters, the layout, the parameter widgets and the parameters
        """
        auto_adjustment_widget = QGroupBox("Auto Adjustment Parameters",self.my_parent)
        auto_adjustment_widget_layout = QVBoxLayout(self.my_parent)
        auto_adjustment_widget.setLayout(auto_adjustment_widget_layout)

        controls = {}
        control_containers = {}

        enable_checkbox_container = QGroupBox("Auto Adjustment", self.my_parent)
        enable_checkbox_container_layout = QHBoxLayout(self.my_parent)
        enable_checkbox_container.setLayout(enable_checkbox_container_layout)

        enable_checkbox = QCheckBox("Enable", self.my_parent)
        enable_checkbox.setChecked(False)
        enable_checkbox_container_layout.addWidget(enable_checkbox)

        auto_adjustment_duration_label = QLabel("Duration(ms)", self.my_parent)
        auto_adjustment_duration_widget = QLineEdit(self.my_parent)
        enable_checkbox_container_layout.addWidget(auto_adjustment_duration_label)
        enable_checkbox_container_layout.addWidget(auto_adjustment_duration_widget)

        auto_adjustment_widget_layout.addWidget(enable_checkbox_container)
        controls["auto adjustment"] = [enable_checkbox, auto_adjustment_duration_widget]
        control_containers["auto adjustment"] = [enable_checkbox_container, enable_checkbox_container_layout]

        edge_size_adjust_container = QGroupBox("Edge Size Adjustment", self.my_parent)
        edge_size_adjust_container_layout = QHBoxLayout(self.my_parent)
        edge_size_adjust_container.setLayout(edge_size_adjust_container_layout)

        edge_size_min_pct_label = QLabel("Minimum Size(%)", self.my_parent)
        edge_size_min_pct_inp = QLineEdit(self.my_parent)
        edge_size_max_pct_label = QLabel("Maximum Size(%)", self.my_parent)
        edge_size_max_pct_inp = QLineEdit(self.my_parent)
        edge_size_step_pct_label = QLabel("Step Size(%)", self.my_parent)
        edge_size_step_pct_inp = QLineEdit(self.my_parent)

        edge_size_adjust_container_layout.addWidget(edge_size_min_pct_label)
        edge_size_adjust_container_layout.addWidget(edge_size_min_pct_inp)
        edge_size_adjust_container_layout.addWidget(edge_size_max_pct_label)
        edge_size_adjust_container_layout.addWidget(edge_size_max_pct_inp)
        edge_size_adjust_container_layout.addWidget(edge_size_step_pct_label)
        edge_size_adjust_container_layout.addWidget(edge_size_step_pct_inp)

        auto_adjustment_widget_layout.addWidget(edge_size_adjust_container)
        controls["edge size"] = [edge_size_min_pct_inp, edge_size_max_pct_inp, edge_size_step_pct_inp]
        control_containers["edge size"] = [edge_size_adjust_container, edge_size_adjust_container_layout]



        self.control_widget_layout.addWidget(auto_adjustment_widget)
        return auto_adjustment_widget, auto_adjustment_widget_layout, control_containers, controls


    def create_frame_processing_params(self):
        """
        This function creates the controls neccessary for controlling the sizes of frames(edge, raw) being processed
        :return: The widget containing parameters, the layout, the parameter widgets and the parameters
        """

        frame_processing_widgets = QGroupBox("Frame Size Adjustment", self.my_parent)
        frame_processing_widgets_layout = QVBoxLayout(self.my_parent)
        frame_processing_widgets.setLayout(frame_processing_widgets_layout)

        controls = {}
        control_containers = {}
        for b in ["original", "edge"]:
            control_widget = QGroupBox(b.upper(), self.my_parent)
            control_widget_layout = QHBoxLayout(self.my_parent)
            control_widget.setLayout(control_widget_layout)

            inp_w_label = QLabel("Width", self.my_parent)
            inp_w_box = QLineEdit(self.my_parent)
            inp_h_label = QLabel("Height", self.my_parent)
            inp_h_box = QLineEdit(self.my_parent)
            controls[b] = [inp_w_box, inp_h_box]
            control_widget_layout.addWidget(inp_w_label)
            control_widget_layout.addWidget(inp_w_box)
            control_widget_layout.addWidget(inp_h_label)
            control_widget_layout.addWidget(inp_h_box)
            frame_processing_widgets_layout.addWidget(control_widget)
            control_containers[b] =[control_widget, control_widget_layout]
        self.control_widget_layout.addWidget(frame_processing_widgets)
        return frame_processing_widgets, frame_processing_widgets_layout, control_containers, controls

    def create_edge_detection_params(self):
        """
        This function creates the parameters for controlling edge and contour detection
        :return:
        """
        main_widget = QGroupBox("Edge & Contour Adjustment", self.my_parent)
        main_widget_layout = QVBoxLayout(self.my_parent)
        main_widget.setLayout(main_widget_layout)

        controls = {}
        control_containers = {}

        edge_widget = QGroupBox("Edge Detection", self.my_parent)
        edge_layout = QHBoxLayout(self.my_parent)
        edge_widget.setLayout(edge_layout)
        edge_algo_label = QLabel("Algorithm", self.my_parent)
        edge_algo_inp = QComboBox(self.my_parent)
        edge_algo_inp.addItem("canny")
        edge_algo_inp.addItem("laplacian")
        edge_algo_inp.addItem("sobel")
        edge_thresh_label = QLabel("Thresholds", self.my_parent)
        edge_thresh_inp = QLineEdit(self.my_parent)
        edge_layout.addWidget(edge_algo_label)
        edge_layout.addWidget(edge_algo_inp)
        edge_layout.addWidget(edge_thresh_label)
        edge_layout.addWidget(edge_thresh_inp)
        controls["edge detection"] = [edge_algo_inp, edge_thresh_inp]
        control_containers["edge detection"] = [edge_widget, edge_layout]
        main_widget_layout.addWidget(edge_widget)


        erosion_widget = QGroupBox("Erosion & Dilation", self.my_parent)
        erosion_widget_layout = QHBoxLayout()
        erosion_widget.setLayout(erosion_widget_layout)

        erosiona_widget = QGroupBox("Erosion 1", self.my_parent)
        erosiona_layout = QVBoxLayout()
        erosiona_widget.setLayout(erosiona_layout)
        erosiona_label = QLabel("Erosion 1 Kernel", self.my_parent)
        erosiona_slider = QSlider(Qt.Horizontal)
        erosiona_slider.setMinimum(0)
        erosiona_slider.setMaximum(11)
        erosiona_slider.setSingleStep(1)
        erosiona_slider.setTickInterval(1)
        erosiona_slider.setValue(0)
        erosiona_layout.addWidget(erosiona_label)
        erosiona_layout.addWidget(erosiona_slider)
        erosion_widget_layout.addWidget(erosiona_widget)

        dilationa_widget = QGroupBox("Dilation 1", self.my_parent)
        dilationa_layout = QVBoxLayout()
        dilationa_widget.setLayout(dilationa_layout)
        dilationa_label = QLabel("Kernel", self.my_parent)
        dilationa_slider = QSlider(Qt.Horizontal)
        dilationa_slider.setMinimum(0)
        dilationa_slider.setMaximum(11)
        dilationa_slider.setSingleStep(1)
        dilationa_slider.setTickInterval(1)
        dilationa_slider.setValue(0)
        dilationa_layout.addWidget(dilationa_label)
        dilationa_layout.addWidget(dilationa_slider)
        erosion_widget_layout.addWidget(dilationa_widget)

        erosionb_widget = QGroupBox("Erosion 2", self.my_parent)
        erosionb_layout = QVBoxLayout()
        erosionb_widget.setLayout(erosionb_layout)
        erosionb_label = QLabel("Erosion 2 Kernel", self.my_parent)
        erosionb_slider = QSlider(Qt.Horizontal)
        erosionb_slider.setMinimum(0)
        erosionb_slider.setMaximum(11)
        erosionb_slider.setSingleStep(1)
        erosionb_slider.setTickInterval(1)
        erosionb_slider.setValue(0)
        erosionb_layout.addWidget(erosionb_label)
        erosionb_layout.addWidget(erosionb_slider)
        erosion_widget_layout.addWidget(erosionb_widget)

        dilationb_widget = QGroupBox("Dilation 2", self.my_parent)
        dilationb_layout = QVBoxLayout()
        dilationb_widget.setLayout(dilationb_layout)
        dilationb_label = QLabel("Dilation 2 Kernel", self.my_parent)
        dilationb_slider = QSlider(Qt.Horizontal)
        dilationb_slider.setMinimum(0)
        dilationb_slider.setMaximum(11)
        dilationb_slider.setSingleStep(1)
        dilationb_slider.setTickInterval(1)
        dilationb_slider.setValue(0)
        dilationb_layout.addWidget(dilationb_label)

        dilationb_layout.addWidget(dilationb_slider)
        erosion_widget_layout.addWidget(dilationb_widget)
        controls["erosion"] = [erosiona_slider, dilationa_slider, erosionb_slider, dilationb_slider]
        control_containers["erosion"] = [erosion_widget, erosion_widget_layout]
        main_widget_layout.addWidget(erosion_widget)

        contour_widget = QGroupBox("Contour Detection", self.my_parent)
        contour_layout = QHBoxLayout(self.my_parent)
        contour_widget.setLayout(contour_layout)
        contour_enable_view = QCheckBox("Enable View", self.my_parent)
        contour_min_box_size_label = QLabel("Minimum Size(Bounding Box)", self.my_parent)
        contour_min_box_size_inp = QLineEdit(self.my_parent)
        contour_layout.addWidget(contour_enable_view)
        contour_layout.addWidget(contour_min_box_size_label)
        contour_layout.addWidget(contour_min_box_size_inp)
        controls["contour detection"] = [contour_enable_view, contour_min_box_size_inp]
        control_containers["contour detection"] = [contour_widget, contour_layout]
        main_widget_layout.addWidget(contour_widget)

        self.control_widget_layout.addWidget(main_widget)
        return main_widget, main_widget_layout, control_containers, controls

    def create_view_frames(self):
        """
        This function creates the GUI where frames will be displayed
        :return:
        """
        main_widget = QGroupBox("Display", self.my_parent)
        main_layout = QVBoxLayout(self.my_parent)
        main_widget.setLayout(main_layout)

        top_widget = QGroupBox("", self.my_parent)
        top_layout = QHBoxLayout(self.my_parent)
        top_widget.setLayout(top_layout)

        bottom_widget = QGroupBox("", self.my_parent)
        bottom_layout = QHBoxLayout(self.my_parent)
        bottom_widget.setLayout(bottom_layout)
        main_layout.addWidget(top_widget)
        main_layout.addWidget(bottom_widget)

        control_widgets = {}
        control_widgets["top"] = [top_widget, top_layout]
        control_widgets["bottom"] = [bottom_widget, bottom_layout]

        controls = {}
        original_frame_widget = ImageDisplayWidget(self.my_parent, self.default_frame_image_path, self.default_frame_width, self.default_frame_height)
        adjusted_frame_widget = ImageDisplayWidget(self.my_parent, self.default_frame_image_path, self.default_frame_width, self.default_frame_height)
        gray_frame_widget = ImageDisplayWidget(self.my_parent, self.default_frame_image_path, self.default_frame_width, self.default_frame_height)
        edge_frame_widget = ImageDisplayWidget(self.my_parent, self.default_frame_image_path, self.default_frame_width, self.default_frame_height)
        top_layout.addWidget(original_frame_widget)
        top_layout.addWidget(adjusted_frame_widget)
        bottom_layout.addWidget(gray_frame_widget)
        bottom_layout.addWidget(edge_frame_widget)
        controls["top"] = [original_frame_widget, adjusted_frame_widget]
        controls["bottom"] = [gray_frame_widget, edge_frame_widget]

        self.view_widget_layout.addWidget(main_widget)

        return main_widget, main_layout, control_widgets, controls


class ContourObjectDetectionModule:
    def __init__(self, parent, fig_num, fig_width, fig_height, default_image=None):
        self.default_image = default_image
        self.object_display_canvas = GraphDisplayWidget(parent, fig_num, default_image_path=self.default_image)

        self.control_widget = QGroupBox("Contour Object Detection Parameters", parent)
        self.control_widget_layout = QVBoxLayout(parent)
        self.control_widget.setLayout(self.control_widget_layout)

        self.display_amount_widget = QGroupBox("No. of Contours to display", parent)
        self.display_amount_widget_layout = QHBoxLayout()
        self.display_amount_widget.setLayout(self.display_amount_widget_layout)
        self.enable_module_checkbox = QCheckBox("Enable Module", parent)
        self.display_control_amount_label = QLabel("Amount", parent)
        self.display_control_amount_inp = QLineEdit(parent)
        self.display_amount_widget_layout.addWidget(self.display_control_amount_label)
        self.display_amount_widget_layout.addWidget(self.display_control_amount_inp)
        self.display_amount_widget_layout.addWidget(self.enable_module_checkbox)
        self.control_widget_layout.addWidget(self.display_amount_widget)

        self.contour_navigationbar_widget = QGroupBox("Contour Plot Navaigation", parent)
        self.contour_navigationbar_widget_layout = QHBoxLayout()
        self.contour_navigationbar_widget.setLayout(self.contour_navigationbar_widget_layout)
        self.contour_navigationbar_widget_layout.addWidget(self.object_display_canvas.get_figure_toolbar())
        self.control_widget_layout.addWidget(self.contour_navigationbar_widget)

        self.controls = {"contour amount": self.display_control_amount_inp, "enable": self.enable_module_checkbox,
                         "navigation": self.object_display_canvas.get_figure_toolbar()}

        self.view_widget_scrollbar = QScrollArea(parent)
        self.view_widget_scrollbar.setMinimumWidth(fig_width)
        self.view_widget_scrollbar.setMinimumHeight(fig_height)
        self.view_widget_contour = QGroupBox("Identified Objects", self.view_widget_scrollbar)


        self.view_widget_layout = QHBoxLayout()
        self.view_widget_contour.setLayout(self.view_widget_layout)



        self.view_widget_scrollbar.setWidget(self.view_widget_contour)
        self.view_widget_scrollbar.setWidgetResizable(True)
        self.view_widget_layout.addWidget(self.object_display_canvas.get_figure_canvas())

        self.view_controls = {"object": self.object_display_canvas}



    def get_control_container(self):
        return self.control_widget
    def get_controls(self):
        return self.controls
    def get_view_container(self):
        return self.view_widget_scrollbar

    def get_view_controls(self):
        return self.view_controls


class ThresholdingModule:
    def __init__(self, parent, fig_num_thresh, fig_num_hist, fig_width, fig_height, default_image=None):
        self.default_image = default_image
        self.threshold_display_canvas = GraphDisplayWidget(parent, fig_num_thresh, default_image_path=self.default_image)
        self.histogram_display_canvas = GraphDisplayWidget(parent, fig_num_hist, default_image_path=self.default_image)

        self.control_widget = QGroupBox("Thresholding and Histogram", parent)
        self.control_widget_layout = QVBoxLayout(parent)
        self.control_widget.setLayout(self.control_widget_layout)

        self.enable_button_widget = QGroupBox("Module Enable Options", parent)
        self.enable_button_widget_layout = QHBoxLayout()
        self.enable_button_widget.setLayout(self.enable_button_widget_layout)
        self.enable_thresholding_checkbox = QCheckBox("Thresholding", parent)
        self.enable_histogram_checkbox = QCheckBox("Histogram", parent)
        self.enable_button_widget_layout.addWidget(self.enable_thresholding_checkbox)
        self.enable_button_widget_layout.addWidget(self.enable_histogram_checkbox)
        self.control_widget_layout.addWidget(self.enable_button_widget)

        self.threshold_navigation_widget = QGroupBox("Thresholding Navigation Bar", parent)
        self.threshold_navigation_widget_layout = QHBoxLayout()
        self.threshold_navigation_widget.setLayout(self.threshold_navigation_widget_layout)
        self.threshold_navigation_widget_layout.addWidget(self.threshold_display_canvas.get_figure_toolbar())
        self.control_widget_layout.addWidget(self.threshold_navigation_widget)

        self.histogram_navigation_widget = QGroupBox("Histogram Navigation Bar", parent)
        self.histogram_navigation_widget_layout = QHBoxLayout()
        self.histogram_navigation_widget.setLayout(self.histogram_navigation_widget_layout)
        self.histogram_navigation_widget_layout.addWidget(self.histogram_display_canvas.get_figure_toolbar())
        self.control_widget_layout.addWidget(self.histogram_navigation_widget)

        self.controls = {"enable_threshold": self.enable_thresholding_checkbox,
                         "enable_histogram": self.enable_histogram_checkbox,
                         "threshold toolbar": self.threshold_display_canvas.get_figure_toolbar(),
                         "histogram toolbar": self.histogram_display_canvas.get_figure_toolbar()}

        self.view_widget_scrollbar = QScrollArea()
        self.view_widget_scrollbar.setMinimumWidth(fig_width)
        self.view_widget_scrollbar.setMinimumHeight(fig_height)
        self.view_widget_scrollbar.setWidgetResizable(True)
        self.view_widget_threshold = QGroupBox("Thresholding and Histogram")
        self.view_widget_layout = QHBoxLayout()
        self.view_widget_threshold.setLayout(self.view_widget_layout)



        self.view_widget_scrollbar.setWidget(self.view_widget_threshold)
        self.view_widget_layout.addWidget(self.threshold_display_canvas.get_figure_canvas())
        self.view_widget_layout.addWidget(self.histogram_display_canvas.get_figure_canvas())

        self.view_controls = {"threshold": self.threshold_display_canvas,
                              "histogram": self.histogram_display_canvas}
    def get_control_container(self):
        return self.control_widget
    def get_controls(self):
        return self.controls
    def get_view_container(self):
        return self.view_widget_scrollbar

    def get_view_controls(self):
        return self.view_controls
class ImageSegmentationModule:
    """
    Layout for Image Segmentation output of VideoProcessing thread
    Features:
    1. Superpixels
    2. Binary Segmentation-Foreground Extraction(Otsu's Algorithm, Graph Cut, Level Set/Chan Vese) & Connected Component Analysis(Gaussian Filters-Scikit)
    3. Marker Based Segmentation-Image Segmentation(Watershed, Random Walker Algorithm)
    4. Region Description-RegionProps

    """
    def __init__(self, parent, fig_num_inp, fig_num_exp, fig_num_out, fig_width, fig_height, default_image=None):
        pass

class NoiseDetectionModule:
    def __init__(self, parent, fig_num_inp, fig_num_exp, fig_num_out, fig_width, fig_height, default_image=None):
        self.default_image_path = default_image
        self.input_display_canvas = GraphDisplayWidget(parent, fig_num_inp, default_image_path=self.default_image_path)
        self.expected_out_display_canvas = GraphDisplayWidget(parent, fig_num_exp, default_image_path=self.default_image_path)
        self.predicted_out_display_canvas = GraphDisplayWidget(parent, fig_num_out, default_image_path=self.default_image_path)

        self.control_widget = QGroupBox("Noise Detection(Neural Network)", parent)
        self.control_widget_layout = QVBoxLayout(parent)
        self.control_widget.setLayout(self.control_widget_layout)

        self.nn_param_widget = QGroupBox("Image Data Parameters", parent)
        self.nn_param_widget_layout = QHBoxLayout()
        self.nn_param_widget.setLayout(self.nn_param_widget_layout)

        inp_w_box = QGroupBox("", parent)
        inp_w_lay = QVBoxLayout()
        inp_w_box.setLayout(inp_w_lay)
        self.nn_inp_w_label = QLabel("Input Width", parent)
        self.nn_inp_w_inp = QLineEdit(parent)
        inp_w_lay.addWidget(self.nn_inp_w_label)
        inp_w_lay.addWidget(self.nn_inp_w_inp)

        inp_h_box = QGroupBox("", parent)
        inp_h_lay = QVBoxLayout()
        inp_h_box.setLayout(inp_h_lay)
        self.nn_inp_h_label = QLabel("Input Height", parent)
        self.nn_inp_h_inp = QLineEdit(parent)
        inp_h_lay.addWidget(self.nn_inp_h_label)
        inp_h_lay.addWidget(self.nn_inp_h_inp)

        out_w_box = QGroupBox("", parent)
        out_w_lay = QVBoxLayout()
        out_w_box.setLayout(out_w_lay)
        self.nn_out_w_label = QLabel("Output Width", parent)
        self.nn_out_w_inp = QLineEdit(parent)
        out_w_lay.addWidget(self.nn_out_w_label)
        out_w_lay.addWidget(self.nn_out_w_inp)

        out_h_box = QGroupBox("", parent)
        out_h_lay = QVBoxLayout()
        out_h_box.setLayout(out_h_lay)
        self.nn_out_h_label = QLabel("Output Height", parent)
        self.nn_out_h_inp = QLineEdit(parent)
        out_h_lay.addWidget(self.nn_out_h_label)
        out_h_lay.addWidget(self.nn_out_h_inp)


        self.nn_param_widget_layout.addWidget(inp_w_box)
        self.nn_param_widget_layout.addWidget(inp_h_box)
        self.nn_param_widget_layout.addWidget(out_w_box)
        self.nn_param_widget_layout.addWidget(out_h_box)
        self.control_widget_layout.addWidget(self.nn_param_widget)

        self.nn_model_widget = QGroupBox("Model Parameters", parent)
        self.nn_model_widget_layout = QHBoxLayout()
        self.nn_model_widget.setLayout(self.nn_model_widget_layout)

        model_name_box = QGroupBox("", parent)
        model_name_lay = QVBoxLayout()
        model_name_box.setLayout(model_name_lay)
        self.nn_enable_module_checkbox = QCheckBox("Enable Module", parent)
        self.nn_model_name_label = QLabel("Model Name", parent)
        self.nn_model_name_inp = QLineEdit()
        model_name_lay.addWidget(self.nn_enable_module_checkbox)
        model_name_lay.addWidget(self.nn_model_name_label)
        model_name_lay.addWidget(self.nn_model_name_inp)

        dense_unit_box = QGroupBox("", parent)
        dense_unit_lay = QVBoxLayout()
        dense_unit_box.setLayout(dense_unit_lay)
        self.nn_dense_unit_label = QLabel("Neurons", parent)
        self.nn_dense_unit_inp = QLineEdit(parent)
        self.nn_dense_dropout_label = QLabel("Dropout rate", parent)
        self.nn_dense_dropout_inp = QLineEdit(parent)
        dense_unit_lay.addWidget(self.nn_dense_unit_label)
        dense_unit_lay.addWidget(self.nn_dense_unit_inp)
        dense_unit_lay.addWidget(self.nn_dense_dropout_label)
        dense_unit_lay.addWidget(self.nn_dense_dropout_inp)

        ksize_box = QGroupBox("", parent)
        ksize_lay = QVBoxLayout()
        ksize_box.setLayout(ksize_lay)
        self.nn_conv_ksize_label = QLabel("Convolution Kernel", parent)
        self.nn_conv_ksize_inp = QLineEdit(parent)
        self.nn_pool_ksize_label = QLabel("Pool Kernel", parent)
        self.nn_pool_ksize_inp = QLineEdit(parent)
        ksize_lay.addWidget(self.nn_conv_ksize_label)
        ksize_lay.addWidget(self.nn_conv_ksize_inp)
        ksize_lay.addWidget(self.nn_pool_ksize_label)
        ksize_lay.addWidget(self.nn_pool_ksize_inp)

        self.nn_model_widget_layout.addWidget(model_name_box)
        self.nn_model_widget_layout.addWidget(dense_unit_box)
        self.nn_model_widget_layout.addWidget(ksize_box)
        self.control_widget_layout.addWidget(self.nn_model_widget)

        self.nn_inp_navigation_widget = QGroupBox("Navigation Toolbar(Input)")
        self.nn_inp_navigation_widget_layout = QHBoxLayout()
        self.nn_inp_navigation_widget.setLayout(self.nn_inp_navigation_widget_layout)
        self.nn_inp_navigation_widget_layout.addWidget(self.input_display_canvas.get_figure_toolbar())
        self.control_widget_layout.addWidget(self.nn_inp_navigation_widget)

        self.nn_exp_navigation_widget = QGroupBox("Navigation Toolbar(Expected)")
        self.nn_exp_navigation_widget_layout = QHBoxLayout()
        self.nn_exp_navigation_widget.setLayout(self.nn_exp_navigation_widget_layout)
        self.nn_exp_navigation_widget_layout.addWidget(self.expected_out_display_canvas.get_figure_toolbar())
        self.control_widget_layout.addWidget(self.nn_exp_navigation_widget)

        self.nn_pre_navigation_widget = QGroupBox("Navigation Toolbar(Predicted)")
        self.nn_pre_navigation_widget_layout = QHBoxLayout()
        self.nn_pre_navigation_widget.setLayout(self.nn_pre_navigation_widget_layout)
        self.nn_pre_navigation_widget_layout.addWidget(self.predicted_out_display_canvas.get_figure_toolbar())
        self.control_widget_layout.addWidget(self.nn_pre_navigation_widget)

        self.controls = {"enable_threshold": self.nn_enable_module_checkbox,
                         "model name": self.nn_model_name_inp,
                         "inp w": self.nn_inp_w_inp,
                         "inp h": self.nn_inp_h_inp,
                         "out w": self.nn_out_w_inp,
                         "out h": self.nn_out_h_inp,
                         "neurons": self.nn_dense_unit_inp,
                         "conv ksize": self.nn_conv_ksize_inp,
                         "pool ksize": self.nn_pool_ksize_inp,
                         "dropout": self.nn_dense_dropout_inp,
                         "input nav": self.input_display_canvas.get_figure_toolbar(),
                         "expected nav": self.expected_out_display_canvas.get_figure_toolbar(),
                         "predicted nav": self.predicted_out_display_canvas.get_figure_toolbar()
                         }

        self.view_widget_scrollbar = QScrollArea()
        self.view_widget_scrollbar.setMinimumWidth(fig_width)
        self.view_widget_scrollbar.setMinimumHeight(fig_height)
        self.view_widget_threshold = QGroupBox("Neural Network(Noise Estimation)", self.view_widget_scrollbar)
        self.view_widget_layout = QHBoxLayout()

        self.view_widget_threshold.setLayout(self.view_widget_layout)



        self.view_widget_scrollbar.setWidget(self.view_widget_threshold)
        self.view_widget_scrollbar.setWidgetResizable(True)
        self.view_widget_layout.addWidget(self.input_display_canvas.get_figure_canvas())
        self.view_widget_layout.addWidget(self.expected_out_display_canvas.get_figure_canvas())
        self.view_widget_layout.addWidget(self.predicted_out_display_canvas.get_figure_canvas())

        self.view_controls = {"input": self.input_display_canvas,
                              "out": self.predicted_out_display_canvas,
                              "exp": self.expected_out_display_canvas}
    def get_control_container(self):
        return self.control_widget
    def get_controls(self):
        return self.controls
    def get_view_container(self):
        return self.view_widget_scrollbar

    def get_view_controls(self):
        return self.view_controls
class VideoProcessingLayout(CommonLayout):
    def __init__(self, parent, default_frame_width=200, default_frame_height=200):
        super(VideoProcessingLayout, self).__init__(parent)
        self.my_parent = parent
        self.default_frame_width = default_frame_width
        self.default_frame_height = default_frame_height
        self.default_frame_image_path = "D:\\default.png"
        self.plot_width = 400
        self.plot_height = 200
        # Load Modules
        self.contour_object_detection_module = ContourObjectDetectionModule(parent, 1, self.plot_width, self.plot_height, self.default_frame_image_path)
        self.threshold_histogram_module = ThresholdingModule(parent, 2, 3, self.plot_width, self.plot_height, self.default_frame_image_path)
        self.noise_detection_module = NoiseDetectionModule(parent, 4, 5, 6, self.plot_width, self.plot_height, self.default_frame_image_path)


        # Set Layout
        self.start_stop_btn_widget = QGroupBox("Processor Control", parent)
        self.start_stop_btn_widget_layout = QHBoxLayout()
        self.start_stop_btn_widget.setLayout(self.start_stop_btn_widget_layout)
        self.start_btn = QPushButton("Start", parent)
        self.stop_btn = QPushButton("Stop", parent)
        self.start_stop_btn_widget_layout.addWidget(self.start_btn)
        self.start_stop_btn_widget_layout.addWidget(self.stop_btn)

        self.control_widget_layout.addWidget(self.start_stop_btn_widget)
        self.control_widget_layout.addWidget(self.contour_object_detection_module.get_control_container())
        self.control_widget_layout.addWidget(self.threshold_histogram_module.get_control_container())
        self.control_widget_layout.addWidget(self.noise_detection_module.get_control_container())
        self.view_widget_layout.addWidget(self.contour_object_detection_module.get_view_container())
        self.view_widget_layout.addWidget(self.threshold_histogram_module.get_view_container())
        self.view_widget_layout.addWidget(self.noise_detection_module.get_view_container())

        # Get Parameters
        self.processor_status_control = {"start": self.start_btn, "stop": self.stop_btn}
        self.contour_object_params = self.contour_object_detection_module.get_controls()
        self.threshold_detection_params = self.threshold_histogram_module.get_controls()
        self.noise_detection_params = self.noise_detection_module.get_controls()
        self.contour_view_params = self.contour_object_detection_module.get_view_controls()
        self.threshold_view_params = self.threshold_histogram_module.get_view_controls()
        self.noise_view_params = self.noise_detection_module.get_view_controls()
class NeuralNetworkLayout(CommonLayout):
    def __init__(self, parent, default_frame_width=200, default_frame_height=200):
        super(NeuralNetworkLayout, self).__init__(parent)
        self.my_parent = parent
        self.default_frame_width = default_frame_width
        self.default_frame_height = default_frame_height
        self.default_frame_image_path = "D:\\default.png"
        self.plot_width = 400
        self.plot_height = 400

class ImageAnnotationLayout(CommonLayout):
    def __init__(self, parent, default_frame_width=200, default_frame_height=200):
        super(ImageAnnotationLayout, self).__init__(parent)
        self.my_parent = parent
        self.default_frame_width = default_frame_width
        self.default_frame_height = default_frame_height
        self.default_frame_image_path = "D:\\default.png"
        #self.plot_width = 400
        #self.plot_height = 200
        self.hog_cells_per_block = 1
        self.hog_pixels_per_cell = 16
        self.hog_orientation = 8
        self.kmeans_clusters = 20
        self.orb_features = 100
        self.img_mode = ['rgb', 'gray'] # rgb or gray
        self.annotation_img_view_options = ['color', 'edges', 'orientation', 'distance', 'clusters', 'orb']
        self.image_props = ['Color', 'Orientation', 'Region Properties & Clusters', 'Edges', 'ORB Features', 'Distance']
        self.base_img_folder = 'annotated_images'
        self.base_img_save_dir = "D:\\thesis\\ConvNet\\MyNet\\image_segmentation\\"+self.base_img_folder+"\\"
        self.view_images_per_row = 2
        self.view_widgets = OrderedDict()

        for img_type in self.annotation_img_view_options:
            self.view_widgets[img_type] = {
                'widget': ImageDisplayWidget(self.parent, self.default_frame_image_path, self.default_frame_width, self.default_frame_height),
            }

        self.image_opt_widgets = OrderedDict()
        self.image_opt_widgets['load image path'] = {
                'widget': QLineEdit(self.parent),
                'default': "D:\\path\\to\\file.extension"
            }
        self.image_opt_widgets['load image button'] = {
                'widget': QPushButton("Load Image", self.parent),
                'default': None
            }
        self.image_opt_widgets['hog cells per block'] = {
                'widget': QLineEdit(self.parent),
                'default': self.hog_cells_per_block
            }
        self.image_opt_widgets['hog pixels per cell'] = {
                'widget': QLineEdit(self.parent),
                'default': self.hog_pixels_per_cell
            }
        self.image_opt_widgets['hog orientations'] = {
                'widget': QLineEdit(self.parent),
                'default': self.hog_orientation
            }
        self.image_opt_widgets['total orb features'] = {
                'widget': QLineEdit(self.parent),
                'default': self.orb_features
            }
        self.image_opt_widgets['hold frame'] =  {
                'widget': QCheckBox("", self.parent),
                'default': False
            }
        self.image_opt_widgets['image width'] = {
            'widget': QLineEdit(self.parent),
            'default': self.default_frame_width
        }
        self.image_opt_widgets['image height'] = {
            'widget': QLineEdit(self.parent),
            'default': self.default_frame_height
        }
        self.image_opt_widgets['image mode'] = {
            'widget': QComboBox(self.parent),
            'default': self.img_mode
        }

        self.annotate_opt_widgets = OrderedDict()
        self.annotate_opt_widgets['annotation view mode'] = {
            'widget': QComboBox(self.parent),
            'default': self.annotation_img_view_options
        }
        self.annotate_opt_widgets['start annotation'] = {
            'widget': QPushButton("Start Annotation", self.parent),
            'default': None
        }
        self.annotate_opt_widgets['region properties'] = {
            'widget': QPushButton("View Region Properties", self.parent),
            'default': None
        }

        self.save_opt_widgets = OrderedDict()
        self.save_opt_widgets['base directory'] = {
            'widget': QLineEdit(self.parent),
            'default': self.base_img_save_dir
        }
        self.containers = OrderedDict()

        self.image_opt_container = QGroupBox("Image Parameters", self.parent)
        self.image_opt_layout = QVBoxLayout(self.parent)
        self.image_opt_container.setLayout(self.image_opt_layout)
        for widget in self.image_opt_widgets:
            if type(self.image_opt_widgets[widget]['widget']) == QLineEdit:
                self.image_opt_widgets[widget]['widget'].setText(str(self.image_opt_widgets[widget]['default']))
            elif type(self.image_opt_widgets[widget]['widget']) == QComboBox:
                for i in range(len(self.image_opt_widgets[widget]['default'])):
                    self.image_opt_widgets[widget]['widget'].addItem(str(self.image_opt_widgets[widget]['default'][i]))

            form_box = QGroupBox("", self.parent)
            form_layout = QFormLayout(self.parent)
            form_box.setLayout(form_layout)
            form_layout.addRow(QLabel(str(widget).upper()), self.image_opt_widgets[widget]['widget'])
            self.image_opt_layout.addWidget(form_box)
        self.control_widget_layout.addWidget(self.image_opt_container)

        self.annotate_opt_container = QGroupBox("Annotation Parameters", self.parent)
        self.annotate_opt_layout = QVBoxLayout(self.parent)
        self.annotate_opt_container.setLayout(self.annotate_opt_layout)
        for widget in self.annotate_opt_widgets:
            if type(self.annotate_opt_widgets[widget]['widget']) == QLineEdit:
                self.annotate_opt_widgets[widget]['widget'].setText(str(self.annotate_opt_widgets[widget]['default']))
            elif type(self.annotate_opt_widgets[widget]['widget']) == QComboBox:
                for i in range(len(self.annotate_opt_widgets[widget]['default'])):
                    self.annotate_opt_widgets[widget]['widget'].addItem(str(self.annotate_opt_widgets[widget]['default'][i]))

            form_box = QGroupBox("", self.parent)
            form_layout = QFormLayout(self.parent)
            form_box.setLayout(form_layout)
            form_layout.addRow(QLabel(str(widget).upper()), self.annotate_opt_widgets[widget]['widget'])
            self.annotate_opt_layout.addWidget(form_box)
        self.control_widget_layout.addWidget(self.annotate_opt_container)

        self.save_opt_container = QGroupBox("Save Parameters", self.parent)
        self.save_opt_layout = QVBoxLayout(self.parent)
        self.save_opt_container.setLayout(self.save_opt_layout)
        for widget in self.save_opt_widgets:
            if type(self.save_opt_widgets[widget]['widget']) == QLineEdit:
                self.save_opt_widgets[widget]['widget'].setText(str(self.save_opt_widgets[widget]['default']))
            elif type(self.save_opt_widgets[widget]['widget']) == QComboBox:
                for i in range(len(self.save_opt_widgets[widget]['default'])):
                    self.save_opt_widgets[widget]['widget'].addItem(str(self.save_opt_widgets[widget]['default'][i]))

            form_box = QGroupBox("", self.parent)
            form_layout = QFormLayout(self.parent)
            form_box.setLayout(form_layout)
            form_layout.addRow(QLabel(str(widget).upper()), self.save_opt_widgets[widget]['widget'])
            self.save_opt_layout.addWidget(form_box)
        self.control_widget_layout.addWidget(self.save_opt_container)

        layout = None
        i=0
        for img in self.view_widgets:
            if i%self.view_images_per_row == 0:
                container = QGroupBox("", self.parent)
                layout = QHBoxLayout()
                container.setLayout(layout)
                self.view_widget_layout.addWidget(container)

            layout.addWidget(self.view_widgets[img]['widget'])
            self.view_widgets[img]['widget'].update_image(self.view_widgets[img]['widget'].default_img)
            i+=1




