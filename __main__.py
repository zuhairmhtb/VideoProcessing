import sys,pdb
from win32api import GetSystemMetrics
from collections import OrderedDict
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel, QPushButton, QScrollBar, QMainWindow
from PyQt5.QtGui import QIcon
from interfaces import MainInterface
from widgets import MyTabWidget
from layouts import WebcamPlayerLayout, VideoProcessingLayout, NeuralNetworkLayout, ImageAnnotationLayout
from controllers import WebcamPlayerController, VideoProcessingController, ImageAnnotationController


import  numpy as np
@MainInterface.register
class App(QMainWindow):
    """
    This is the main thread of the GUI that controls all the interface threads and data flow between them.
    Available Interfaces(Tabs):
    1. Main GUI Interface(this)
    2. Webcam Player Interface / Video Input Interface
    3. Video Processing Interface
    4. Neural Network Interface
    Tasks:
    1. Initializes and creates the main layout(Tabs for each available interface, scrollable layout for Control Widgets
       and scrollable layout for View Widgets).
    2. Initializes all interface objects and acquires view widget containing all placed views and control widget
       containing all placed control buttons of the default interface(Webcam Player).
    3. Handles Input for user trying to exit and closes all threads correctly before shutting down the application
    Storage:
    1. Stores current frame provided by a running WebcamPlayer thread.
    2. Stores a list of last frame provided by a running WebcamPlayer thread.
    3. Stores each NeuralNetwork Object created by VideoProcessor thread. It provides a reference of the object to
       VideoProcessor thread upon each create_neural_network request.
    Interface Communication Protocols:
    2. Implements all interfaces in order to transfer data between them.
       The protocols for data flow between interface objects are as follows:
           a. Webcam Player interface can store current frame object in the main thread. The current frame object
              contains a 3D numpy array of raw RGB frame captured from webcam and adjusted using (H,S,V,Contrast) params,
              a 2D numpy array of the corresponding grayscale image, a 2D numpy array of Edge Image detected using
              available edge detection algorithms(Canny, Sobel, Laplacian), a list of 2D contours detected from the
              edge map and a list of hierarchy information obtained from the contour detection algorithm.
           b. Webcam Player Interface can update queue of the main thread which stores a list of previously received
              frames from the WebcamPlayer Object.
           c. VideoProcessor Interface can receive current Image from the main thread which needs to be processed.
              VideoProcessor Interface can request an image for processing which will be provided from the current
              frame stored by WebcamPlayer.
           d. VideoProcessor Interface can read data from from the queue where a set of last processed frames/images
              are stored by the WebcamPlayer in order to perform NeuralNetwork Operations.
           e. VideoProcessor Interface can request a NeuralNetwork Interface object in order to perform pattern
               recognition.
           g. The NeuralNetwork Interface can be requested to train using a set of valid input data and label and
              returns performance metric of the training. Views(Performance View and Model View) of the NeuralNetwork
              interface will automatically be updated after each training request.
           h. The NeuralNetwork Interface can be requested for predicting output using a valid set of input which returns
              the predicted output image and corresponding performance metrics. Views(Data View) of the NeuralNetwork
              interface will be automatically updated at each prediction request.

    """
    def __init__(self):
        super().__init__()

        # Window Parameters
        self.title = 'Video Processing Software(VPS)'
        self.layout_position_offset = 100 # The offset from which the main window will be drawn in the screen starting from (0,0)
        self.window_size_reduction_offset = 200  # The number of pixels by which the main window width and height will be reduced
        self.left = 0 + self.layout_position_offset  # Starting Left pixel coordinate of the main window
        self.top = 0 + self.layout_position_offset  # Starting Top pixel coordinate of the main window
        self.width = GetSystemMetrics(0) - self.window_size_reduction_offset  # Width of the main window
        self.height = GetSystemMetrics(1) - self.window_size_reduction_offset  # height of the main window


        # Core parameters

        self.neural_network_save_dir = "D:\\thesis\\ConvNet\\MyNet\\image_segmentation\\video_processing_software\\"
        self.neural_network_model_filename = "model"  # The base name of a neural network model file
        self.neural_network_model_fileext = "json"  # The format of the neural network model file
        self.neural_network_weight_filename = "weights"  # The base name of the neural network weights file
        self.neural_network_weight_fileext = "h5"  # The format of the neural network weight file


        self.default_frame_width = 600  # The default width for the webcam feed received by the webcam player
        self.default_frame_height = 600 # Default height for the webcam feed received by the webcam player
        self.default_nn_input_image_width = 200  # Default width for the input image provided to neural network
        self.default_nn_input_image_height = 200  # Default height for the input image provided to neural network
        self.default_nn_output_image_width = 200  # Default width for the output image provided by the network
        self.default_nn_output_image_height = 200  # Default height of the output image provided by the network
        self.default_nn_dense_neurons = 100  # The number of neurons in each dense layer of the neural network
        self.default_nn_conv_kernel_size = 5  # The default kernel size for convolution layer of the neural network
        self.default_nn_conv_kernel_stride = 1  # the number of strides for convolution layer of the neural network
        self.default_nn_pool_kernel_size = 2  # The default kernel size for pooling layer of the neural network
        self.default_nn_pool_kernel_stride = 2  # The number os strides for the pooling layer
        self.default_nn_dropout_rate = 0.4  # The default rate of dropout of dense layer of the network
        self.default_nn_activation_function = 'relu'  # the default activation function used
        self.default_nn_loss = 'binary_crossentropy'  # Default loss calculation method
        self.default_nn_optimizer = 'adam'  # Default optimizer for the neural network
        self.default_nn_train_epochs_per_frame = 1  # The number of epochs for which to train a single image
        self.default_input_image_dimension = 2  # Dimension for input image(Grayscale)
        self.default_output_image_dimension = 2  # Dimension for output image(Grayscale)


        self.default_contour_display_amount = 8  # Default number of identified objects using contours to display
        self.default_contours_column = 2  # Default Number of columns for each contour plot




        # Interface data variables
        self.current_frame_object = None  # The current frame object
        self.last_frame_amount_max = 20 # The maximum number of obtained last frame objects to be stored
        self.last_frame_list = []  # The last frame objects stored
        self.neural_networks = []  # The list of neural network objects created by the video processor



        # Core Interface layout
        default_original_frame_width = 352
        default_original_frame_height = 288
        default_edge_frame_width = 352
        default_edge_frame_height = 288
        QApplication.instance().aboutToQuit.connect(self.cleanUp)
        self.available_interfaces = OrderedDict()
        self.available_interfaces["Webcam Player"] = WebcamPlayerLayout(self, default_original_frame_width, default_original_frame_height)
        self.available_interfaces["Video Processing"] = VideoProcessingLayout(self, default_original_frame_width, default_original_frame_height)
        self.available_interfaces["Image Annotation"] = ImageAnnotationLayout(self, default_original_frame_width,
                                                                              default_original_frame_height)
        self.available_interfaces["Neural Network"] = NeuralNetworkLayout(self, default_original_frame_width, default_original_frame_height)


        # Image Annotation Setup
        self.image_annotation_controller = ImageAnnotationController(self, self.available_interfaces["Image Annotation"])
        image_annotate_int = self.available_interfaces["Image Annotation"]
        image_annotate_int.image_opt_widgets['load image button']['widget'].clicked.connect(self.image_annotation_controller.load_image)
        image_annotate_int.image_opt_widgets['hog cells per block']['widget'].textChanged.connect(
            self.image_annotation_controller.set_hog_cpb)
        image_annotate_int.image_opt_widgets['hog pixels per cell']['widget'].textChanged.connect(
            self.image_annotation_controller.set_hog_ppc)
        image_annotate_int.image_opt_widgets['hog orientations']['widget'].textChanged.connect(
            self.image_annotation_controller.set_hog_orientation)
        image_annotate_int.image_opt_widgets['total orb features']['widget'].textChanged.connect(
            self.image_annotation_controller.set_orb_features)
        image_annotate_int.image_opt_widgets['hold frame']['widget'].clicked.connect(
            self.image_annotation_controller.set_hold_frame)
        image_annotate_int.image_opt_widgets['image width']['widget'].textChanged.connect(
            self.image_annotation_controller.set_img_width)
        image_annotate_int.image_opt_widgets['image height']['widget'].textChanged.connect(
            self.image_annotation_controller.set_img_height)
        image_annotate_int.image_opt_widgets['image mode']['widget'].currentTextChanged.connect(
            self.image_annotation_controller.set_img_mode)
        image_annotate_int.annotate_opt_widgets['annotation view mode']['widget'].currentTextChanged.connect(
            self.image_annotation_controller.set_annotation_view_mode)
        image_annotate_int.annotate_opt_widgets['start annotation']['widget'].clicked.connect(
            self.image_annotation_controller.start_annotation_btn)
        image_annotate_int.annotate_opt_widgets['region properties']['widget'].clicked.connect(
            self.image_annotation_controller.regionprop_view_btn)


        # Video Processor Setup
        self.video_proc_is_plotting = False
        self.video_processing_controller = VideoProcessingController(self, self.available_interfaces["Video Processing"])
        video_proc_int = self.available_interfaces["Video Processing"]
        video_proc_int.processor_status_control["start"].clicked.connect(self.start_video_processor)
        video_proc_int.processor_status_control["stop"].clicked.connect(self.stop_video_processor)
        video_proc_int.contour_object_params["enable"].clicked.connect(self.enable_contour_object_detection)
        video_proc_int.contour_object_params["enable"].setChecked(self.video_processing_controller.plot_contours)
        video_proc_int.contour_object_params["contour amount"].textChanged.connect(self.change_contours_to_display)
        video_proc_int.contour_object_params["contour amount"].setText(str(self.video_processing_controller.display_contours_amount))
        video_proc_int.threshold_detection_params["enable_threshold"].clicked.connect(self.enable_thresholding)
        video_proc_int.threshold_detection_params["enable_threshold"].setChecked(self.video_processing_controller.plot_thresholding)
        video_proc_int.threshold_detection_params["enable_histogram"].clicked.connect(self.enable_histogram)
        video_proc_int.threshold_detection_params["enable_histogram"].setChecked(self.video_processing_controller.plot_histogram)
        video_proc_int.noise_detection_params["enable_threshold"].clicked.connect(self.enable_noise_detection)
        video_proc_int.noise_detection_params["inp w"].textChanged.connect(self.change_noise_detection_io_params)
        video_proc_int.noise_detection_params["inp w"].setText(str(self.video_processing_controller.nn_inp_w))
        video_proc_int.noise_detection_params["inp h"].textChanged.connect(self.change_noise_detection_io_params)
        video_proc_int.noise_detection_params["inp h"].setText(str(self.video_processing_controller.nn_inp_h))
        video_proc_int.noise_detection_params["out w"].textChanged.connect(self.change_noise_detection_io_params)
        video_proc_int.noise_detection_params["out w"].setText(str(self.video_processing_controller.nn_out_w))
        video_proc_int.noise_detection_params["out h"].textChanged.connect(self.change_noise_detection_io_params)
        video_proc_int.noise_detection_params["out h"].setText(str(self.video_processing_controller.nn_out_h))
        video_proc_int.noise_detection_params["neurons"].textChanged.connect(self.change_noise_detection_model_params)
        video_proc_int.noise_detection_params["neurons"].setText(str(self.video_processing_controller.nn_neurons))
        video_proc_int.noise_detection_params["conv ksize"].textChanged.connect(self.change_noise_detection_model_params)
        video_proc_int.noise_detection_params["conv ksize"].setText(str(self.video_processing_controller.nn_conv_kszie))
        video_proc_int.noise_detection_params["pool ksize"].textChanged.connect(self.change_noise_detection_model_params)
        video_proc_int.noise_detection_params["pool ksize"].setText(str(self.video_processing_controller.nn_pool_ksize))
        video_proc_int.noise_detection_params["dropout"].textChanged.connect(self.change_noise_detection_model_params)
        video_proc_int.noise_detection_params["dropout"].setText(str(self.video_processing_controller.nn_dropout))

        # WebcamPlayer Interface Setup
        self.cont_bri_shar_thresh = 20
        self.webcam_player_controller = WebcamPlayerController(default_original_frame_width,
                                                               default_original_frame_height,
                                                               default_edge_frame_width, default_edge_frame_height)

        webcam_player_int = self.available_interfaces["Webcam Player"]
        webcam_player_int.stream_buttons["play"].clicked.connect(self.start_stream)
        webcam_player_int.stream_buttons["pause"].clicked.connect(self.stop_stream)
        webcam_player_int.stream_buttons["stop"].clicked.connect(self.stop_stream)

        self.webcam_player_controller.store_frame_signal.connect(self.store_current_frame)

        webcam_player_int.brightness_control_sliders["brightness"].setValue(
            int(self.webcam_player_controller.brightness_level*self.cont_bri_shar_thresh)
        )
        webcam_player_int.brightness_control_sliders["brightness"].valueChanged.connect(self.change_brightness_params)

        webcam_player_int.brightness_control_sliders["contrast"].setValue(
            int(self.webcam_player_controller.contrast_level * self.cont_bri_shar_thresh)
        )
        webcam_player_int.brightness_control_sliders["contrast"].valueChanged.connect(self.change_contrast_params)

        webcam_player_int.brightness_control_sliders["sharpness"].setValue(
            int(self.webcam_player_controller.sharpness_level * self.cont_bri_shar_thresh)
        )
        webcam_player_int.brightness_control_sliders["sharpness"].valueChanged.connect(self.change_sharpness_params)

        webcam_player_int.auto_adjustment_control_input["auto adjustment"][0].setChecked(self.webcam_player_controller.perform_automatic_param_adjustment)
        webcam_player_int.auto_adjustment_control_input["auto adjustment"][0].clicked.connect(self.toggle_auto_adjust_param)

        webcam_player_int.auto_adjustment_control_input["auto adjustment"][1].setText(str(self.webcam_player_controller.parameter_change_duration))
        webcam_player_int.auto_adjustment_control_input["auto adjustment"][1].textChanged.connect(self.set_automatic_update_dur)

        webcam_player_int.auto_adjustment_control_input["edge size"][0].setText(str(self.webcam_player_controller.edge_size_min))
        webcam_player_int.auto_adjustment_control_input["edge size"][0].textChanged.connect(self.set_edge_size_adjustment_params)
        webcam_player_int.auto_adjustment_control_input["edge size"][1].setText(
            str(self.webcam_player_controller.edge_size_max))
        webcam_player_int.auto_adjustment_control_input["edge size"][1].textChanged.connect(
            self.set_edge_size_adjustment_params)
        webcam_player_int.auto_adjustment_control_input["edge size"][2].setText(
            str(self.webcam_player_controller.edge_size_increase))
        webcam_player_int.auto_adjustment_control_input["edge size"][2].textChanged.connect(
            self.set_edge_size_adjustment_params)


        webcam_player_int.frame_process_control_inputs["original"][0].setText(str(
            self.webcam_player_controller.original_frame_width
        ))
        webcam_player_int.frame_process_control_inputs["original"][0].textChanged.connect(self.set_frame_size_adjustment)
        webcam_player_int.frame_process_control_inputs["original"][1].setText(str(
            self.webcam_player_controller.original_frame_height
        ))
        webcam_player_int.frame_process_control_inputs["original"][1].textChanged.connect(
            self.set_frame_size_adjustment)
        webcam_player_int.frame_process_control_inputs["edge"][0].setText(str(
            self.webcam_player_controller.edge_frame_width
        ))
        webcam_player_int.frame_process_control_inputs["edge"][0].textChanged.connect(
            self.set_frame_size_adjustment)
        webcam_player_int.frame_process_control_inputs["edge"][1].setText(str(
            self.webcam_player_controller.edge_frame_height
        ))
        webcam_player_int.frame_process_control_inputs["edge"][1].textChanged.connect(
            self.set_frame_size_adjustment)
        webcam_player_int.edge_detect_controls["edge detection"][1].setText(
            str(self.webcam_player_controller.get_thresholds(self.webcam_player_controller.selected_edge_algorithm)))
        webcam_player_int.edge_detect_controls["edge detection"][0].currentTextChanged.connect(self.set_edge_detection_alg)
        webcam_player_int.edge_detect_controls["contour detection"][0].clicked.connect(self.toggle_contour_enable)
        webcam_player_int.edge_detect_controls["contour detection"][1].setText(str(self.webcam_player_controller.contour_min_object_size))
        webcam_player_int.edge_detect_controls["erosion"][0].valueChanged.connect(self.set_erosion_dilation_param)
        webcam_player_int.edge_detect_controls["erosion"][1].valueChanged.connect(self.set_erosion_dilation_param)
        webcam_player_int.edge_detect_controls["erosion"][2].valueChanged.connect(self.set_erosion_dilation_param)
        webcam_player_int.edge_detect_controls["erosion"][3].valueChanged.connect(self.set_erosion_dilation_param)


        self.initUI()  # Initialize the main window


    def start_video_processor(self):
        if not self.video_processing_controller.should_run:
            self.video_processing_controller.start()
    def stop_video_processor(self):
        if self.video_processing_controller.should_run:
            self.video_processing_controller.should_run = False
    def enable_contour_object_detection(self):
        self.video_processing_controller.plot_contours = self.available_interfaces["Video Processing"].contour_object_params["enable"].isChecked()
    def change_contours_to_display(self):
        amount = self.available_interfaces["Video Processing"].contour_object_params["contour amount"].text()
        try:
            amount = int(amount)
            if amount > 0 and amount < 10:
                self.video_processing_controller.display_contours_amount = amount
        except:
            pass
    def enable_thresholding(self):
        self.video_processing_controller.plot_thresholding = self.available_interfaces["Video Processing"].threshold_detection_params["enable_threshold"].isChecked()
    def enable_histogram(self):
        self.video_processing_controller.plot_histogram = self.available_interfaces["Video Processing"].threshold_detection_params["enable_histogram"].isChecked()
    def enable_noise_detection(self):
        self.video_processing_controller.plot_neural_net_output = self.available_interfaces["Video Processing"].noise_detection_params["enable_threshold"].isChecked()
    def change_noise_detection_io_params(self):
        pass
    def change_noise_detection_model_params(self):
        pass

    def start_stream(self):
        if not self.webcam_player_controller.should_run:
            self.webcam_player_controller.start()

        if not self.image_annotation_controller.should_run:
            self.image_annotation_controller.start()
    def stop_stream(self):
        if self.image_annotation_controller.should_run:
            self.image_annotation_controller.stop_thread()
            self.image_annotation_controller.wait()
        if self.webcam_player_controller.should_run:
            self.webcam_player_controller.stop_thread()
            self.webcam_player_controller.wait()



    def change_brightness_params(self):
        brightness_controls = self.available_interfaces["Webcam Player"].brightness_control_sliders
        brightness = int(brightness_controls["brightness"].value())
        if brightness < self.cont_bri_shar_thresh:
            self.webcam_player_controller.brightness_level = brightness

        else:
            self.webcam_player_controller.brightness_level = float(brightness / 100)
    def change_contrast_params(self):
        brightness_controls = self.available_interfaces["Webcam Player"].brightness_control_sliders
        contrast = int(brightness_controls["contrast"].value())
        if contrast < self.cont_bri_shar_thresh:
            self.webcam_player_controller.contrast_level = contrast
        else:
            self.webcam_player_controller.contrast_level = float(contrast / 100)
    def change_sharpness_params(self):
        brightness_controls = self.available_interfaces["Webcam Player"].brightness_control_sliders
        sharpness = int(brightness_controls["sharpness"].value())
        if sharpness  < self.cont_bri_shar_thresh:
            self.webcam_player_controller.sharpness_level = sharpness
        else:
            self.webcam_player_controller.sharpness_level = float(sharpness / 100)

    def toggle_auto_adjust_param(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        self.webcam_player_controller.perform_automatic_param_adjustment = webcam_player_int.auto_adjustment_control_input["auto adjustment"][0].isChecked()

    def set_automatic_update_dur(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        try:
            val = int(webcam_player_int.auto_adjustment_control_input["auto adjustment"][1].text())
            if val > 0:
                self.webcam_player_controller.parameter_change_duration = val
        except:
            pass

    def set_edge_size_adjustment_params(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        try:
            min_size = float( webcam_player_int.auto_adjustment_control_input["edge size"][0].text())
            max_size = float(webcam_player_int.auto_adjustment_control_input["edge size"][1].text())
            step_size = float(webcam_player_int.auto_adjustment_control_input["edge size"][2].text())
            if min_size >= 0 and min_size <= 1:
                self.webcam_player_controller.edge_size_min = min_size
            if max_size > min_size and max_size <=1:
                self.webcam_player_controller.edge_size_max = max_size
            if step_size > 0 and step_size < 0.5:
                self.webcam_player_controller.edge_size_increase = step_size
        except:
            pass
    def set_frame_size_adjustment(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        try:
            or_f_w = int(webcam_player_int.frame_process_control_inputs["original"][0].text())
            or_f_h = int(webcam_player_int.frame_process_control_inputs["original"][1].text())
            e_f_w = int(webcam_player_int.frame_process_control_inputs["edge"][0].text())
            e_f_h = int(webcam_player_int.frame_process_control_inputs["edge"][1].text())
            #print("Adjusting size")
            if or_f_w > 0:
                self.webcam_player_controller.original_frame_width = or_f_w
            if or_f_h > 0:
                self.webcam_player_controller.original_frame_height = or_f_h
            if e_f_w > 0:
                self.webcam_player_controller.edge_frame_width = e_f_w
            if e_f_h > 0:
                self.webcam_player_controller.edge_frame_height = e_f_h


        except:
            pass

    def set_edge_detection_alg(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        self.webcam_player_controller.selected_edge_algorithm = str(webcam_player_int.edge_detect_controls["edge detection"][0].currentText())

    def set_erosion_dilation_param(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        try:
            print("Calculating")
            erosion_dilation = webcam_player_int.edge_detect_controls["erosion"]
            erosion_a = int(erosion_dilation[0].value())
            dilation_a = int(erosion_dilation[1].value())
            erosion_b = int(erosion_dilation[2].value())
            dilation_b = int(erosion_dilation[3].value())
            if erosion_a >= 0:
                if erosion_a != 0 and erosion_a%2 == 0:
                    erosion_a += 1
                self.webcam_player_controller.erosiona_ksize = erosion_a
            if erosion_b >= 0:
                if erosion_b != 0 and erosion_b%2 == 0:
                    erosion_b += 1
                self.webcam_player_controller.erosionb_ksize = erosion_b
            if dilation_a >= 0:
                if dilation_a != 0 and dilation_a%2 == 0:
                    dilation_a += 1
                self.webcam_player_controller.dilationa_ksize = dilation_a
            if dilation_b >= 0:
                if dilation_b != 0 and dilation_b%2 == 0:
                    dilation_b += 1
                self.webcam_player_controller.dilationb_ksize = dilation_b
        except:
            print("error")

    def toggle_contour_enable(self):
        webcam_player_int = self.available_interfaces["Webcam Player"]
        self.webcam_player_controller.contour_display_enabled = webcam_player_int.edge_detect_controls["contour detection"][0].isChecked()
    def store_current_frame(self, frame_object):
        # Store current frame
        self.current_frame_object = frame_object
        if len(self.last_frame_list) >= self.last_frame_amount_max:
            self.last_frame_list = self.last_frame_list[1:] + [frame_object.copy()]
        else:
            self.last_frame_list.append(frame_object.copy())

        # Update webcam player display
        # Update original frame
        self.available_interfaces["Webcam Player"].display_frame_controls["top"][0].update_image(frame_object[0].copy())
        # Update adjusted frame
        self.available_interfaces["Webcam Player"].display_frame_controls["top"][1].update_image(frame_object[1].copy())
        # Update grayscale frame
        self.available_interfaces["Webcam Player"].display_frame_controls["bottom"][0].update_image(frame_object[2].copy())
        # Update edge frame
        edge_frame = frame_object[3].copy()
        contours = frame_object[4].copy()
        hierarchy = frame_object[5].copy()

        # Add contour boundaries if boundary view is enabled
        if self.webcam_player_controller.contour_display_enabled:
            edge_frame = self.webcam_player_controller.draw_bounding_box(edge_frame, contours, hierarchy)
        self.available_interfaces["Webcam Player"].display_frame_controls["bottom"][1].update_image(
            edge_frame)



    def get_current_frame(self):
        if not (self.current_frame_object is None):
            return self.current_frame_object.copy()
        else:
            return None
    def get_last_frame_queue(self, nmax=1):
        if nmax > 0 and nmax < len(self.last_frame_list):
            return self.last_frame_list[-int(nmax):]
        else:
            return self.last_frame_list
    def create_neural_network_model_params(self, inp_img_w, inp_img_h, out_img_w, out_img_h, dense_units,
                                           conv_ksize, conv_kstride, pool_ksize, pool_kstride, dropout,
                                           activation, loss, optimizer):
        pass
    def request_neural_network(self, model_name, model_params):
        pass

    def cleanUp(self):
        """
        This function runs before the app is about to quit. It closes all active threads and stores any unsaved
        data.
        :return: null
        """
        print("Quitting")
        if self.webcam_player_controller.isRunning() or self.webcam_player_controller.should_run:
            self.webcam_player_controller.stop_thread()
            self.webcam_player_controller.wait()

        if self.video_processing_controller.isRunning() or self.video_processing_controller.should_run:
            self.video_processing_controller.stop_thread()
            self.video_processing_controller.wait()

        if self.image_annotation_controller.isRunning() or self.image_annotation_controller.should_run:
            self.image_annotation_controller.stop_thread()
            self.image_annotation_controller.wait()

        #self.video_processing_controller.neural_net.save_model()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tab_width = self.width - 50
        self.tab_height = self.height - 50
        self.tab_names = [k for k in self.available_interfaces]
        self.tab_layouts = [self.available_interfaces[k] for k in self.available_interfaces]

        # Create layouts for each interface
        self.my_tab = MyTabWidget(self, self.tab_width, self.tab_height, self.tab_names, self.tab_layouts)
        self.setCentralWidget(self.my_tab)

        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())