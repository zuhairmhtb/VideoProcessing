from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import cv2, os, sys, time
from cv2 import Canny, Sobel, Laplacian
from PIL import ImageEnhance, Image
from keras import Sequential
from keras.layers import Dense, Dropout, MaxPool2D, Flatten, Conv2D
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter

from skimage.measure import compare_ssim
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb, quickshift, watershed
from skimage.filters import sobel, threshold_otsu, median
from scipy.ndimage import distance_transform_edt
from skimage import measure, segmentation, feature, color, exposure
from skimage.util import img_as_float
from tensorflow import Graph, Session
from keras import backend as K

from layouts import ImageAnnotationLayout


# your logic here
class MyNeuralNetwork:
    def __init__(self, inp_w, inp_h, out_w, out_h, neurons, conv_ksize, pool_ksize, activation='relu', loss='mean_squared_error', optimizer='adam'):
        """
        This is constructor for the MyNeuralNetwork class that detects possible noise pixels from a grayscale/binary
        edge image
        :param inp_w:  The width of the input numpy 2D edge image array
        :param inp_h: The height of the input numpy 2D edge image array
        :param out_w: The width of the output numpy 2D noise image array
        :param out_h: The height of the output numpy 2D noise image array
        :param neurons: Number of neurons in the dense layer
        :param activation: Activation function for the dense layer
        :param loss: Loss calculation Method for backpropagation
        :param optimizer: Optimization method for error adjustment
        """
        self.inp_w = inp_w
        self.inp_h = inp_h
        self.inp_dim = (self.inp_w, self.inp_h, 1)
        self.neurons = neurons
        self.out_w = out_w
        self.out_h = out_h
        self.out_dim = self.out_w*self.out_h
        self.conv_filters = 32
        self.conv_ksize = conv_ksize
        self.pool_ksize = pool_ksize
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.train_epochs = 5  # Number of epochs for training with a single input
        # Directory for the neural network model and weight files
        self.model_save_url = "D:\\thesis\\ConvNet\\MyNet\\image_segmentation\\video_processing_software\\neural_net_model\\"
        self.model_file = self.model_save_url + "model.json"  # Model file path for storing the network
        self.model_weights_file = self.model_save_url + "weights.h5"  # Weight file path for storing training session
        self.graph = Graph()
        with self.graph.as_default():
            self.session = Session()
        # If the model file exists
            with self.session.as_default():
                self.estimator = KerasRegressor(build_fn=self.generate_model, epochs=self.train_epochs, verbose=1)

    def generate_model(self):
        if os.path.exists(self.model_file):
            # Load the network model from existing file
            with open(self.model_file, 'r', encoding='utf8') as f:
                self.model_json = f.read()
            self.model = model_from_json(self.model_json)
            #if os.path.exists(self.model_weights_file):
             #   self.model.load_weights(self.model_weights_file)
            print("Loaded Model")
        else:
            # Create a neural network model
            self.model = Sequential()
            self.model.add(Conv2D(self.conv_filters, (self.conv_ksize, self.conv_ksize),
                                  input_shape=self.inp_dim, activation='relu', border_mode='valid'))
            self.model.add(MaxPool2D(pool_size=(self.pool_ksize, self.pool_ksize)))
            self.model.add(Flatten())
            self.model.add(Dense(self.neurons, activation=self.activation))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(self.out_dim, activation=self.activation))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return self.model
    def train(self, img, label):
        """
        This function trains the initialized neural network model with a single binary edge image array and an expected
        noise output image label array
        :param img: The numpy 2D binary/grayscale edge image array
        :param label: The numpy 2D binary/grayscale expected noise label array
        :return: The numpy 2D binary/grayscale input edge image array and predicted noise label array
        """

        if len(img.shape) >= 2 and img.shape[-2:] == (self.inp_w, self.inp_h) and label.shape[-2:] == (self.out_h, self.out_w):
            K.set_session(self.session)
            with self.graph.as_default():
                print("Training")
                # If input parameters are valid
                if len(label.shape) >= 2 and label.shape[-2:] == (self.out_w, self.out_h):
                    # Reshape the image to one dimensional and scale it[WIll be replaced when convolution layer is added]
                    X = (img.reshape((-1, self.inp_w, self.inp_h, 1))).astype(np.float64)
                    if np.max(X) > 1:
                        X = X/255
                    y = (label.reshape(-1, self.out_w*self.out_h)).astype(np.float64)
                    if np.max(y) > 1:
                        y = y/np.max(y)
                    # Train the model

                    #self.model.fit(X, y, epochs=self.train_epochs, batch_size=1, verbose=2)

                    #results = cross_val_score(self.estimator, X, y)
                    self.estimator.fit(X, y)

                    #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
                    # Return output
                    return X.reshape(img.shape)
                else:
                    print("Invalid train output image dimension: Expected: " + str(self.out_w) + ", " + str(
                        self.out_h) + ", Found: " + str(label.shape))
        else:
            print("Invalid train input image dimension: Expected: " + str(self.inp_w) + ", " + str(self.inp_h) + ", Found: " + str(img.shape))
        return None, None
    def predict(self, img):
        """
        This function predicts a noise image array output for a provided input edge image array
        :param img: The numpy 2D binary/grayscale input edge image array
        :return: The numpy 2D binary/grayscale input edge image array and predicted noise label array
        """
        if len(img.shape) == 2 and img.shape == (self.inp_w, self.inp_h):
            K.set_session(self.session)
            with self.graph.as_default():
                print("predicting")
                X = (img.reshape((1, self.inp_w, self.inp_h, 1))).astype(np.float64)
                if np.max(X) > 1:
                    X = X/255
                prediction = self.estimator.predict(X)
                print("Prediction shape: " + str(prediction.shape))
                if prediction.size == self.out_dim:
                    return prediction.reshape((self.out_w, self.out_h)), X.reshape((self.inp_w, self.inp_h))
                else:
                    print("Invalid output image dimension: Expected: " + str(self.out_w) + ", " + str(
                        self.out_h) + ", Found: " + str(prediction.shape))
        else:
            print("Invalid input image dimension: Expected: " + str(self.inp_w) + ", " + str(self.inp_h) + ", Found: " + str(img.shape))
        return None, None
    def save_model(self):
        K.set_session(self.session)
        with self.graph.as_default():
            model_json = self.model.to_json()
            with open(self.model_file, 'w',encoding='utf8') as f:
                f.write(model_json)
            self.model.save_weights(self.model_weights_file)



class CommonInstance(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.should_run = False  # Checks if the main loop of the thread should run
        self.thread_sleep_time_sec = 0.2

    def __del__(self):
        pass

    def run(self):
        pass
    def stop_thread(self):
        pass
class ImageAnnotationController(CommonInstance):
    """
    This is the Image Annotation Interface that runs on a separate thread and can annotate Images either loaded from
    a drive or frames that appear when streaming with a video input device like WebcamPlayerController by acquiring
    the frames stored in the main thread.
    A. System defined Tasks:
    1. Initiated by the main thread and stopped by the main thread.
    2. Runs in a loop while active.
    3. If 'Frame Input Mode' is active then:
        3.1 Deactivate Image Annotation Mode
        3.2 Receive a stored frame object from the main thread
        3.3 If the frame is available then calculate the following features from the frame:
            3.2.1 RGB COLOR IMAGE(Stored in frame)
            3.2.2 ORIENTATION MAP(Histogram of Oriented Gradients)
            3.2.3 Edge Detection(Stored in the frame)
            3.2.4 DISTANCE MAP(From the Edge image-Euclidean Distance from the edges)
            3.2.5 CLUSTERS OF SEGMENTED IMAGE(Watershed Algorithm with mean intensity of distance map as markers and
                  K-Means Clustering)
            3.2.6 FEATURE DESCRIPTORS(Oriented Fast and Rotated Brief Algorithm)
            3.2.7 REGION PROPERTIES(From Clusters of segmented image)
        3.4 Display the features in GUI
        3.5 Set the last frame processed objects as the list of features obtained
    4. Else if the FIM is inactive then:
        4.1 Update the GUI with last frame processed objects(if available).
        4.2 Activate Image Annotation Mode for the user to interact with the loaded image
    5. Deactivate Image Annotation Mode and FIM when the loop ends

    B. User defined Tasks:
    1. Choose an Image from the calculated features in order to guide annotation(Annotate Image View Option-ComboBox).
    2. Load an Image from computer in order to annotate. The loaded image undergoes the complete feature extraction
       steps(3.3 and 3.4) before being displayed in the GUI and replaces last processed feature object.
    3. Create a window for Annotating the last processed frames with GUI(Main Image Annotation GUI Module).
    4. View Region Property parameters of different clusters of the segmented image.
    5. Save annotated image as a whole(In PASCAL VOC FORMAT) along with the features obtained from the image as numpy
       array or bytes in a 'Data Hierarchical Structure'.
    6. Save only annotated portions of the image(with a black background) as separate data files.


    C. Image Annotation Functionality:
    1. Draw boundary around an object(in the image). The boundary might be rectangular, elliptical, Polygonal(Point based)
       or Freehand(Drawing a closed contour with cursor).
    2. Provide each contour one or more label in a 'Data Hierarchical Structure'.
    3. Save the annotated segments individually with a colored(black) background for region not belonging
       to the annotated segment along with its separate features.
       or as a whole image along with the complete feature list object.
    4. Save images or features as numpy array(or byte) and annotation information(PASCAL VOC Format) as XML or JSON file

    D. Saving an individual Annotated Segment of the image:
    1. Take the annotated segment which is being saved.
    2. Put it infront of black background.
    3. Label the background as 0 and their corresponding features as null(or zero in most cases).
    4. Save the recalculated features and corresponding files in the 'Data Hierarchical Structure'

    E. Saving whole image with multiple annotations and segments:
    1. Take the whole image.
    2. Save the features and corresponding files as multiple copies for each annotated segment in'Data Hierarchical Structure'


    F. DATA HIERARCHICAL STRUCTURE
    1. Each annotated image has the following files:
       a. A set of Feature files(Numpy arrays)
       b. An annotation file(JSON or XML file according to PASCAL VOC FORMAT) where each segment contains a id
          (starting from 0 as the background)
        e.g.
       <annotation>
            <folder>GeneratedData_Train</folder>
            <filename>000001.png</filename>
            <path>/my/path/GeneratedData_Train/000001.png</path>
            <source>
                <database>Unknown</database>
            </source>
            <size>
                <width>224</width>
                <height>224</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            <object>
                <name>21</name> <!-- ID of the segemnt/object -->
                <pose>Frontal</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <occluded>0</occluded>
                <bndbox>
                    <xmin>82</xmin>
                    <xmax>172</xmax>
                    <ymin>88</ymin>
                    <ymax>146</ymax>
                </bndbox>
            </object>
        </annotation>
       c. A label file containing labels in a hierarchical structure for each corresponding segment ID where each
          level of the hierarchy corresponds to similarity in context.

          e.g.
          {
            "21" : {
                "value": {
                    "physical attributes": ["Fruit", "Apple", "Red", "Circular", "Elliptical", "Circular"],
                    "physical condition": ["Eaten", "Chewed", "fresh", "Ripe"]
                }
            },
            "22": {
                "value": {
                    "physical properties": ["Table", "Wooden", "Rectangular", "brown", "Furniture"],
                    "visual condition": ["Filled with food", "Low Height"]
                }
            }

          }
        d. Complete folder structure for an annotated Image:
           Example:
               current time: 11:15:32AM
               current date: 12-12-2018
               current user_id: 1
           1. 11_15_32_12_12_2018_1 (Folder)
               1.1 RGB/GrayScale Image(Numpy Array)
               1.2 Orientation Map Image(Numpy Array-GrayScale)
               1.3 Orientation Map Features(Numpy Array-Optional)
               1.4 Edge Map Image(Numpy Array-Binary)
               1.5 Clusters of the segmentation(List)
               1.6 Region Properties(List of Numpy Array-For the clusters)
               1.7 Feature Descriptors(ORB Object)
               1.8 Annotation File(XML-PASCAL VOC)
               1.9 Label File(JSON-Data Hierarchical Structure)
    """
    def __init__(self, main_thread, image_annotate_interface:ImageAnnotationLayout):
        super().__init__()
        self.main_thread = main_thread
        self.layout = image_annotate_interface


        self.frame_width = self.layout.default_frame_width
        self.frame_height = self.layout.default_frame_height
        self.hog_cells_per_block = self.layout.hog_cells_per_block
        self.hog_pixels_per_cell = self.layout.hog_pixels_per_cell
        self.hog_orientation = self.layout.hog_orientation
        self.kmeans_clusters = self.layout.kmeans_clusters
        self.orb_features = self.layout.orb_features
        self.hold_frame = self.layout.image_opt_widgets['hold frame']['widget'].isChecked()
        self.img_mode = self.layout.image_opt_widgets['image mode']['widget'].currentText()
        self.img_save_dir = self.layout.base_img_save_dir
        self.annotation_view_mode = str(self.layout.annotate_opt_widgets['annotation view mode']['widget'].currentText())

        self.last_updated_features = None




    def stop_thread(self):
        self.should_run = False

    def load_image(self):
        if self.hold_frame:
            url = str(self.layout.image_opt_widgets['load image path']['widget'].text())
            if os.path.exists(url):
                img = cv2.imread(url)
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                edge = cv2.Canny(gray, 50, 50)

                fd, hog_image, dt, local_max, markers, labels, regions, region_means, classified_labels, orb_kp, orb_fd, orb_img = self.calculate_features(img, gray, edge)
                self.update_features(img, gray, edge, fd, hog_image, dt, local_max, markers, labels, regions, region_means,
                                     classified_labels, orb_kp, orb_fd, orb_img)
            print("Loading Image from " + url)
        else:
            print("Hold frame to load image")

    def set_hog_cpb(self):
        try:
            cpb = int(self.layout.image_opt_widgets['hog cells per block']['widget'].text())
            if cpb > 0:
                self.hog_cells_per_block = cpb
        except:
            pass
    def set_hog_ppc(self):
        try:
            ppc = int(self.layout.image_opt_widgets['hog pixels per cell']['widget'].text())
            if ppc > 0:
                self.hog_pixels_per_cell = ppc
        except:
            pass
    def set_hog_orientation(self):
        try:
            ori = int(self.layout.image_opt_widgets['hog orientations']['widget'].text())
            if ori > 0:
                self.hog_orientation = ori
        except:
            pass
    def set_orb_features(self):
        try:
            orb = int(self.layout.image_opt_widgets['total orb features']['widget'].text())
            if orb > 0:
                self.orb_features = orb
        except:
            pass
    def set_hold_frame(self):
        self.hold_frame = self.layout.image_opt_widgets['hold frame']['widget'].isChecked()
    def set_img_width(self):
        try:
            img_w = int(self.layout.image_opt_widgets['image width']['widget'].text())
            if img_w > 0:
                self.frame_width = img_w
        except:
            pass
    def set_img_height(self):
        try:
            img_h = int(self.layout.image_opt_widgets['image height']['widget'].text())
            if img_h > 0:
                self.frame_height = img_h
        except:
            pass
    def set_img_mode(self):
        self.img_mode = str(self.layout.image_opt_widgets['image mode']['widget'].currentText())
    def set_annotation_view_mode(self):
        self.annotation_view_mode = str(self.layout.annotate_opt_widgets['annotation view mode']['widget'].currentText())
    def start_annotation_btn(self):
        print("Starting Image Annotation")
        if not (self.last_updated_features is None) and self.hold_frame:
            self.layout.annotation_window.set_annotation_image(self.last_updated_features[0])
            self.layout.annotation_window.show()
    def regionprop_view_btn(self):
        print("Viewing region properties")
    def save_annotation_btn(self):
        print("saving annotation at " + str(self.layout.save_opt_widgets['base directory']['widget'].text()))

    def calculate_features(self, frame, gray, edge):
        fd, hog_image = feature.hog(gray, orientations=self.hog_orientation,
                                    pixels_per_cell=(self.hog_pixels_per_cell, self.hog_pixels_per_cell),
                                    cells_per_block=(self.hog_cells_per_block, self.hog_cells_per_block),
                                    visualize=True, multichannel=False, feature_vector=False)
        # result_view_frame.append(hog_image.copy())
        # result_view_frame.append(fd)

        dt = distance_transform_edt((~edge))  # Euclidean distance from edge
        # result_view_frame.append(dt.copy())
        # dt = cv2.erode(dt, np.ones((10, 10)), iterations=1)
        # Local Peaks by calculating the Euclidean distance map
        # (Create automated markers for watershed background extraction)
        local_max = feature.peak_local_max(dt, indices=False, min_distance=5)
        # Create the markers from the local peaks(Connected component from the local max map)
        markers = measure.label(local_max)
        # Watershed algorithm to segment connected components from the negative Euclidean distance map
        labels = watershed(-dt, markers)

        # Calculate Region Properties from the watershed segmented map
        regions = measure.regionprops(labels, intensity_image=gray)

        # result_view_frame.append(regions)
        # Calculate Mean intensities for each region of the  watershed segmented map
        # in order for background segmentation using KMeans Clustering
        region_means = [
            [r.local_centroid[0], r.local_centroid[1], np.average(frame[r.coords[:, 0], r.coords[:, 1], 0]),
             np.average(frame[r.coords[:, 0], r.coords[:, 1], 1]),
             np.average(frame[r.coords[:, 0], r.coords[:, 1], 2])]
            for r in regions]
        # Reshape the mean intensity array as single column
        region_means = np.asarray(region_means).reshape(-1, len(region_means[0]))

        classified_labels = None
        """if region_means.size > 0 and len(regions) >= self.kmeans_clusters - 1:
            # Fit mean intensities of the watershed background map as a feature vector of the model
            model = KMeans(n_clusters=self.kmeans_clusters)
            model.fit(region_means)
            # Predict foreground and background labels for mean intensity of each region of the watershed segmented map
            kmeans_segmentation = model.predict(region_means)
            # Get a copy of the watershed segmented map of different connected components in the image
            classified_labels = labels.copy()
            # For each label/cluster(background or foreground) of mean intensity of each connected component
            for bg_fg, region in zip(kmeans_segmentation, regions):
                # Set the color of the region as predicted label(foreground or background)
                classified_labels[tuple(region.coords.T)] = bg_fg"""
        # result_view_frame.append(classified_labels)

        orb = cv2.ORB_create(nfeatures=self.orb_features)
        orb_kp, orb_fd = orb.detectAndCompute(gray, None)
        orb_img = cv2.drawKeypoints(gray, orb_kp, None)
        return fd, hog_image, dt, local_max, markers, labels, regions, region_means, classified_labels, orb_kp, orb_fd, orb_img

    def update_features(self, frame, gray, edge, fd, hog_image, dt, local_max, markers, labels, regions, region_means, classified_labels, orb_kp, orb_fd, orb_img):
        # self.annotation_img_view_options = ['color', 'edges', 'orientation', 'distance', 'clusters', 'orb']
        self.layout.view_widgets['color']['widget'].update_image(frame)
        self.layout.view_widgets['edges']['widget'].update_image(
            ((edge / edge.max()) * 255).astype(np.uint8)
        )
        self.layout.view_widgets['orientation']['widget'].update_image(
            ((exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max()))) * 255).astype(np.uint8)
        )

        self.layout.view_widgets['distance']['widget'].update_image(
            ((exposure.rescale_intensity(dt, in_range=(dt.min(), dt.max()))) * 255).astype(np.uint8)
        )
        """if not (classified_labels is None):
            self.layout.view_widgets['clusters']['widget'].update_image(
                ((classified_labels.astype(np.float64) / classified_labels.max()) * 255).astype(np.uint8)
            )"""
        self.layout.view_widgets['orb']['widget'].update_image(orb_img)
    def run(self):
        print("Starting Image Annotation Thread")
        self.should_run = True

        while self.should_run:
            if not self.hold_frame:
                # frame_object = [raw_frame, frame, gray_image, edge_image, contours, hierarchy]
                current_frame = self.main_thread.get_current_frame()

                if not (current_frame is None):

                    current_frame = current_frame.copy()
                    # Result frame containing features: [raw frame(RGB/Gray), Edge frame(Binary), Orientation Frame(Gray),
                    # orientation features(List), Distance Map(Gray), Region Properties(List),
                    #Cluster Image(RGB), Feature Descriptor(Array), Feature Descriptor Image(Array)]
                    #result_view_frame = []
                    frame = cv2.resize(current_frame[1], (self.frame_width, self.frame_height), interpolation=cv2.INTER_CUBIC).copy()
                    #result_view_frame.append(frame)
                    gray = cv2.resize(current_frame[2], (self.frame_width, self.frame_height), interpolation=cv2.INTER_CUBIC).copy()
                    gray = exposure.equalize_hist(gray)
                    gray = median(gray, selem=np.ones((5, 5)))
                    edge = cv2.resize(current_frame[3], (self.frame_width, self.frame_height), interpolation=cv2.INTER_CUBIC).copy()

                    fd, hog_image, dt, local_max, markers, labels, regions, region_means, classified_labels, orb_kp, orb_fd, orb_img = self.calculate_features(frame, gray, edge)
                    #result_view_frame.append(edge)


                    #result_view_frame.append((orb_kp, orb_fd))
                    #result_view_frame.append(orb_img)

                    self.update_features(frame, gray, edge, fd, hog_image, dt, local_max, markers, labels, regions, region_means, classified_labels, orb_kp, orb_fd, orb_img)

                    #self.last_updated_features = result_view_frame
                    self.last_updated_features = [frame, gray, edge]
            else:
                # Result frame containing features: [raw frame(RGB/Gray), Edge frame(Binary), Orientation Frame(Gray),
                # orientation features(List), Distance Map(Gray), Region Properties(List),
                # Cluster Image(RGB), Feature Descriptor(Array), Feature Descriptor Image(Array)]
                """if not (self.last_updated_features is None):
                    self.layout.view_widgets['color']['widget'].update_image(result_view_frame[0])
                    self.layout.view_widgets['edges']['widget'].update_image(
                        ((result_view_frame[1] / result_view_frame[1].max()) * 255).astype(np.uint8)
                    )
                    self.layout.view_widgets['orientation']['widget'].update_image(
                        ((exposure.rescale_intensity(result_view_frame[2], in_range=(0, result_view_frame[2].max()))) * 255).astype(np.uint8)
                    )

                    self.layout.view_widgets['distance']['widget'].update_image(
                        ((exposure.rescale_intensity(result_view_frame[4], in_range=(result_view_frame[4].min(), result_view_frame[4].max()))) * 255).astype(np.uint8)
                    )
                    if not (classified_labels is None):
                        self.layout.view_widgets['clusters']['widget'].update_image(
                            ((result_view_frame[6].astype(np.float64) / result_view_frame[6].max()) * 255).astype(np.uint8)
                        )
                    self.layout.view_widgets['orb']['widget'].update_image(result_view_frame[8])"""

                pass


            self.sleep(self.thread_sleep_time_sec)

class VideoProcessingController(CommonInstance):
    """
    This is the Video Processing interface that runs on a separate thread and uses Image Processing on Images
    stored in the main thread.
    Tasks:
    1. Initiated by the main thread and stopped by the main thread.
    2. Runs in a loop while active.
    3. Edge linking and noise reduction: Links edges and reduces possible noise components which might be present
       due to different external factors like lighting condition of the environment, quality of the webcam, etc.
       [N.B: No implemented module found in python for these taks. Created a custom algorithm for edge linking but
       needs speeding up in order to support realtime image processing of webcam video feed. Trying to use simple
       neural networks on the edge images in order to identify pixels in the edge image file which might possibly
       be a noise rather than a real edge. This problem is faced when there are constant little fluctuations in the
       edge stream's display window due to light reflecting off from objects; even skin surface. WILL BE SHIFTED TO
       WEBCAMPLAYER CLASS WHEN A FAST ALGORITHM IS FOUND.]
    4. Contour(Boundary detection): This function tries to detect boundary in an image file containing binary edges
       of a grayscale image. It is performed in order to identify each separate object in an image. The boundary
       detection algorithm creates clusters of pixels within the image which might represent the same object because
       they lie within the same edge boundary. This functions converts a simple edge image into a map of potential
       objects in an image.
    5. Thresholding(Foreground and Background Separation): This function implements different algorithms like
       Otsus's Algorithm for trying to separate background from foreground in a grayscale image based on the global
       and local intensity or brightness of each pixel in the image. This is done in order to perform image
       segmentation. This algorithm creates multiple set of pixels which might represent the same region of an
       object in an image.
    """
    store_result_signal = pyqtSignal(list)
    def __init__(self, main_thread, video_proc_int):
        super().__init__()
        self.video_proc_int = video_proc_int
        #self.thread_sleep_time_sec = 1
        self.main_thread = main_thread
        self.plot_histogram = True  # Toggle Original Webcam frame's histogram plot window(Matplotlib)
        self.plot_contours = False  # Toggle Original Webcam frame's contour plot for boundary detection in edges
        self.plot_thresholding = False  # Toggle Original Webcam frame's threshold/ForegroundExtraction view
        self.plot_neural_net_output = False  # Toggle plot for neural network's prediction of noise in a captured edge image

        # Contour Object settings
        self.display_contours_amount = 10  # The number of identified potential objects(using boundary detection) to be displayed
        self.contour_object_plot_cols = 2  # The number of columns in the plot of potential identified object images

        # Neural network settings
        self.nn_inp_w = 200  # The width of input edge image to the neural network
        self.nn_inp_h = 200  # The height of input edge image to the neural network
        self.nn_out_w = 200  # The width of output noise image from the neural network
        self.nn_out_h = 200  # The height of output noise image from the neural network
        self.nn_neurons = 100  # The number of neruons in the dense layer of the neural network
        self.nn_conv_kszie = 3
        self.nn_pool_ksize = 3
        self.nn_dropout = 0.4
        # The neural network class that detects possible noise pixels in an edge image
        self.neural_net = MyNeuralNetwork(self.nn_inp_w, self.nn_inp_h, self.nn_out_w, self.nn_out_h, self.nn_neurons, self.nn_conv_kszie, self.nn_pool_ksize)



    def get_slic_superpixels(self, img_rgb, seg=100, sigma=5, mark_boundary=False):
        """
        This function computes superpixels for an RGB image using SLIC(Scikit module)
        :param img_rgb: The 3D Numpy array for a RGB image
        :param seg: The number of segments to use for superpixels
        :return: The superpixels array of he image
        """
        img_float = img_as_float(img_rgb)
        # Apply SLIC and extract(approximately) the supplied number of segments
        segments = slic(img_float, n_segments=seg, sigma=sigma)
        if mark_boundary:
            result = mark_boundaries(img_float, segments)
            return result, segments
        else:
            return img_float, segments

    def stop_thread(self):
        """
        This function is called when the thread needs to be stopped
        :return: null
        """
        self.should_run = False

    def run(self):
        print("Starting Video Processing Thread")
        self.should_run = True
        while self.should_run:
            # frame_object = [raw_frame, frame, gray_image, edge_image, contours, hierarchy]
            current_frame = self.main_thread.get_current_frame().copy()

            if not (current_frame is None):
                result_view_frame = []

                # Display Objects detected using Contour Finding Algorithm
                # Get a copy of the contourss frame or boundary map captured from the webcam player
                contours = current_frame[4].copy()
                edge_frame = current_frame[3].copy()
                gray_image = current_frame[2].copy()

                frame = current_frame[1].copy()
                # If identified objects in the edge frame is available
                if self.plot_contours and len(contours) > 0:
                    print("Obtaining contours")
                    # Get area of each contour/object in the image. Each object is assumed to be in a rectangular region
                    contour_sizes = [cv2.boundingRect(contours[i])[2] * cv2.boundingRect(contours[i])[3] for i in
                                     range(len(contours))]
                    self.sleep(0.01)
                    # Sort the objects by ascending area
                    contour_order = np.argsort(contour_sizes)
                    # Get number of contours to plot
                    total_plots = self.display_contours_amount
                    if len(contours) < self.display_contours_amount:
                        total_plots = len(contours)
                    identified_objects = []

                    for i in range(len(contour_order) - 1, len(contour_order) - total_plots - 1, -1):
                        #print("Obtaining object from contour number: " + str(i) + " out of " + str(len(contour_order) - total_plots - 1))
                        (x, y, w, h) = cv2.boundingRect(contours[contour_order[i]])
                        identified_objects.append(frame[y:y+h, x:x+w])
                        self.sleep(0.01)
                    result_view_frame.append(identified_objects)
                    #self.sleep(100)
                else:
                    result_view_frame.append(None)
                if self.plot_thresholding:
                    # Extract foreground detected binary image file where 2 represents foreground and 0 represents
                    # background. It uses Otsus's algorithm

                    ret, thresholded_frame = cv2.threshold(
                        gray_image,
                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Convert array to binary format from grayscale
                    gray_image[thresholded_frame > 0] = 0
                    result_view_frame.append(gray_image)
                    #self.sleep(100)
                else:
                    result_view_frame.append(None)

                if self.plot_histogram:

                    result_view_frame.append(frame)
                else:
                    result_view_frame.append(None)

                video_proc_int = self.video_proc_int
                # result_view_frame = [identified_objects[]/None, thresholded_gray_image/none, frame for histogram creation/None]
                # Display identified objects
                objects = result_view_frame[0]
                if not (objects is None) and self.plot_contours:
                    video_proc_int.contour_object_detection_module.view_controls["object"].plot(objects)
                    self.sleep(0.01)
                    #self.sleep(100)
                # Display Thresholded image
                thresholded = result_view_frame[1]
                if not (thresholded is None) and self.plot_thresholding:
                    video_proc_int.threshold_histogram_module.view_controls["threshold"].plot([thresholded])
                    self.sleep(0.01)
                    #self.sleep(100)
                # Display Histogram
                gray_image = result_view_frame[2]
                if not (gray_image is None) and self.plot_histogram:
                    video_proc_int.threshold_histogram_module.view_controls["histogram"].plot([gray_image],
                                                                                              histogram=True)
                    self.sleep(0.01)

                 # Display Noise Estimation
                last_frames = self.main_thread.get_last_frame_queue(nmax=10).copy()

                if not (last_frames is None) and self.plot_neural_net_output:

                    current_edge_frame = edge_frame.copy()

                    last_edge_frames = [last_frames[i][3].copy() for i in range(len(last_frames))] + [current_edge_frame]

                    results = [compare_ssim(last_edge_frames[i-1], last_edge_frames[i], full=True) for i in range(1, len(last_edge_frames), 1)]
                    self.sleep(0.01)
                    current_edge_labels = [results[i][1] for i in range(len(results))]
                    current_edge_inputs = last_edge_frames[1:]

                    #if np.max(current_edge_inputs) > 1:
                     #   current_edge_inputs = np.divide(current_edge_inputs, 255)
                    #if np.max(current_edge_labels) > 1:
                     #   current_edge_labels = np.divide(current_edge_labels, np.max(np.abs(current_edge_labels)))
                    #self.sleep(100)

                    self.neural_net.train(
                        np.asarray(current_edge_inputs), np.asarray(current_edge_labels))
                    #self.sleep(100)
                    self.sleep(0.01)

                    prediction, inp = self.neural_net.predict(
                        np.asarray(current_edge_inputs[-1]))
                    # Resize the predicted noise image for display
                    prediction = cv2.resize(prediction, current_edge_frame.shape, interpolation=cv2.INTER_CUBIC)

                    video_proc_int.noise_detection_module.input_display_canvas.plot(
                        [cv2.resize(inp, current_edge_frame.shape, interpolation=cv2.INTER_CUBIC)]
                    )
                    self.sleep(0.01)
                    video_proc_int.noise_detection_module.expected_out_display_canvas.plot(
                        [current_edge_labels[-1]]
                    )
                    video_proc_int.noise_detection_module.predicted_out_display_canvas.plot(
                        [prediction]
                    )
                    self.sleep(0.01)


            self.sleep(self.thread_sleep_time_sec)

class WebcamPlayerController(CommonInstance):
    """
    This is the WebcamPlayer Interface that runs on a separate thread and manages webcam video feed.
    Tasks:
    1. Initiated by the main thread and stopped by the main thread
    2. Runs in a loop while active.
    3. Adjusted automated parameter option(if enabled) if certain time have passed.
    4. Capture frame from webcam device if stream is enabled(not stopped or paused).
    5. Adjust brightness, Hue, Saturation, Contrast according to specified parameter for each frame.
    6. Create grayscale copy of the adjusted frame.
    7. Detect edges from the grayscale frame according to specified parameters(Algorithm, threshold).
    8. Detect contours from the edge image frame according to specified parameters(Min Contour size)
    9. Create current frame object containing copies of original frame, grayscale frame, edge frame, contour and
       hierarchy frame.
    10. Send the current frame object to main thread for displaying, storing and adding to last acquired frame list.

    11. Release captured resource when loop finishes

    """

    store_frame_signal = pyqtSignal(list)
    def __init__(self, or_frame_w=200, or_frame_h=200, ed_frame_w=200, ed_frame_h=200):
        super().__init__()
        self.original_frame_width = or_frame_w
        self.original_frame_height = or_frame_h
        self.edge_frame_width = ed_frame_w
        self.edge_frame_height = ed_frame_h
        # The amount by which grayscale file will be reduced before detecting edge. This is done in order to control
        # edge detection perspective dynamically simulating the action of pupil in the eye
        self.edge_size_pct = 1.0
        self.edge_size_min = 0.3
        self.edge_size_max = 1.0
        self.edge_size_increase = 0.1
        self.perform_automatic_param_adjustment = False

        # HSV and Contrast Settings
        self.contrast_level = 0
        self.brightness_level = 0
        self.sharpness_level = 0

        self.edge_detection_algorithms = {
            "canny": self.my_canny,
            "sobel": self.my_sobel,
            "laplacian": self.my_laplacian
        }
        self.selected_edge_algorithm = "canny"
        # Thresholds for canny, sobel and laplacian edge detector
        self.canny_thresh1 = 10
        self.canny_thresh2 = 80
        self.laplacian_ksize = 3
        self.sobel_ksize = 3
        self.erosiona_ksize = 0
        self.dilationa_ksize = 0
        self.erosionb_ksize = 0
        self.dilationb_ksize = 0
        # Contour Setting
        self.contour_display_enabled = False
        self.contour_min_object_size = 50
        # Parameter Change setting
        self.parameter_change_duration = 100  # ms
        self.last_param_updated = -1

    def draw_bounding_box(self, img, contours, hierarchy):
        """
        This function draws a bounding box around identified contours in a binary edge image array
        :param img: The numpy 2D binary/grayscale edge image array
        :param contours: The list of numpy 2D binary edge pixels of contours
        :param hierarchy: The hierarchy of the identified contours in the edge image
        :return: The numy 2D binary/grayscale edge image with rectangle drawn around the contours.
        """
        height, width = img.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        for contour, hier in zip(contours, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            if w > self.contour_min_object_size or h > self.contour_min_object_size:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # if max_x - min_x > 0 and max_y - min_y > 0:
        #   cv2.rectangle(img_canny, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        return img
    def my_canny(self, img):
        edge_image = Canny(
            img.copy(),
            self.canny_thresh1, self.canny_thresh2
        )
        return edge_image

    def my_sobel(self, img):
        if self.sobel_ksize < 0:
            self.sobel_ksize = 3
        elif self.sobel_ksize == 0 or self.sobel_ksize%2 == 0:
            self.sobel_ksize += 1
        sobelx = Sobel(
            img.copy(),
            cv2.CV_64F, 1, 0, ksize=self.sobel_ksize
        )
        sobely = Sobel(
            img.copy(),
            cv2.CV_64F, 0, 1, ksize=self.sobel_ksize
        )
        edge_image = np.sqrt(sobelx**2 + sobely**2).astype(img.dtype)
        return edge_image

    def my_laplacian(self, img):
        if self.laplacian_ksize < 0:
            self.laplacian_ksize = 3
        elif self.laplacian_ksize == 0 or self.laplacian_ksize%2 == 0:
            self.laplacian_ksize += 1
        edge_image = Laplacian(
            img.copy(),-1, ksize=self.laplacian_ksize
        )
        return edge_image
    def get_thresholds(self, edge_detector):
        val = str(edge_detector).lower()
        if val == "canny":
            return str(self.canny_thresh1) + "," + str(self.canny_thresh2)
        elif val == "sobel":
            return str(self.sobel_ksize)
        elif val == "laplacian":
            return str(self.laplacian_ksize)
    def run(self):
        """
        This is the main function that runs while the thread is active. It performs different tasks like webcam feed
        capture, edge and boundary detection, webcam and edge feed display, etc.

        :return: null
        """
        print("Starting Webcam Player Thread")
        self.should_run = True
        cap = cv2.VideoCapture(0)  # Get an available webcam resource connected to the device
        cap.set(4, self.original_frame_width)
        cap.set(3, self.original_frame_height)
        add = True  # Checks whether to increase perspective of the edge detection procedure or decrease it
        while self.should_run:
            # Parameter Processing: Adjust dynamic parameters like perspective size before performing edge detection, etc.
            if self.perform_automatic_param_adjustment:
                current_time = int(round(time.time() * 1000))
                if (current_time - self.last_param_updated > self.parameter_change_duration):
                    if self.edge_size_pct - self.edge_size_increase < self.edge_size_min:
                        add = True
                    elif self.edge_size_pct + self.edge_size_increase > self.edge_size_max:
                        add = False
                    if add:
                        self.edge_size_pct += self.edge_size_increase
                    else:
                        self.edge_size_pct -= self.edge_size_increase
                    self.last_param_updated = current_time

            # Original Frame Brightness processing
            ret, raw_frame = cap.read()  # Capture a frame in RGB format from the webcam
            frame = raw_frame.copy()
            #if frame.shape != (self.original_frame_width, self.original_frame_height):
             #   frame = cv2.resize(frame, (self.original_frame_width, self.original_frame_height))
            if self.contrast_level != 0:
                frame = Image.fromarray(frame)
                contrast = ImageEnhance.Contrast(frame)
                frame = contrast.enhance(self.contrast_level)
                frame = np.array(frame)

            if self.brightness_level != 0:
                frame = Image.fromarray(frame)
                brightness = ImageEnhance.Brightness(frame)
                frame = brightness.enhance(self.brightness_level)
                frame = np.array(frame)

            if self.sharpness_level != 0:
                frame = Image.fromarray(frame)
                sharpness = ImageEnhance.Sharpness(frame)
                frame = np.array(sharpness.enhance(self.sharpness_level))

            # Edge Processing
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            if self.edge_size_pct < 1:
                resized_gray = cv2.resize(gray_image, (0,0), None, self.edge_size_pct, self.edge_size_pct)
            else:
                resized_gray = gray_image.copy()
            edge_image = self.edge_detection_algorithms[self.selected_edge_algorithm](resized_gray)  # Calculate edges
            if edge_image.shape != (self.edge_frame_width, self.edge_frame_height):
                edge_image = cv2.resize(edge_image, (self.edge_frame_width, self.edge_frame_height), interpolation=cv2.INTER_CUBIC)

            # Erosion and dilation
            if self.erosiona_ksize > 0:
                edge_image = cv2.erode(edge_image, np.ones((self.erosiona_ksize, self.erosiona_ksize), np.uint8), iterations=1)
            if self.dilationa_ksize > 0:
                edge_image = cv2.dilate(edge_image, np.ones((self.dilationa_ksize, self.dilationa_ksize), np.uint8), iterations=1)
            if self.erosionb_ksize > 0:
                edge_image = cv2.erode(edge_image, np.ones((self.erosionb_ksize, self.erosionb_ksize), np.uint8), iterations=1)
            if self.dilationb_ksize > 0:
                edge_image = cv2.dilate(edge_image, np.ones((self.dilationb_ksize, self.dilationb_ksize), np.uint8), iterations=1)
            # Find Contours and Hierarchy from the edge image
            _, contours, hierarchy = cv2.findContours(
                edge_image.copy(),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            try:
                hierarchy = hierarchy[0]
            except:
                hierarchy = []

            # Create frame object to send to the main thread
            frame_object = [raw_frame, frame, gray_image, edge_image, contours, hierarchy]
            self.store_frame_signal.emit(frame_object)
            self.sleep(self.thread_sleep_time_sec)
        cap.release()
        print("Webcam Player thread complete")
    def stop_thread(self):
        """
        This function is called when the thread needs to be stopped
        :return: null
        """
        self.should_run = False
def perform_image_segmentation():

    import matplotlib.pyplot as plt


    cap = cv2.VideoCapture(0)
    cap.set(3, 400)
    cap.set(4, 400)
    clusters = 20
    #fig = plt.figure(1)
    #ax = fig.subplots(2, 2)
    #fig.suptitle("Gray, Blue, Green, Red")
    model = KMeans(n_clusters=clusters)
    orb = cv2.ORB_create(nfeatures=100)
    last_frame = None
    current = int(round(time.time() * 1000))
    update_duration = 2000
    #plt.ion()
    #plt.show(block=False)


    while True:
        ret, frame = cap.read()
        contrast = ImageEnhance.Contrast(Image.fromarray(frame))
        frame = np.array(contrast.enhance(2))

        #frame = gaussian_filter(frame, 5, mode='nearest')
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = exposure.equalize_hist(gray)
        gray = median(gray, selem=np.ones((5, 5)))


        edge = Canny(gray, 50, 50)

        # Calculate Histogram of Oriented Gradients
        fd, hog_image = feature.hog(gray, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=False)


        orb_kp, orb_fd = orb.detectAndCompute(gray, None)

        if not (last_frame is None):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(orb_fd, last_frame[2].copy())
            matches = sorted(matches, key=lambda x: x.distance)
            img_comp = cv2.drawMatches(gray.copy(), orb_kp.copy(), last_frame[0], last_frame[1], matches[:10], None)



        dt = distance_transform_edt((~edge))  # Euclidean distance from edge
        #dt = cv2.erode(dt, np.ones((10, 10)), iterations=1)
        # Local Peaks by calculating the Euclidean distance map
        # (Create automated markers for watershed background extraction)
        local_max = feature.peak_local_max(dt, indices=False, min_distance=5)
        # Create the markers from the local peaks(Connected component from the local max map)
        markers = measure.label(local_max)
        # Watershed algorithm to segment connected components from the negative Euclidean distance map
        labels = watershed(-dt, markers)
        # Calculate Region Properties from the watershed segmented map
        regions = measure.regionprops(labels, intensity_image=gray)

        # Calculate Mean intensities for each region of the  watershed segmented map
        # in order for background segmentation using KMeans Clustering
        region_means = [
            [r.local_centroid[0], r.local_centroid[1], np.average(frame[r.coords[:, 0], r.coords[:, 1], 0]),
             np.average(frame[r.coords[:, 0], r.coords[:, 1], 1]), np.average(frame[r.coords[:, 0], r.coords[:, 1], 2])]
                        for r in regions]
        # Reshape the mean intensity array as single column
        region_means = np.asarray(region_means).reshape(-1, len(region_means[0]))
        if region_means.size > 0 and np.count_nonzero(region_means) >= clusters-1:
            # Fit mean intensities of the watershed background map as a feature vector of the model
            model.fit(region_means)
            # Predict foreground and background labels for mean intensity of each region of the watershed segmented map
            kmeans_segmentation = model.predict(region_means)
            # Get a copy of the watershed segmented map of different connected components in the image
            classified_labels = labels.copy()
            # For each label/cluster(background or foreground) of mean intensity of each connected component
            for bg_fg, region in zip(kmeans_segmentation, regions):
                # Set the color of the region as predicted label(foreground or background)
                classified_labels[tuple(region.coords.T)] = bg_fg

        #print(str(type(local_max)) + " " + str(local_max.shape) + " max: " + str(local_max.max()))
        #segments = slic(frame, n_segments=100, compactness=10, sigma=1)
        #segments = felzenszwalb(gray, scale=100, sigma=0.5, min_size=100)
        import pdb
        pdb.set_trace()
        cv2.imshow("Original", frame)
        #cv2.imshow("Superpixels", mark_boundaries(frame, segments))
        #cv2.imshow("Gray", gray)
        cv2.imshow("HOG", exposure.rescale_intensity(hog_image, in_range=(0,10)))
        #cv2.imshow("ORB", orb_img)

        n = int(round(time.time() * 1000))
        if n - current > update_duration:
            if not (last_frame is None):
                cv2.imshow("ORB", img_comp)
            current = n
            last_frame = [gray.copy(), orb_kp.copy(), orb_fd.copy()]
        cv2.imshow("Edges", edge.astype(np.float32))
        cv2.imshow("Distance", dt/dt.max())
        markers[markers>0] = 1
        #cv2.imshow("Markers", markers.astype(np.float32))
        #cv2.imshow("Peaks", local_max.astype(np.float32))
        cv2.imshow("Watershed", color.label2rgb(labels, image=gray, kind='avg'))
        if region_means.size > 0 and np.count_nonzero(region_means) >= clusters-1:
            cv2.imshow("Kmeans", color.label2rgb(classified_labels, image=gray))

            """plt.clf()
            cols = 3
            rows = int(len(regions)/cols)+1
            
            for i in range(len(regions)):
                coords = regions[i].coords
                print(len(coords[:,0]))
                img = np.zeros(gray.shape, dtype=gray.dtype)
                img[coords[:,0], coords[:,1]] = gray[coords[:,0], coords[:,1]]
                plt.subplot(rows, cols, i+1)
                plt.imshow(img)
            #plt.hist(region_means, bins=256)
            plt.pause(0.001)"""



        #edge = cv2.Canny(gray, 100, 200)
        """edge = sobel(gray)
        edge_b = sobel(frame[:,:,0])
        edge_g = sobel(frame[:,:,1])
        edge_r = sobel(frame[:,:,2])

        frame_float = img_as_float(gray)
        #segments_slic = slic(frame_float, n_segments=100, compactness=10, sigma=1)
        #segments_felz = felzenszwalb(frame_float, scale=100, sigma=0.5, min_size=50)
        #segments_watershed = watershed(edge, markers=250, compactness=0.001)
        otsu_thresh = threshold_otsu(gray)
        otsu_thresh_b = threshold_otsu(frame[:,:,0])
        otsu_thresh_g = threshold_otsu(frame[:, :, 1])
        otsu_thresh_r = threshold_otsu(frame[:, :, 2])

        otsu_mask = gray > otsu_thresh
        otsu_mask_b = gray > otsu_thresh_b
        otsu_mask_g = gray > otsu_thresh_g
        otsu_mask_r = gray > otsu_thresh_r
        #otsu_mask = segmentation.clear_border(otsu_mask)
        frame_otsu = otsu_mask.astype(np.uint8)
        frame_otsu_b = otsu_mask_b.astype(np.uint8)
        frame_otsu_g = otsu_mask_g.astype(np.uint8)
        frame_otsu_r = otsu_mask_r.astype(np.uint8)
        frame_otsu[frame_otsu > 0] = 255
        frame_otsu_b[frame_otsu_b > 0] = 255
        frame_otsu_g[frame_otsu_g > 0] = 255
        frame_otsu_r[frame_otsu_r > 0] = 255




        # Display the resulting frame
        #cv2.imshow('frame', frame)
        #cv2.imshow('Gray', gray)
        #cv2.imshow("Edge", edge)
        #cv2.imshow("Otsu", frame_otsu)

        #cv2.imshow("Blue O C", edge_b)
        #cv2.imshow("Green O C", edge_g)
        #cv2.imshow("Red O C", edge_r)
        #cv2.imshow("Blue C", frame[:,:,0])
        #cv2.imshow("Green C", frame[:, :, 1])
        #cv2.imshow("Red C", frame[:, :, 2])"""

        """ax[0][0].clear()
        ax[0][0].grid()
        ax[0][0].hist(gray.ravel(), 256, [0, 256])
        ax[0][0].set_title("Gray")
        plt.pause(0.001)

        ax[0][1].clear()
        ax[0][1].grid()
        ax[0][1].hist(frame[:,:,0].ravel(), 256, [0, 256])
        ax[0][1].set_title("Blue")

        ax[1][0].clear()
        ax[1][0].grid()
        ax[1][0].hist(frame[:,:,1].ravel(), 256, [0, 256])
        ax[1][0].set_title("Green")

        ax[1][1].clear()
        ax[1][1].grid()
        ax[1][1].hist(frame[:,:,2].ravel(), 256, [0, 256])
        ax[1][1].set_title("Red")



        plt.pause(0.001)"""


        #cv2.imshow("SLIC", mark_boundaries(frame_float, segments_slic))
        #cv2.imshow("Felzenszwalb", mark_boundaries(frame_float, segments_felz))
        #cv2.imshow("Watershed", mark_boundaries(frame_float, segments_watershed))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    plt.show()
    cap.release()
    cv2.destroyAllWindows()
#perform_image_segmentation()