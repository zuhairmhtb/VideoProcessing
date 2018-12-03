'Video Processing Software' is developed using Python for real time Image processing & Image Annotation with webcam stream.
It performs the following tasks:

1. Video Adjustment(Contrast, Sharpness, Brightness, etc.)
2. Edge Detection(Canny, Laplacain, Sobel)
3. Contour, Boundary and Object Detection using edge contours
4. Thresholding using Histogram
5. Noise Estimation using Convolution Neural Network
6. Image Annotation For Image Segmentation.
7. Annotated Image Storage in a data hierarchical Structure(More on https://github.com/zuhairmhtb/AudioClassification)
8. Image Segmentation(In progress)
9. Object Detection using Image Segmentation Algorithms(In progress)

The major modules used are as follows:

1. PyQt5
2. Keras
3. Scikit
4. PIL
5. OpenCV

The main architecture of the software is divided into 3 main parts:

1. Main Thread: The thread that handles all other threads and GUI updates.
2. Controllers: Objects(WebcamPlayer, VideoProcessor, ImageAnnotator) that run on separate threads in order to reduce
load on main thread.
3. Views: The layout file for each Controller in order to receive user input for parameter adjustment and display
outputs produced by the controller.

The complete software is divided into 5 main files. They are:

1. __main__.py: This file runs the main thread of PyQt5 GUI and manages all other threads in order to maintain
   consistency of speed when receiving and displaying webcam feed. It receives data from Controllers and updates layout
   files of the corresponding controllers.
2. controllers.py: The file containing classes for all the Controllers(NeuralNetwork, VideoProcessor, Image Annotator
and WebcamPlayer).
3. interfaces.py: The file containing interface for the main thread in order to interact with controller objects.
4. layouts.py: Layouts of different controllers and sub-modules(Contour & Object Detection, Thresholding, Noise Estimation)
in order to display in GUI.
5. widgets.py: Reusable PyQt5 customized widgets in order to use in layouts.

Image Features Calculation:

1. RGB COLOR IMAGE(Stored in frame)
2 ORIENTATION MAP(Histogram of Oriented Gradients)
3 Edge Detection(Stored in the frame)
4 DISTANCE MAP(From the Edge image-Euclidean Distance from the edges)
5 CLUSTERS OF SEGMENTED IMAGE(Watershed Algorithm with mean intensity of distance map as markers and
      K-Means Clustering)
6 FEATURE DESCRIPTORS(Oriented Fast and Rotated Brief Algorithm)
7 REGION PROPERTIES(From Clusters of segmented image)

(For Full Documentation, see controllers.py-->ImageAnnotationController)

Data Hierarchical Structure:

    1. Each annotated image saved in a folder with unique name(time_userid) has the following files:

       a. A set of Feature files(Numpy arrays) - See implemented features for all feature file names

       b. An annotation file(JSON or XML file according to PASCAL VOC FORMAT) where each segment contains a id
          (starting from 0 as the background)

        e.g.
        ```xml
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
                <name>21</name> <!-- ID of the segmented object -->
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
        ```
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
