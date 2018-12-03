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