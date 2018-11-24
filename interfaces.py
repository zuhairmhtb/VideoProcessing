import abc
class MainInterface(abc.ABC):
    """
    Frame Object: list of 3D numpy array of raw RGB frame captured from webcam and adjusted using (H,S,V,Contrast) params,
              a 2D numpy array of the corresponding grayscale image, a 2D numpy array of Edge Image detected using
              available edge detection algorithms(Canny, Sobel, Laplacian), a list of 2D contours detected from the
              edge map and a list of hierarchy information obtained from the contour detection algorithm
    """
    @abc.abstractmethod
    def store_current_frame(self, frame):
        """
        This function stores a frame object provided by the Webcam Player
        :param frame: the frame object
        :return: null
        """
        pass
    #@abc.abstractmethod
    #def update_last_frame_queue(self, frame):
    #    """
    #    This function stores a frame object to the list containing a set of last frames provided by the WebcamPlayer
    #    :param frame: The frame object
    #    :return: null
    #    """
    #    pass
    @abc.abstractmethod
    def get_current_frame(self):
        """
        This function returns a copy of the current frame object stored by WebcamPlayer
        :return: the frame object
        """
        pass
    @abc.abstractmethod
    def get_last_frame_queue(self, nmax=1):
        """
        This function returns 'n' number of last frames stored by the webcam player
        :param nmax: The number of last frames to return(integer). If the number is greater than available, then all
        frames are returned
        :return: A list of frame objects
        """
        pass
    @abc.abstractmethod
    def create_neural_network_model_params(self, inp_img_w, inp_img_h, out_img_w, out_img_h, dense_units,
                                           conv_ksize, conv_kstride, pool_ksize, pool_kstride, dropout,
                                           activation, loss, optimizer):
        """
        This function generates the parameters required to create a neural network model
        :param inp_img_w: Width of the input image
        :param inp_img_h: Height of the input image
        :param out_img_w: Width of the output image
        :param out_img_h: Height of the output image
        :param dense_units: Number of units in dense layer
        :param conv_ksize: Kernel Size of Convolution layer
        :param conv_kstride: Kernel Stride of Convolution layer
        :param pool_ksize: Kernel Size of Pool layer
        :param pool_kstride: Kernel Stride of Pool layer
        :param dropout: Dropout rate of convolution layer
        :param activation: Activation function
        :param loss: Loss calculation method
        :param optimizer: Optimizer method
        :return: The generated network model params
        """
        pass
    @abc.abstractmethod
    def request_neural_network(self, model_name, model_params):
        """
        This function requests the main thread to create a neural network with the specified model parameters and return
        a reference to the created neural network object.
        :param model_name: A string containing name for the neural network
        :param model_params: A String which contains the suffix of the filename of a saved neural network model and its
        corresponding weight file or A list of arguments containing the specification for the network
        :return: A reference of the created neural network interface object
        """
        pass
class NeuralNetworkInterface(abc.ABC):
    @abc.abstractmethod
    def train(self, X, y):
        """
        This function trains a Neural Network object with the specified training data and label. The performance
        metric of the network is returned and the view(Performance Graph and Model Weight View) updated.
        :param X: The Input Image data on which the model has to be trained
        :param y: The input image's label data
        :return: The performance metric of training
        """
        pass

    @abc.abstractmethod
    def predict(self, X, y=None):
        """
        This function uses the network to predict an output for the specified input image. View(Data View and Performance
        Graph) of the network interface is updated whenever the function is called if an expected label is provided.
        Performace metric is returned for prediction.
        :param X: The input image data for which output needs to be predicted
        :param y: An optional label data in order to compare expected and predicted output to update view
        :return: The performance metric and predicted output image data for the specified input data
        """
        pass
class VideoProcessorInterface(abc.ABC):
    pass
class WebcamPlayerInterface(abc.ABC):
    pass