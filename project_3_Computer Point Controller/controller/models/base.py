import os
import sys
import cv2

from openvino.inference_engine import IENetwork, IECore


class BaseModel:
    '''
    Class for a Model.
    '''
    def __init__(self, model_name, folder_name, device='CPU', extensions=None):
        model_xml = os.path.join(os.getcwd(), folder_name, model_name + '.xml')
        model_bin = os.path.join(os.getcwd(), folder_name, model_name + '.bin')

        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.device = device
        self.extensions = extensions

    def load_model(self, num_requests=1):
        '''
        This method is for loading the model to the device specified by the user.
        This is where to load any Plugins,.
        '''
        if self.device == 'CPU' and self.extensions is not None:
            self.plugin.add_extension(self.extensions, self.device)

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, self.device, num_requests=num_requests)

        # Get the input layer
        self.input_blob = list(self.network.inputs.keys())
        self.output_blob = list(self.network.outputs.keys())

    def check_plugin(self, plugin):
        '''
        This method checks whether the model(along with the plugin) is supported
        on the CPU device or not. If not, then this raises and Exception
        '''
        pass

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def preprocess_input(self, image, shape):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image = cv2.resize(image, shape)
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
