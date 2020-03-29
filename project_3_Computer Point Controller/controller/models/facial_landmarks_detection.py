from controller.models.base import BaseModel


class FacialLandmarksDetection(BaseModel):
    '''
    This is a lightweight landmarks regressor for the Smart Classroom scenario.
    It has a classic convolutional design: stacked 3x3 convolutions, batch normalizations, PReLU activations,
    and poolings. Final regression is done by the global depthwise pooling head and FullyConnected layers.
    The model predicts five facial landmarks: two eyes, nose, and two lip corners.
    '''
    def __init__(self, folder_name, device='CPU', extensions=None):
        super(FacialLandmarksDetection, self).__init__(
            "landmarks-regression-retail-0009", folder_name, device=device, extensions=extensions)
        self.load_model()
        self.check_plugin(self.plugin)

    def __call__(self, image):
        inp = self.preprocess_input(image)
        out = self.predict(inp)
        out = self.preprocess_output(out, inp)
        return out

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        return self.exec_network.infer(inputs={self.input_blob[0]: image})

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        name: "data" , shape: [1x3x48x48] - An input image in [BxCxHxW] format. Expected color order is BGR
        '''
        return super().preprocess_input(image, (48, 48))

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values
        for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        All the coordinates are normalized to be in range [0,1].
        '''
        outputs =  outputs['95'].squeeze()
        right_eye = image[
            int(image.shape[0] * outputs[0]) - 25:int(image.shape[0] * outputs[0]) + 25,
            int(image.shape[1] * outputs[1]) - 25:int(image.shape[1] * outputs[1]) + 25
        ]
        left_eye = image[
            int(image.shape[0] * outputs[6]) - 25:int(image.shape[0] * outputs[6]) + 25,
            int(image.shape[1] * outputs[7]) - 25:int(image.shape[1] * outputs[7]) + 25
        ]
        return left_eye.squeeze().transpose(1, 2, 0), right_eye.squeeze().transpose(1, 2, 0)