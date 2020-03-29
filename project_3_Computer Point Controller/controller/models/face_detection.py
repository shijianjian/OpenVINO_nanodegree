from controller.models.base import BaseModel


class FaceDetection(BaseModel):
    '''
    Face detector for driver monitoring and similar scenarios.
    The network features a pruned MobileNet backbone that includes depth-wise convolutions to reduce
    the amount of computation for the 3x3 convolution block. Also some 1x1 convolutions are binary
    that can be implemented using effective binary XNOR+POPCOUNT approach
    '''
    def __init__(self, folder_name, device='CPU', extensions=None):
        super(FaceDetection, self).__init__(
            "face-detection-adas-binary-0001", folder_name, device=device, extensions=extensions)
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
        name: "input" , shape: [1x3x384x672] - An input image in the format [BxCxHxW]
        '''
        return super().preprocess_input(image, (672, 384))

    def preprocess_output(self, outputs, input=None):
        '''
        Before feeding the output of this model to the next model,
        The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        '''
        x_out = outputs['detection_out']
        shape = input.squeeze().transpose(1,  2,  0).shape
        idx = ((x_out[0, 0, :, 1]  == 1) & (x_out[0, 0, :, 2]  > 0.6)).astype(int).argmax()
        x_crop = input.squeeze().transpose(1,  2,  0)[
            int(shape[0] * x_out[0, 0][idx][4]):int(shape[0] * x_out[0, 0][idx][6]),
            int(shape[1] * x_out[0, 0][idx][3]):int(shape[1] * x_out[0, 0][idx][5]),
        ]
        return x_crop