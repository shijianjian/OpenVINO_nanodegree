import numpy as np

from controller.models.base import BaseModel


class HeadPoseEstimationModel(BaseModel):
    '''
    Head pose estimation network based on simple, handmade CNN architecture.
    Angle regression layers are convolutions + ReLU + batch norm + fully connected with one output.
    '''
    def __init__(self, folder_name, device='CPU', extensions=None):
        super(HeadPoseEstimationModel, self).__init__(
            "head-pose-estimation-adas-0001", folder_name, device=device, extensions=extensions)
        self.load_model()
        self.check_plugin(self.plugin)

    def __call__(self, image):
        inp = self.preprocess_input(image)
        out = self.predict(inp)
        out = self.preprocess_output(out)
        return out

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        return self.exec_network.infer(inputs={self.input_blob[0]: image})

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        name: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR
        '''
        return super().preprocess_input(image, (60, 60))

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        
        1. name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        2. name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        3. name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        '''
        return np.concatenate(list(outputs.values()), axis=-1)