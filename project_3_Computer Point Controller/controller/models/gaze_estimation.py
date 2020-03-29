from controller.models.base import BaseModel


class GazeEstimationModel(BaseModel):
    '''
    The network takes three inputs: square crop of left eye image, square crop of right eye image,
    and three head pose angles – (yaw, pitch, and roll) (see figure).
    The network outputs 3-D vector corresponding to the direction of a person’s gaze in a Cartesian coordinate system
    in which z-axis is directed from person’s eyes (mid-point between left and right eyes’ centers) 
    to the camera center, y-axis is vertical, and x-axis is orthogonal to both z,y axes so that (x,y,z)
    constitute a right-handed coordinate system.
    '''
    def __init__(self, folder_name, device='CPU', extensions=None):
        super(GazeEstimationModel, self).__init__(
            "gaze-estimation-adas-0002", folder_name, device=device, extensions=extensions)
        self.load_model()
        self.check_plugin(self.plugin)

    def __call__(self, left_eye_image, right_eye_image, head_pose_angles):
        inp = self.preprocess_input(left_eye_image, right_eye_image, head_pose_angles)
        out = self.predict(*inp)
        out = self.preprocess_output(out)
        return out

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        This method is meant for running predictions on the input image.
        '''
        return self.exec_network.infer(inputs={
            self.input_blob[0]: head_pose_angles,
            self.input_blob[1]: left_eye_image,
            self.input_blob[2]: right_eye_image
        })

    def preprocess_input(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        Before feeding the data into the model for inference,

        Blob in the format [BxCxHxW],
            with the name left_eye_image and the shape [1x3x60x60].

        Blob in the format [BxCxHxW],
            with the name right_eye_image and the shape [1x3x60x60].

        Blob in the format [BxC],
            with the name head_pose_angles and the shape [1x3].
        '''
        return (
            super().preprocess_input(left_eye_image, (60, 60)),
            super().preprocess_input(right_eye_image, (60, 60)),
            head_pose_angles
        )

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector.
        Please note that the output vector is not normalizes and has non-unit length.
        '''
        return tuple(outputs['gaze_vector'].squeeze())
        