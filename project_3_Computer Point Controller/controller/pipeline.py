from controller.models import (
    FaceDetection,
    FacialLandmarksDetection,
    GazeEstimationModel,
    HeadPoseEstimationModel
)


class InferencePipeline(object):
    def __init__(self, folder_name, device='CPU', extensions=None):
        self.face_detection = FaceDetection(folder_name, device=device, extensions=extensions)
        self.head_pose_estimation = HeadPoseEstimationModel(folder_name, device=device, extensions=extensions)
        self.gaze_estimation = GazeEstimationModel(folder_name, device=device, extensions=extensions)
        self.facial_landmark = FacialLandmarksDetection(folder_name, device=device, extensions=extensions)

    def predict(self, input):
        cropped_face = self.face_detection(input)
        left_eye, right_eye = self.facial_landmark(cropped_face)
        head_pose_angles = self.head_pose_estimation(cropped_face)
        direction = self.gaze_estimation(left_eye, right_eye, head_pose_angles)
        return direction
    
    def __call__(self, input):
        return self.predict(input)
