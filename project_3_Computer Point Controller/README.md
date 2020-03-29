# Computer Pointer Controller

This project includes code to control your computer pointer by estimate where you are gazing at.

## Project Set Up and Installation
Project structured as follow:
```
.
├── README.md
├── bin
│   ├── demo.mp4
│   ├── libcpu_extension_sse4.so
│   └── model_flow.png
├── controller
│   ├── __init__.py
│   ├── input_feeder.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── face_detection.py
│   │   ├── facial_landmarks_detection.py
│   │   ├── gaze_estimation.py
│   │   └── head_pose_estimation.py
│   ├── mouse_controller.py
│   └── pipeline.py
├── index.py
└── pre_trained_2019
    ├── face-detection-adas-binary-0001.bin
    ├── face-detection-adas-binary-0001.xml
    ├── gaze-estimation-adas-0002.bin
    ├── gaze-estimation-adas-0002.xml
    ├── head-pose-estimation-adas-0001.bin
    ├── head-pose-estimation-adas-0001.xml
    ├── landmarks-regression-retail-0009.bin
    └── landmarks-regression-retail-0009.xml
```

## Demo
For a quick demo:
```python
$ python index.py
```


## Documentation
If you need a more detailed control, please run:
```python
$ python index.py --help
```
to see more options.

Currently, all model used are from [Openvino 2019 R4](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/). By defult, FP16 is used. You may also switch around using ```--model_folder``` flag To be specific, current model used:

- Face Detection [FP32-INT1](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/face-detection-adas-binary-0001/FP32-INT1/)
- Face Landmark Regression [FP16](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/landmarks-regression-retail-0009/FP16/)
- Head Pose Estimation [FP16](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/head-pose-estimation-adas-0001/FP16/)
- Gase Estimation [FP16](https://download.01.org/opencv/2019/open_model_zoo/R4/20200117_150000_models_bin/gaze-estimation-adas-0002/FP16/)

## Benchmarks

| Precision        | Loading Time           | Accuracy  |
| ------------- |:-------------:| -----:|
| FP32      | SLOW | High |
| FP16     | MEDIUM      |   MEDIUM |
| INT8 | Fast     |   Low |

## Results
FP32 is time consuming but with the best accuracy.
FP16 reduces the time cost along with a accuracy cost.
INT8 is the most efficient precision but sacrifices accuracy. 
