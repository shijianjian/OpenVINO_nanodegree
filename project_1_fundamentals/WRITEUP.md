# Project Write-Up


## Model Conversion
According to [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html), we utilized a model from [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/4563c282d3d664853eae3e99b6fd3453aacc39b0/research/object_detection/g3doc/detection_model_zoo.md), ssd_mobilenet_v2_coco_2018_03_29. The model contains:
  - checkpoint
  - frozen_inference_graph.pb
  - model.ckpt.data-00000-of-00001
  - model.ckpt.index
  - model.ckpt.meta
  - model.tflite
  - pipeline.config

Then using:
For OpenVINO 2020R1:
```bash
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
export MOD_PTH=./models/ssd_mobilenet_v2_coco_2018_03_29
$MOD_OPT/mo_tf.py \
  --input_model=$MOD_PTH/frozen_inference_graph.pb \
  --transformations_config $MOD_OPT/extensions/front/tf/ssd_v2_support.json \
  --tensorflow_object_detection_api_pipeline_config $MOD_PTH/pipeline.config
```
For OpenVINO 2019R3:
```bash
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
export MOD_PTH=./models/ssd_mobilenet_v2_coco_2018_03_29
$MOD_OPT/mo.py \
  --input_model=$MOD_PTH/frozen_inference_graph.pb \
  --tensorflow_object_detection_api_pipeline_config $MOD_PTH/pipeline.config \
  --tensorflow_use_custom_operations_config $MOD_OPT/extensions/front/tf/ssd_v2_support.json
```

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

For single_image (10 loops):
| Methods              | Timing           |
| -------------------- |:----------------:|
| Tensorflow Session   | 5.24s ± 74.4ms   |
| OpenVino             | 23.2ms ± 1.68ms  |

In single image test, I used the image under ```./images/people-counter-image.png```. The results are identical between those two frameworks.

For Video (1394 frame, 1 loop only):
| Methods              | Timing        | Frames Misclassifed (threshold = 0.5) | Frames Misclassifed (threshold = 0.4) | Frames Misclassifed (threshold = 0.6) |
| -------------------- |:-------------:| ---------:| ---------:| ---------:|
| Tensorflow Session   | 2h 1min 33s   | 14 | 12 | 30 |
| OpenVino             |    2min 52s   | 87 | 104 | 103 |

If using batchsize one for tensorflow session, it can be quite slow compare to OpenVino. Meanwhile, converted model has a lower performance compare to the original one with higher misclassified frames which failed on detection in some cases. In most cases, it failed when people showed their back to the camera than facing it.

Note:
Since there is no ground truth label for the video. For simplicity, I only calculated those frames which is correctly predicted by one framework while failed by another.

## Assess Model Use Cases

- Supper Market:
  If too many people wait in a queue for purchasing, more counters may open to buisnesses. It helps human resource management.

- Traffic:
  Can be used for smart city. If too many people waiting for crossing while no cars come through, green light can be switched.

## Assess Effects on End User Needs

Like aforementioned, the deployed model can have more misclassfied cases when the people showed their back onto the camera. This is likely to have some security issues if the model is going to be applyed to a high-accuracy needed field.

## Error:
1. If got:
    ```bash
      IPADDRESS = socket.gethostbyname(HOSTNAME)
    socket.gaierror: [Errno 8] nodename nor servname provided, or not known
    ```
    Then run:
    ```bash
    $ echo 127.0.0.1 $HOST >> /etc/hosts
    ```
2. FFMPEG cannot be used in the latest version. Remember to uninstall ffmpeg using ```brew uninstall ffmpeg``` first. New installation will not override. Previous version can be downloaded from [here](https://ffmpeg.org/download.html#release_3.4). It was 2.8.15 used on Udacity workspaces.
3. [ssd_mobilenet_v3_large_coco_2020_01_14](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz) is still buggy to go for.