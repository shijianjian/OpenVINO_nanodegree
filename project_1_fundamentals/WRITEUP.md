# Project Write-Up


## Model Conversion
According to [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html), we utilized a model from [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/4563c282d3d664853eae3e99b6fd3453aacc39b0/research/object_detection/g3doc/detection_model_zoo.md), [ssd_mobilenet_v3_large_coco_2020_01_14](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz). The model contains:
  - checkpoint
  - frozen_inference_graph.pb
  - model.ckpt.data-00000-of-00001
  - model.ckpt.index
  - model.ckpt.meta
  - model.tflite
  - pipeline.config

Then using:
```bash
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
export MOD_PTH=./models/ssd_mobilenet_v3_large_coco_2020_01_14
$MOD_OPT/mo_tf.py \
  --input_model=$MOD_PTH/frozen_inference_graph.pb \
  --transformations_config $MOD_OPT/extensions/front/tf/ssd_support_api_v1.14.json \
  --tensorflow_object_detection_api_pipeline_config $MOD_PTH/pipeline.config
```

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...


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
