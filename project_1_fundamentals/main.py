"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import datetime
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from helpers import preprocessing, ssd_boxes_counting, is_new, draw_boxes

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### Handle the input stream ###
    if os.path.splitext(args.input)[1] in ['jpg', 'png']:
        mode = 'single_image_mode'
    else:
        mode = 'video_mode'
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))
    prev_result  = None
    total = 0
    counts = []
    emit = True
    frame_id = -1
    epoch = 0
    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        start_time = datetime.datetime.now()
        flag, frame = cap.read()
        if not flag:
            break
        frame_id += 1
        if frame_id >= 200:
            epoch += 1
            frame_id = 0
        ### Pre-process the image as needed ###
        _frame = preprocessing(frame, (net_input_shape[3], net_input_shape[2]))

        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(_frame, frame_id)

        ### Wait for the result ###
        if infer_network.wait(frame_id) == 0:
            ### Get the results of the inference request ###
            result = infer_network.get_output(frame_id)
            ### Extract any desired stats from the results ###
            count = ssd_boxes_counting(result, prob_threshold, target_class_index=1)
            frame = draw_boxes(frame, result, prob_threshold, width, height, target_class_index=1)
            if len(counts) > 30 and count > 0:
                if count in counts[-30:]:
                    total += 0
                else:
                    if prev_result is not None:
                        if is_new(result, prev_result, prob_threshold):
                            total += 1
                            if emit:
                                start_frame = epoch * 200 + frame_id
                                prev_result = result
                                emit = False
                    else:
                        total += 1
                        if emit:
                            start_frame = epoch * 200 + frame_id
                            prev_result = result
                            emit = False
            elif len(counts) > 30 and count == 0:
                if total > 0 and 1 == counts[-31] and 1 not in counts[-30:-1]:
                    # Only emit once for each count
                    if not emit:
                        time_diff = (epoch * 200 + frame_id - start_frame) / 10
                        client.publish('person/duration', json.dumps({"duration": time_diff}))
                        emit = True
            counts.append(count)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person', json.dumps({"count": count, "total": total}))
        ### Send the frame to the FFMPEG server ###
        out  = np.uint8(frame)
        sys.stdout.buffer.write(out)  
        sys.stdout.flush()
        ###  Write an output image if `single_image_mode` ###
        if mode == 'single_image_mode':
            img = draw_boxes(frame, result, prob_threshold, width, height, target_class_index=1)
            cv2.imwrite('./output.jpg', img)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
