#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_xml, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        # Check unsupported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, num_requests=200)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image, request_id):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        ### Start asynchronous inference
        self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: image})

    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        ### Wait for the async request to be complete
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### Return the outputs of the network from the output_blob
        return self.exec_network.requests[request_id].outputs[self.output_blob]
