import os
import argparse

from controller.input_feeder import InputFeeder
from controller.pipeline import InferencePipeline
from controller.mouse_controller import MouseController


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computer Point Controller by gaze estimation')
    parser.add_argument('--precision', type=str, default='medium', help='high, medium or low (default: medium)')
    parser.add_argument('--speed', type=str, default='fast', help='fast, medium or slow (default: fast)')
    parser.add_argument('--device', type=str, default='CPU', help='CPU only for now.')
    parser.add_argument('--model_folder', type=str, default='pre_trained_2019_FP16', help='Folder contains models')
    parser.add_argument('--DEMO', action='store_true', help='Using demo.mp4 as input.')
    args = parser.parse_args()

    if args.device == 'CPU':
        cpu_extension = "/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
        # cpu_extension = os.path.abspath("./bin/libcpu_extension_sse4.so")
    else:
        cpu_extension = None
    if args.DEMO:
        input_type = 'video'
    else:
        input_type = 'cam'

    feed = InputFeeder(input_type=input_type, input_file='./bin/demo.mp4')
    pipeline = InferencePipeline(folder_name=args.model_folder, device=args.device, extensions=cpu_extension)
    mouse_controller = MouseController(args.precision, args.speed)
    feed.load_data()
    for batch in feed.next_batch():
        if batch is None:
            continue
        x, y, z = pipeline(batch)
        mouse_controller.move(x, y)
    feed.close()