#! /usr/bin/env python

"""
Lane Detection
    # Train a new model using kitti
    Usage: python3 train.py  --conf=./config.json \
    --test_input_folder=./data/data_road/testing/image_2 \
    --test_output_folder=./data/data_road/testing/segment_result
"""
import cv2
import numpy as np
import argparse
import json
import os
import glob

from src.DataHandler import DataSanity
from src.frontend import Segment

# define command line arguments
argparser = argparse.ArgumentParser(
    description='Train and validate Kitti Road Segmentation Model'
)

argparser.add_argument(
    '-c',
    '--conf', default="config.json",
    help='path to configuration file'
)

argparser.add_argument(
    '-tif',
    '--test_input_folder',
    help='path to test input image folder')

argparser.add_argument(
    '-tof',
    '--test_output_folder',
    help='path to test output image folder')

def mask_with_color(img, mask, color=(255,255,255)):
    color_mask = np.zeros(img.shape, img.dtype)
    color_mask[:,:] = color
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    return cv2.addWeighted(color_mask, 1, img, 1, 0)

def _main_(args):
    """

    :param args: command line arguments
    """

    # parse command line arguments
    config_path = args.conf

    # open json file and load the configurations
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # parse the json to retrieve the training configurations
    backend = config["model"]["backend"]
    input_size = (config["model"]["im_width"], config["model"]["im_height"])
    classes = config["model"]["classes"]
    data_dir = config["train"]["data_directory"] + '/'

    # Trigger the the dataset downloader if the dataset is not present
    DataSanity(data_dir).dispatch()

    # define the model and train
    segment = Segment(backend, input_size, classes)
    # segment.train(config["train"])

    #----------------------------Testing----------------------------------#
    model = segment.feature_extractor

    # load best model
    # model.load_weights(config["train"]["save_model_name"])
    model.load_weights(config["train"]["save_model_name"])

    inps = None
    inps = glob.glob(os.path.join(args.test_input_folder, "*.jpg")) + \
           glob.glob(os.path.join(args.test_input_folder, "*.png")) + \
           glob.glob(os.path.join(args.test_input_folder, "*.jpeg"))

    assert type(inps) is list

    count = 0

    for inp in inps:
        raw = cv2.imread(inp)
        raw = cv2.resize(raw, (input_size[1], input_size[0]))

        # Sub mean
        img = raw.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]

        net_input = np.expand_dims(img, axis=0)
        preds = model.predict(net_input, verbose=1)
        pred_1 = preds[:, :, :, 1].reshape((input_size[1], input_size[0]))
        # pred_2 = preds[:, :, :, 2].reshape((input_size[1], input_size[0]))
        # pred_3 = preds[:, :, :, 3].reshape((input_size[1], input_size[0]))

        # Create uint8 masks
        road_mask = np.zeros((input_size[1], input_size[0]), np.uint8)
        # car_mask = np.zeros((input_size[1], input_size[0]), np.uint8)
        # pedestrian_mask = np.zeros((input_size[1], input_size[0]), np.uint8)
        road_mask[pred_1 > 0.5] = 255
        # car_mask[pred_2 > 0.5] = 255
        # pedestrian_mask[pred_3 > 0.5] = 255

        # Bind mask with img
        out_img = raw.copy()
        out_img = cv2.resize(out_img, (input_size[0], input_size[1]))
        out_img = mask_with_color(out_img, road_mask, color=(0, 255, 0))
        # out_img = mask_with_color(out_img, car_mask, color=(0, 255, 0))
        # out_img = mask_with_color(out_img, pedestrian_mask, color=(255, 0, 0))

        # Write output
        if args.test_output_folder is not None:
            cv2.imwrite(os.path.join(args.test_output_folder, str(count) + ".png"), out_img)

        count += 1
        cv2.imshow("out_img", out_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
