import numpy as np
import cv2
import os
import json
import time
import random
import pandas as pd
from glob import glob
import shutil


import kfbReader

import cfg
from util import *


def save_roi_to_npz(path):
    pos_paths = glob(os.path.join(path, "pos_[0-8]/*.kfb"))

    for pos_path in pos_paths:
        filename = pos_path.split("/")[-1].split('.')[0]
        json_path = glob(os.path.join(path, "labels", filename + ".json"))[0]
        with open(json_path, 'r') as f:
            json_infos = json.loads(f.read())

        r = kfbReader.reader()
        r.ReadInfo(pos_path, 20, True)

        roi_coords = []
        for json_info in json_infos:
            if json_info['class'] == 'roi':
                coord = {'x': json_info['x'], 'y': json_info['y'], 'w': json_info['w'], 'h': json_info['h']}
                roi_coords.append(coord)

        roi_cnt = 1
        for roi_coord in roi_coords:
            X, Y, W, H = roi_coord['x'], roi_coord['y'], roi_coord['w'], roi_coord['h']
            img = r.ReadRoi(X, Y, W, H, 20).copy()
            label = np.zeros((0, 4), dtype="int")

            pos_cnt = 0
            for json_info in json_infos:
                if json_info['class'] == 'pos':
                    x, y, w, h = json_info['x'], json_info['y'], json_info['w'], json_info['h']
                    if X < x < X + W and Y < y < Y + H:
                        pos_cnt += 1
                        box = np.zeros((1, 4), dtype="int")
                        box[0, 0] = max(int(x - X), 0)
                        box[0, 1] = max(int(y - Y), 0)
                        box[0, 2] = min(int(x - X + w), W)
                        box[0, 3] = min(int(y - Y + h), H)
                        label = np.append(label, box, axis=0)
            if pos_cnt == 0:
                continue

            sample_path = cfg.sample_path
            mkdir(sample_path)
            save_path = os.path.join(sample_path, filename + "_" + str(roi_cnt) + ".npz")
            np.savez_compressed(save_path, img=img, label=label)

            roi_cnt += 1
        print("Finish: ", filename, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == "__main__":
    save_roi_to_npz(cfg.train_path)
