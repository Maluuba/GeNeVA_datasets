# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to parse and read CoDraw data and save it in HDF5 format for Object Detector & Localizer
"""
from glob import glob
import json
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def create_object_detection_dataset():
    # load required keys
    scenes_path = keys['codraw_scenes']
    images_path = keys['codraw_images']
    background_img = cv2.imread(keys['codraw_background'])
    h5_path = keys['codraw_hdf5_folder']
    codraw_extracted_coords = keys['codraw_extracted_coordinates']

    # set height, width, scaling parameters
    h, w, _ = background_img.shape
    scale_x = 128. / w
    scale_y = 128. / h
    scaling_ratio = np.array([scale_x, scale_y, 1])

    # create hdf5 files for train, val, test
    h5_train = h5py.File(os.path.join(h5_path, 'codraw_obj_train.h5'), 'w')
    h5_val = h5py.File(os.path.join(h5_path, 'codraw_obj_val.h5'), 'w')
    h5_test = h5py.File(os.path.join(h5_path, 'codraw_obj_test.h5'), 'w')

    # set objects and bow (bag of words) dicts for each image
    bow_dim = 0
    GT_BOW = {}
    GT_OBJECTS = {}
    with open(codraw_extracted_coords, 'r') as f:
        for line in f:
            splits = line.split('\t')
            image = splits[0]
            split_coords = lambda x: [int(c) for c in x.split(',')]
            bow = np.array([split_coords(b) for b in splits[1].split()])
            bow_dim = len(bow)
            GT_BOW[image] = bow[:, 0]
            scaling = scaling_ratio * np.expand_dims(bow[:, 0], axis=1).repeat(3, 1)
            GT_OBJECTS[image] = (bow[:, 1:] * scaling).astype(int)

    # start saving data into hdf5; loop over all scenes
    c_train = -1
    c_val = -1
    c_test = -1
    for scene_file in tqdm(sorted(glob('{}/*json'.format(scenes_path)))):
        # identify if scene belongs to train / val / test
        split = scene_file.split('/')[-1].split('_')[0]

        with open(scene_file, 'r') as f:
            scene = json.load(f)
        scene_id = scene['image_id']

        # loop over turns in a single scene
        idx = 0
        for i in range(len(scene['dialog'])):
            turn = scene['dialog'][i]

            bow = GT_BOW['Scene{}_{}'.format(scene_id, idx)]
            coords = GT_OBJECTS['Scene{}_{}'.format(scene_id, idx)]

            # if there is no image for current turn: merge with next turn
            if turn['abs_d'] == '':
                continue

            image = cv2.imread(os.path.join(images_path, 'Scene{}_{}.png'.format(scene_id, idx)))
            image = cv2.resize(image, (128, 128))
            idx += 1

            if split == 'train':
                c_train += 1
                ex = h5_train.create_group(str(c_train))
            elif split == 'val':
                c_val += 1
                ex = h5_val.create_group(str(c_val))
            elif split == 'test':
                c_test += 1
                ex = h5_test.create_group(str(c_test))

            ex.create_dataset('image', data=image)
            ex.create_dataset('objects', data=np.array(bow))
            ex.create_dataset('coords', data=np.array(coords))
            ex.create_dataset('scene_id', data=scene_id)


if __name__ == '__main__':
    create_object_detection_dataset()
