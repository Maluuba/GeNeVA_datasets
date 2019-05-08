# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to parse and read i-CLEVR data and save it in HDF5 format for Object Detector & Localizer
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


def create_h5():
    # load required keys
    data_path = keys['iclevr_data_source']
    output_path = keys['iclevr_hdf5_folder']
    OBJECTS = keys['iclevr_objects']
    with open(OBJECTS, 'r') as f:
        OBJECTS = f.readlines()
        OBJECTS = [tuple(x.strip().split()) for x in OBJECTS]
    background_path = keys['iclevr_background']

    # create hdf5 files for train, val, test
    train_h5 = h5py.File(os.path.join(output_path, 'clevr_obj_train.h5'), 'w')
    val_h5 = h5py.File(os.path.join(output_path, 'clevr_obj_val.h5'), 'w')
    test_h5 = h5py.File(os.path.join(output_path, 'clevr_obj_test.h5'), 'w')

    json_path = os.path.join(data_path, 'scenes')
    images_path = os.path.join(data_path, 'images')

    background_image = cv2.imread(background_path)

    entites = json.dumps(['{} {}'.format(e[0], e[1]) for e in OBJECTS])

    # start saving data into hdf5; loop over all scenes
    c_train = -1
    c_val = -1
    c_test = -1
    for scene in tqdm(glob(json_path + '/*.json')):
        filename = os.path.basename(scene)
        with open(scene, 'r') as f:
            scene = json.load(f)

        # identify if scene belongs to train / val / test
        split = filename.split('_')[1]
        scene_id = filename.split('_')[2][:-5]

        # add images
        images_files = sorted(glob(os.path.join(images_path, 'CLEVR_{}_{}_*'.format(split, scene_id))))
        images = []
        for t, image_file in enumerate(images_files):
            image = cv2.imread(image_file)
            image = cv2.resize(image, (128, 128))
            images.append(image)

        # add objects and object coordinates
        agg_object = np.zeros(24)
        objects = np.zeros((5, 24))
        agg_object_coords = np.zeros((24, 3))
        object_coords = np.zeros((5, 24, 3))
        for t, obj in enumerate(scene['objects']):
            color = obj['color']
            shape = obj['shape']
            index = OBJECTS.index((shape, color))
            agg_object[index] = 1
            objects[t] = agg_object
            agg_object_coords[index] = [obj['pixel_coords'][0]/320.*128, obj['pixel_coords'][1]/240.*128, obj['pixel_coords'][2]]
            object_coords[t] = agg_object_coords

        for t, obj in enumerate(scene['objects']):
            if split == 'train':
                c_train += 1
                sample = train_h5.create_group(str(c_train))
            elif split == 'val':
                c_val += 1
                sample = val_h5.create_group(str(c_val))
            else:
                c_test += 1
                sample = test_h5.create_group(str(c_test))

            sample.create_dataset('scene_id', data=scene_id)
            sample.create_dataset('image', data=np.array(images)[t])
            sample.create_dataset('objects', data=objects[t])
            sample.create_dataset('coords', data=np.array(object_coords)[t])


if __name__ == '__main__':
    create_h5()
