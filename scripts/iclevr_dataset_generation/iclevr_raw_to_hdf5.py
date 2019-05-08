# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to parse and read raw i-CLEVR data and save it in HDF5 format for GeNeVA-GAN
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
    train_h5 = h5py.File(os.path.join(output_path, 'clevr_train.h5'), 'w')
    val_h5 = h5py.File(os.path.join(output_path, 'clevr_val.h5'), 'w')
    test_h5 = h5py.File(os.path.join(output_path, 'clevr_test.h5'), 'w')

    json_path = os.path.join(data_path, 'scenes/')
    images_path = os.path.join(data_path, 'images/')
    text_path = os.path.join(data_path, 'text/')

    # add background image to hdf5
    background_image = cv2.imread(background_path)
    train_h5.create_dataset('background', data=background_image)
    val_h5.create_dataset('background', data=background_image)
    test_h5.create_dataset('background', data=background_image)

    # add object properties to hdf5
    entites = json.dumps(['{} {}'.format(e[0], e[1]) for e in OBJECTS])
    train_h5.create_dataset('entities', data=entites)
    val_h5.create_dataset('entities', data=entites)
    test_h5.create_dataset('entities', data=entites)

    # start saving data into hdf5; loop over all scenes
    for scene in tqdm(glob(json_path + '/*.json')):
        filename = os.path.basename(scene)
        with open(scene, 'r') as f:
            scene = json.load(f)

        # identify if scene belongs to train / val / test
        split = filename.split('_')[1]
        scene_id = filename.split('_')[2][:-5]

        # add text
        text_file = os.path.join(text_path, 'CLEVR_{}_{}.txt'.format(split, scene_id))
        with open(text_file, 'r') as f:
            text = [line.strip() for line in f]

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

        if split == 'train':
            sample = train_h5.create_group(scene_id)
        elif split == 'val':
            sample = val_h5.create_group(scene_id)
        else:
            sample = test_h5.create_group(scene_id)

        sample.create_dataset('scene_id', data=scene_id)
        sample.create_dataset('images', data=np.array(images))
        sample.create_dataset('text', data=json.dumps(text))
        sample.create_dataset('objects', data=objects)
        sample.create_dataset('coords', data=np.array(object_coords))


if __name__ == '__main__':
    create_h5()
