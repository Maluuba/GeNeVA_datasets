# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to extract object names and coordinate information from the CoDraw data
"""
from glob import glob
import json

import numpy as np
from tqdm import tqdm
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def extract_object_names():
    # save names of all the objects in the dataset
    with open(keys['codraw_objects_source'], 'r') as f:
        names = f.readlines()
        names = [name.strip().split('\t')[0].lower() for name in names]
    with open(keys['codraw_objects'], 'w') as f:
        for name in names:
            f.write(name+'\n')


def extract_objects():
    json_path = keys['codraw_scenes']
    extracted_objects_file = open(keys['codraw_extracted_coordinates'], 'w')

    # load object names
    with open(keys['codraw_objects'], 'r') as f:
        OBJECTS = [l.split()[0] for l in f]
    # mapping from filenames of individual object images to the object names
    PNG_MAPPING = {}
    with open(keys['codraw_png_to_object'], 'r') as f:
        for l in f:
            splits = l.split('\t')
            PNG_MAPPING[splits[0]] = splits[1].strip()

    # loop through scene jsons and extract object information
    for scene_json in tqdm(sorted(glob('{}/*.json'.format(json_path)))):
        with open(scene_json, 'r') as f:
            scene = json.load(f)
            scene_id = scene['image_id']

            turn_id = -1
            for dialog in scene['dialog']:
                if len(dialog['abs_d']) < 2:
                    continue

                turn_id += 1
                bow = np.zeros((len(OBJECTS)), dtype=int)
                x_coords = np.ones((len(OBJECTS)), dtype=int) * -1
                y_coords = np.ones((len(OBJECTS)), dtype=int) * -1
                z_coords = np.ones((len(OBJECTS)), dtype=int) * -1

                dialog = dialog['abs_d'].split(',')

                idx = 1
                for _ in range(int(dialog[0])):
                    png_filename = dialog[idx][:-4]
                    x = float(dialog[idx + 4])
                    y = float(dialog[idx + 5])
                    z = float(dialog[idx + 6])

                    if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
                        idx = idx + 8
                        continue

                    index = OBJECTS.index(PNG_MAPPING[png_filename])
                    bow[index] = 1
                    x_coords[index] = x
                    y_coords[index] = y
                    z_coords[index] = z

                    idx = idx + 8

                meta_data = []
                for e in range(len(OBJECTS)):
                    object_meta = str.join(',',
                                           [str(bow[e]),
                                            str(x_coords[e]),
                                            str(y_coords[e]),
                                            str(z_coords[e])])
                    meta_data.append(object_meta)

                extracted_objects_file.write('Scene{}_{}\t{}\n'.format(scene_id,
                                                                       turn_id,
                                                                       str.join(' ', meta_data)))


if __name__ == '__main__':
    extract_object_names()
    extract_objects()
