# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to parse and read raw CoDraw data and save it in HDF5 format for GeNeVA-GAN
"""
from glob import glob
import json
import os
import pickle
import string

import cv2
import h5py
import nltk
import numpy as np
from tqdm import tqdm
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def replace_at_offset(msg, tok, offset, tok_replace):
    before = msg[:offset]
    after = msg[offset:]
    after = after.replace(tok, tok_replace, 1)
    return before + after


def create_h5():
    # load required keys
    scenes_path = keys['codraw_scenes']
    images_path = keys['codraw_images']
    background_img = cv2.imread(keys['codraw_background'])
    h5_path = keys['codraw_hdf5_folder']
    spell_check = keys['codraw_spell_check']
    codraw_extracted_coords = keys['codraw_extracted_coordinates']

    # set height, width, scaling parameters
    h, w, _ = background_img.shape
    scale_x = 128. / w
    scale_y = 128. / h
    scaling_ratio = np.array([scale_x, scale_y, 1])
    background_img = cv2.resize(background_img, (128, 128))

    # load spelling corrections - obtained via Bing Spell Check API
    with open(spell_check, 'rb') as f:
        spell_check = pickle.load(f)

    # create hdf5 files for train, val, test
    h5_train = h5py.File(os.path.join(h5_path, 'codraw_train.h5'), 'w')
    h5_val = h5py.File(os.path.join(h5_path, 'codraw_val.h5'), 'w')
    h5_test = h5py.File(os.path.join(h5_path, 'codraw_test.h5'), 'w')
    h5_train.create_dataset('background', data=background_img)
    h5_val.create_dataset('background', data=background_img)
    h5_test.create_dataset('background', data=background_img)

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

    # mark purely chitchat turns to be removed
    chitchat = ['hi', 'done', 'ok', 'alright', 'okay', 'thanks', 'bye', 'hello']

    # start saving data into hdf5; loop over all scenes
    c_train = 0
    c_val = 0
    c_test = 0
    for scene_file in tqdm(sorted(glob('{}/*json'.format(scenes_path)))):
        # identify if scene belongs to train / val / test
        split = scene_file.split('/')[-1].split('_')[0]

        images = []
        utterences = []
        objects = []
        coordinates = []

        with open(scene_file, 'r') as f:
            scene = json.load(f)
        scene_id = scene['image_id']

        # loop over turns in a single scene
        idx = 0
        prev_bow = np.zeros((bow_dim))
        description = []
        for i in range(len(scene['dialog'])):
            bow = GT_BOW['Scene{}_{}'.format(scene_id, idx)]
            # new objects added in this turn
            hamming_distance = np.sum(bow - prev_bow)
            turn = scene['dialog'][i]
            # lowercase all messages
            teller = str.lower(turn['msg_t'])
            drawer = str.lower(turn['msg_d'])
            # clear chitchat turns
            if teller in chitchat:
                teller = ''
            if drawer in chitchat:
                drawer = ''

            # replace with spelling suggestions returned by Bing Spell Check API
            if teller in spell_check and len(spell_check[teller]['flaggedTokens']) != 0:
                for flagged_token in spell_check[teller]['flaggedTokens']:
                    tok = flagged_token['token']
                    tok_offset = flagged_token['offset']
                    assert len(flagged_token['suggestions']) == 1
                    tok_replace = flagged_token['suggestions'][0]['suggestion']
                    teller = replace_at_offset(teller, tok, tok_offset, tok_replace)
            if drawer in spell_check and len(spell_check[drawer]['flaggedTokens']) != 0:
                for flagged_token in spell_check[drawer]['flaggedTokens']:
                    tok = flagged_token['token']
                    tok_offset = flagged_token['offset']
                    assert len(flagged_token['suggestions']) == 1
                    tok_replace = flagged_token['suggestions'][0]['suggestion']
                    drawer = replace_at_offset(drawer, tok, tok_offset, tok_replace)

            # add delimiting tokens: <teller>, <drawer>
            if teller != '':
                description += ['<teller>'] + nltk.word_tokenize(teller)
            if drawer != '':
                description += ['<drawer>'] + nltk.word_tokenize(drawer)

            description = [w for w in description if w not in chitchat]
            description = [w for w in description if w not in string.punctuation]

            bow = GT_BOW['Scene{}_{}'.format(scene_id, idx)]
            coords = GT_OBJECTS['Scene{}_{}'.format(scene_id, idx)]

            # if there is no image for current turn: merge with next turn
            if turn['abs_d'] == '':
                continue

            # if no new object is added in image for current turn: merge with next turn
            if hamming_distance < 1:
                prev_bow = bow
                idx += 1
                continue

            # queue image, instruction, objects bow, object coordinates for saving
            if len(description) > 0:
                image = cv2.imread(os.path.join(images_path, 'Scene{}_{}.png'.format(scene_id, idx)))
                image = cv2.resize(image, (128, 128))

                images.append(image)
                utterences.append(str.join(' ', description))
                objects.append(bow)
                coordinates.append(coords)

                description = []
                idx += 1
                prev_bow = bow

        # add current scene's data to hdf5
        if len(images) > 0:
            if split == 'train':
                scene = h5_train.create_group(str(c_train))
                c_train += 1
            elif split == 'val':
                scene = h5_val.create_group(str(c_val))
                c_val += 1
            elif split == 'test':
                scene = h5_test.create_group(str(c_test))
                c_test += 1

            scene.create_dataset('images', data=images)
            dt = h5py.special_dtype(vlen=str)
            scene.create_dataset('utterences', data=np.string_(utterences), dtype=dt)
            scene.create_dataset('objects', data=np.array(objects))
            scene.create_dataset('coords', data=np.array(coordinates))
            scene.create_dataset('scene_id', data=scene_id)
        else:
            print(scene_id)


if __name__ == '__main__':
    create_h5()
