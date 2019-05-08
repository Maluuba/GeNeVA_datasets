# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to create a list of all objects in the i-CLEVR data
"""
import itertools
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SHAPES = ['cube', 'sphere', 'cylinder']


def create_vocab():
    obj_list = list(itertools.product(SHAPES, COLORS))
    obj_list = [' '.join(x) for x in obj_list]

    with open(keys['iclevr_objects'], 'w') as f:
        for item in obj_list:
            f.write("%s\n" % item)


if __name__ == '__main__':
    create_vocab()
