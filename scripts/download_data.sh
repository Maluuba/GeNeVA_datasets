#!/bin/bash

set -e

function download () {
    URL=$1
    TGTDIR=.
    if [ -n "$2" ]; then
        TGTDIR=$2
        mkdir -p $TGTDIR
    fi
    echo "Downloading ${URL} to ${TGTDIR}"
    wget $URL -P $TGTDIR
}

# GloVe data
if [ ! -f raw-data/GloVe/glove.840B.300d.txt ]
then
    download http://nlp.stanford.edu/data/glove.840B.300d.zip
    mkdir --parents raw-data/GloVe
    unzip glove.840B.300d.zip glove.840B.300d.txt -d raw-data/GloVe
    rm -f glove.840B.300d.zip
fi

# get CoDraw GitHub repository
if [ ! -d raw-data/CoDraw/asset ]
then
    git clone https://github.com/facebookresearch/CoDraw.git raw-data/CoDraw
fi

# get CoDraw individual json files
if [ ! -d raw-data/CoDraw/output ]
then
    cd raw-data/CoDraw
    if [ ! -f dataset/CoDraw_1_0.json ]
    then
        mkdir --parents dataset
        wget -O dataset/CoDraw_1_0.json https://github.com/facebookresearch/CoDraw/releases/download/v1.0/CoDraw_1_0.json
    fi
    python script/preprocess.py dataset/CoDraw_1_0.json
    rm dataset/CoDraw_1_0.json
    cd ../../
fi

# get CoDraw background image and object names
if [ ! -f raw-data/CoDraw/background.png ]
then
    download https://vision.ece.vt.edu/clipart/dataset/AbstractScenes_v1.1.zip
    unzip -j AbstractScenes_v1.1.zip AbstractScenes_v1.1/Pngs/background.png -d raw-data/CoDraw
    unzip -j AbstractScenes_v1.1.zip AbstractScenes_v1.1/VisualFeatures/10K_instance_occurence_58_names.txt -d raw-data/CoDraw
    rm AbstractScenes_v1.1.zip
fi
