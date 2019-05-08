# Generative Neural Visual Artist (GeNeVA) - Datasets - Generation Code

Scripts to generate the `CoDraw` and `i-CLEVR` datasets used for the `GeNeVA` task proposed in [Tell, Draw, and Repeat: Generating and modifying images based on continual linguistic instruction](https://arxiv.org/abs/1811.09845).

## Setup ##

### 1. Install Miniconda

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh

You will now have to restart your shell for the path changes to take effect.

### 2. Clone the repository

    git clone git@github.com:Maluuba/GeNeVA_datasets.git  # use https://github.com/Maluuba/GeNeVA_datasets.git for HTTPS
    cd GeNeVA_datasets

### 3. Create a conda environment for this repository

    conda env create -f environment.yml

### 4. Activate the environment

    source activate geneva

### 5. Download external data files

    ./scripts/download_data.sh

### 6. Download GeNeVA data files to the repository

Download the [GeNeVA zip file](https://www.microsoft.com/en-us/research/project/generative-neural-visual-artist-geneva/) and extract it as specified below:
 - `GeNeVA-v1.zip`
    ```
    unzip GeNeVA-v1.zip
    ```
    Please review the LICENSE for the GeNeVA zip file in the extracted `GeNeVA-v1` folder
 - `data.rar`: pre-generated data files for both datasets
    ```
    rar x GeNeVA-v1/data.rar ./  # `sudo apt-get install rar` if rar is not installed
    ```
 - `CoDraw_images.rar`: CoDraw images for each scene's json
    ```
    rar x GeNeVA-v1/CoDraw_images.rar raw-data/CoDraw
    ```
 - `i-CLEVR.rar`: i-CLEVR scene images, scene jsons, background image
    ```
    rar x GeNeVA-v1/i-CLEVR.rar raw-data/
    ```

### 7. Generate dataset HDF5 files

 - Vocabulary
    ```
    python scripts/joint_codraw_iclevr/generate_glove_file.py
    ```
 - CoDraw
    ```
    python scripts/codraw_dataset_generation/codraw_add_data_to_raw.py
    python scripts/codraw_dataset_generation/codraw_raw_to_hdf5.py       # dataset for GeNeVA-GAN
    python scripts/codraw_dataset_generation/codraw_object_detection.py  # dataset for Object Detector & Localizer
    ```
 - i-CLEVR
    ```
    python scripts/iclevr_dataset_generation/iclevr_add_data_to_raw.py
    python scripts/iclevr_dataset_generation/iclevr_raw_to_hdf5.py       # dataset for GeNeVA-GAN
    python scripts/iclevr_dataset_generation/iclevr_object_detection.py  # dataset for Object Detector & Localizer
    ```

### 8. (Optional) Downloaded data can now be deleted

    rm raw-data/ -rf
    rm GeNeVA-v1/ -rf
    rm GeNeVA-v1.zip

## Reference ##
If you use this code or the GeNeVA datasets as part of any published research, please cite the following paper:

Alaaeldin El-Nouby, Shikhar Sharma, Hannes Schulz, Devon Hjelm, Layla El Asri, Samira Ebrahimi Kahou, Yoshua Bengio, and Graham W. Taylor.
**"Tell, Draw, and Repeat: Generating and modifying images based on continual linguistic instruction"**
*arXiv preprint arXiv:1811.09845* (2018).

```bibtex
@article{elnouby2018tell_draw_repeat,
    author  = {El{-}Nouby, Alaaeldin and Sharma, Shikhar and Schulz, Hannes and Hjelm, Devon and El Asri, Layla and Ebrahimi Kahou, Samira and Bengio, Yoshua and Taylor, Graham W.},
    title   = {Tell, Draw, and Repeat: Generating and modifying images based on continual linguistic instruction},
    journal = {CoRR},
    volume  = {abs/1811.09845},
    year    = {2018},
    url     = {http://arxiv.org/abs/1811.09845},
    archivePrefix = {arXiv},
    eprint  = {1811.09845}
}
```

## Microsoft Open Source Code of Conduct ##
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License ##
See [LICENSE.txt](LICENSE.txt).
