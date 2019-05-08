# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Script to generate the GloVe embedding file for the CoDraw and i-CLEVR dataset vocabularies
"""
from tqdm import tqdm
import yaml


with open('config.yml', 'r') as f:
    keys = yaml.load(f, Loader=yaml.FullLoader)


def generate_glove_file():
    codraw_vocab = keys['codraw_vocab']
    clevr_vocab = keys['iclevr_vocab']
    output_file = keys['glove_output']
    original_glove = keys['glove_source']

    # read CoDraw vocabulary
    with open(codraw_vocab, 'r') as f:
        codraw_vocab = f.readlines()
        codraw_vocab = [x.strip().rsplit(' ', 1)[0] for x in codraw_vocab]

    # read i-CLEVR vocabulary
    with open(clevr_vocab, 'r') as f:
        clevr_vocab = f.readlines()
        clevr_vocab = [x.strip().rsplit(' ', 1)[0] for x in clevr_vocab]

    # combine vocabularies and add special tokens for CoDraw Drawer and Teller
    codraw_vocab += clevr_vocab + ['<drawer>', '<teller>']
    codraw_vocab = list(set(codraw_vocab))
    codraw_vocab.sort()

    print('Loading GloVe file. This might take a few minutes.')
    with open(original_glove, 'r') as f:
        original_glove = f.readlines()
        tok_glove_pairs = [x.strip().split(' ', 1) for x in original_glove]

    # extract GloVe vectors for vocabulary tokens
    for token, glove_emb in tqdm(tok_glove_pairs):
        if token == 'unk':
            unk_embedding = glove_emb
        try:
            token_idx = codraw_vocab.index(token)
        except ValueError:
            continue
        else:
            codraw_vocab[token_idx] = ' '.join([token, glove_emb])

    # set Drawer and Teller token vectors; assign 'unk' GloVe embedding to unknown words
    unk_count = 0
    for itidx, item in enumerate(codraw_vocab):
        if len(item.split(' ')) == 1:
            if item == '<drawer>':
                codraw_vocab[itidx] = ' '.join(['<drawer>', ('0.1 ' * 150 + '0.0 ' * 150)[:-1]])
            elif item == '<teller>':
                codraw_vocab[itidx] = ' '.join(['<teller>', ('0.0 ' * 150 + '0.1 ' * 150)[:-1]])
            else:
                unk_count += 1
                codraw_vocab[itidx] = ' '.join([item, unk_embedding])

    # write GloVe vector file for the CoDraw and i-CLEVR datasets combined
    with open(output_file, 'w') as f:
        for item in codraw_vocab:
            f.write('%s\n' % item)

    print('Total words in vocab: {}\n`unk` embedding words: {}'.format(len(codraw_vocab), unk_count))


if __name__ == '__main__':
    generate_glove_file()
