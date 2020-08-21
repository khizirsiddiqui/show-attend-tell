import json
import h5py
import cv2
import numpy as np
import os.path as path
from collections import Counter
from datetime import datetime
from random import seed, choice, sample
from PIL import Image
from tqdm import tqdm


HEIGHT = 256
WIDTH = 256

BEGIN_TOKEN = '<begin>'
END_TOKEN = '<end>'
UNKNOWN_TOKEN = '<unknown>'
PAD_TOKEN = '<zeropad>'



def print_message(msg, tag):
    message = '[{0: <8}] '.format(str(tag))
    message += datetime().now().strftime("%d-%m-%Y %H:%M") + ': '
    message += msg
    print(message)


def preprocess_data(json_path, img_folder, output_folder, captions_per_img=5, maxlen=100, min_word_freq=10, rndseed=1331, verbose=1):

        with open(json_path, 'r') as f:
            data = json.load(f)

        all_image_paths = {
            'train': []
            'val': []
            'test': []
        }
        all_image_captions = {
            'train': []
            'val': []
            'test': []
        }

        word_counter = Counter()

        if verbose:
            print_message("Loading Captions...", 'LOG')

        for img in tqdm(data['images']):
            img_captions = []
            for item in img['sentences']:
                token = item['tokens']
                word_counter.update(token)
                if len(token) <= maxlen:
                    img_captions.append(token)

            if len(img_captions) == 0:
                continue

            img_path = path.join(img_folder, img['filename'])

            img_split = img['split']
            all_image_paths[img_split].append(img_path)
            all_image_captions[img_split].append(img_captions)

        for i in range(3):
            assert len(all_image_paths[i]) == len(all_image_captions[i]), "key {}: Unequal lengths of image paths and captions.".format(i)

        if verbose:
            print_message("Creating Word Map...", 'LOG')

        word_list = []
        for word in word_counter.keys():
            if word_counter[word] > min_word_freq:
                word_list.append(word)

        word_map = {k: v+1 for v, k in enumerate(word_list)}
        word_map[UNKNOWN_TOKEN] = len(word_map) + 1
        word_map[BEGIN_TOKEN] = len(word_map) + 1
        word_map[END_TOKEN] = len(word_map) + 1
        word_map[PAD_TOKEN] = 0

        fname = 'flikr8k_' + str(captions_per_img) + '_' + str(min_word_freq)

        wmap_fname = path.join(output_folder, 'word_map_', fname, '.json')
        with open(wmap_fname, 'w') as f:
            json.dump(word_map, f)

        if verbose:
            print_message("Word Map saved to " + wmap_fname, 'SUCCESS')

        seed(rndseed)

        for split in ['train', 'val', 'test']:
            if verbose:
                print_message("Creating {} hdf5 file...".format(split), 'LOG')

            h5py_fname = path.join(output_folder, split + '_images_' + fname + '.h5py')

            with h5py.File(h5py_fname, 'a') as h:
                h.attrs['captions_per_img'] = captions_per_img
                images = h.create_dataset('images', (len(all_image_paths[split], 3, HEIGHT, WIDTH), dtype='uint8'))

                encoded_captions = []
                caption_lengths = []

                for idx, img_path in enumerate(tqdm(all_image_paths[split])):
                    img_captions = all_image_captions[split][idx]

                    missing_cap_count = captions_per_img - len(img_captions)

                    if missing_cap_count > 0:
                        captions = img_captions + [choice(img_captions) for _ in range(missing_cap_count)]
                    elif missing_cap_count == 0:
                        captions = img_captions
                    else:
                        captions = sample(img_captions, k=captions_per_img)

                    img = cv2.imread(img_path)

                    if len(img.shape) == 2:
                        # Gray Scale Image
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)
                    img = imresize(img, (HEIGHT, WIDTH))
                    img = np.clip(img, 0, 255)
                    # BGR to RBG
                    img = img.transpose(2, 0, 1)

                    images[idx] = img

                    for each_caption in captions:
                        encoded_caption = [word_counter[BEGIN_TOKEN]]
                        for word in each_caption:
                            encoded_caption += [word_counter.get(word, word_counter[UNKNOWN_TOKEN])]
                        encoded_caption += [word_counter[END_TOKEN]]
                        encoded_caption += [word_counter[PAD_TOKEN]] * (maxlen - len(each_caption))
                        cap_len = len(each_caption) + 2

                        encoded_captions.append(encoded_caption)
                        caption_lengths.append(cap_len)

                assert images.shape[0] * captions_per_img == len(encoded_captions) == len(caption_lengths)

                cap_fpath = path.join(output_folder, split + '_captions_' + fname + '.json')
                with open(cap_fpath, 'w') as f:
                    json.dump(encoded_captions, f)
                if verbose:
                    print_message('{} captions saved to {}.'.format(split, cap_fpath), 'SUCCESS')

                caplen_fpath = path.join(output_folder, split + '_caption_lengths_' + fname + '.json')
                with open(caplen, 'w') as f:
                    json.dump(caption_lengths, f)
                if verbose:
                    print_message('{} caption lengths saved to {}.'.format(split, caplen_fpath), 'SUCCESS')

            if verbose:
                print_message('{} images saved to {}.'.format(split, h5py_fname), 'SUCCESS')

        if verbose:
            print_message('Image and caption pre-processing of dataset completed.', 'SUCCESS')
