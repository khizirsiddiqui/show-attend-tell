import torch
import h5py
import json
import os.path as path
from torch.utils.data import Dataset


class flickr8k(Dataset):
    def __init__(self, basepath, data_folder, split, transform=None):
        self.split = split

        h5py_fpath = path.join(data_folder, split + '_images_', basepath + '.h5py')
        self.h5py_file = h5py.File(h5py_fpath, 'r')
        self.images = self.h['images']
        self.captions_per_img = self.h.attrs['captions_per_img']

        encoded_cap_fpath = path.join(data_folder, split + '_captions_', basepath, '.json')
        with open(encoded_cap_fpath, 'r') as f:
            self.captions = json.load(f)

        encoded_caplen_fpath = path.join(data_folder, split + '_caption_lengths_', basepath, '.json')
        with open(encoded_caplen_fpath, 'r') as f:
            self.caption_lengths = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = torch.FloatTensor(self.images[i // self.captions_per_img])
        if self.transform:
            image = self.transform(image)

        caption = torch.LongTensor(self.captions[i])
        caption_lengths = torch.LongTensor(self.caption_lengths[i])

        if self.split is 'train':
            return image, caption, caption_lengths
        else:
            all_captions = torch.LongTensor(
                self.captions[((i // self.captions_per_img) * self.cpi):(((i // self.captions_per_img) * self.captions_per_img) + self.captions_per_img)]
            )
            return image, caption, caption_lengths, all_captions
