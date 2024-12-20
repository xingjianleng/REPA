import os
import json
import zipfile
import h5py
import io

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])


class CustomZipDataset(Dataset):
    # NOTE: This code is buggy and shouldn't be used because zip and multithreading are incompatible
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = os.path.join(data_dir, 'images.zip')
        self.features_dir = os.path.join(data_dir, 'vae-sd.zip')
        self.image_zip = zipfile.ZipFile(self.images_dir, "r")
        self.feature_zip = zipfile.ZipFile(self.features_dir, "r")

        # images
        self._image_fnames = {fname for fname in self.image_zip.namelist()}
        self.image_fnames = sorted(fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext)

        # features
        self._feature_fnames = {fname for fname in self.feature_zip.namelist()}
        self.feature_fnames = sorted(fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext)
        
        # labels
        fname = 'dataset.json'
        with self.feature_zip.open(fname, 'r') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __del__(self):
        # Close zip archives when done
        self.images_zip.close()
        self.vae_sd_zip.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with self.image_zip.open(image_fname, 'r') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        with self.feature_zip.open(feature_fname, 'r') as f:
            features = np.load(f)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])


def load_h5_file(hf, path):
    # Helper function to load files from h5 file
    if path.endswith('.png'):
        if pyspng is not None:
            rtn = pyspng.load(io.BytesIO(np.array(hf[path])))
        else:
            rtn = np.array(PIL.Image.open(io.BytesIO(np.array(hf[path]))))
        rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
    elif path.endswith('.json'):
        rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
    elif path.endswith('.npy'):
        rtn= np.array(hf[path])
    else:
        raise ValueError('Unknown file type: {}'.format(path))
    return rtn


class CustomH5Dataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_h5 = h5py.File(os.path.join(data_dir, 'images.h5'), "r")
        self.features_h5 = h5py.File(os.path.join(data_dir, 'vae-sd.h5'), "r")
        images_json = os.path.join(data_dir, 'images_h5.json')
        features_json = os.path.join(data_dir, 'vae-sd_h5.json')

        with open(images_json, 'r') as f:
            images_json = json.load(f)
        with open(features_json, 'r') as f:
            features_json = json.load(f)

        # images
        self._image_fnames = {fname for fname in images_json}
        self.image_fnames = sorted(fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext)

        # features
        self._feature_fnames = {fname for fname in features_json}
        self.feature_fnames = sorted(fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext)
        
        # labels
        fname = 'dataset.json'
        labels = load_h5_file(self.features_h5, fname)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __del__(self):
        self.images_h5.close()
        self.features_h5.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image = load_h5_file(self.images_h5, image_fname)
        if image_ext == '.npy':
            # npy needs some extra care
            image = image.reshape(-1, *image.shape[-2:])

        features = load_h5_file(self.features_h5, feature_fname)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])
