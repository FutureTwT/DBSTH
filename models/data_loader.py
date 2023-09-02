import sys
import h5py
from collections import namedtuple
from PIL import Image
import torch.utils.data as data

import models.networks.tool as tool
from utils import *

sys.path.append('..')

ROOT = 'Data/'

dataset_tuple = namedtuple('dataset_tuple', ['I_tr', 'T_tr', 'L_tr', 'I_tr_image',
                                             'I_db', 'T_db', 'L_db', 'I_db_image',
                                             'I_te', 'T_te', 'L_te', 'I_te_image'])

paths_feature = {
    'flickr': ROOT + 'flickr_all.mat',
    'nuswide': ROOT + 'nuswide_all.mat',
    'coco': ROOT + 'coco_all.mat'
}

paths_images = {
    'flickr': ROOT + 'flickr_images.mat',
    'nuswide': ROOT + 'nuswide_images.mat',
    'coco': ROOT + 'coco_images.mat'
}

# Check in 2022-1-3
def load_data(DATANAME): # training_ratio
    data = h5py.File(paths_feature[DATANAME], 'r')
    data_images = h5py.File(paths_images[DATANAME], 'r')

    I_tr_image = data_images['I_tr'][:].transpose(3, 2, 1, 0) # (H,W,C,N)
    I_tr = data['I_tr'][:].T
    T_tr = data['T_tr'][:].T
    L_tr = data['L_tr'][:].T

    I_db_image = data_images['I_db'][:].transpose(3, 2, 1, 0)
    I_db = data['I_db'][:].T
    T_db = data['T_db'][:].T
    L_db = data['L_db'][:].T

    I_te_image = data_images['I_te'][:].transpose(3, 2, 1, 0)
    I_te = data['I_te'][:].T
    T_te = data['T_te'][:].T
    L_te = data['L_te'][:].T

    return dataset_tuple(I_tr=I_tr, T_tr=T_tr, L_tr=L_tr, I_tr_image=I_tr_image,
                         I_db=I_db, T_db=T_db, L_db=L_db, I_db_image=I_db_image,
                         I_te=I_te, T_te=T_te, L_te=L_te, I_te_image=I_te_image)

# Check in 2022-1-3
# Flickr: RGB / NUSWIDE: RGB / COCO: BGR
class BSTH_dataset(data.Dataset):
    def __init__(self, text, label, name, **kwargs):
        ## origin-level
        self.image_available = False
        if kwargs['image'] is not None:
          self.image_available = True
          self.image = kwargs['image'] # (H,W,C,N)
          assert self.image.shape[0] == self.image.shape[1]
          print('[Dataloader]Data shape: ', self.image.shape)
          dp = tool.DataPreProcess()
          self.transform = dp.image_transform

        ## feature-level
        self.text = torch.Tensor(text)
        self.label = torch.Tensor(label)
        self.length = self.label.size(0)
        self.name = name

        self.training = False
        if 'B_tr' in kwargs.keys():
            print('Convert the dataset state to "Training"')
            self.training = True
            self.image_patch_feature = torch.Tensor(kwargs['image_patch_feature']) # pre-extracted [N, S, D]
            self.B_tr = torch.Tensor(kwargs['B_tr'])

    def __getitem__(self, item):
        item_list = [item]
        if self.image_available:
          # Numpy->PIL
          image_np = np.uint8(self.image[:, :, :, item].squeeze())
          if self.name == 'coco':
              image = Image.fromarray(image_np[:, :, ::-1], mode='RGB')  # (H,W,C)
          elif self.name == 'flickr' or self.name == 'nuswide':
              image = Image.fromarray(image_np, mode='RGB')  # (H,W,C)
          image = self.transform(image)
          item_list.append(image)
        else:
          item_list.append(-1)
        
        item_list.extend([self.text[item, :], self.label[item, :]])

        if self.training:
            item_list.append(self.image_patch_feature[item, :, :])
            item_list.append(self.B_tr[item, :])
        return item_list

    def __len__(self):
        return self.length