import torch
import torch.nn as nn
import torch.utils.data as tdata

from PIL import Image
import timm
import numpy as np
from tqdm import tqdm

import models.networks.tool as tool
from models.layers import patch_masking

ROOT = '' # TODO: give you data root

# Check in 2023-03-02
class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super(ViT, self).__init__()
        self.encoder = timm.create_model(model_name=model_name, pretrained=True)
        self.dim_out = self.encoder.head.in_features

    def forward_features(self, x):
        # x: B C H W
        with torch.no_grad():
            x = self.encoder.patch_embed(x)
            cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.encoder.pos_drop(x + self.encoder.pos_embed)
            x = self.encoder.blocks(x)
            # x = self.encoder.norm(x) # everything is ok
        return x

    def forward(self, x):
        # x[0] is global feature, x[1:end] are local features
        x = self.forward_features(x)
        return x
    
# Check in 2023-06-12
class MaskedViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', mask_ratio=0.1):
        super(MaskedViT, self).__init__()
        self.mask_ratio = mask_ratio
        
        self.encoder = timm.create_model(model_name=model_name, pretrained=True)
        self.dim_out = self.encoder.head.in_features

    def forward_features(self, x):
        # x: B C H W
        with torch.no_grad():
            x = self.encoder.patch_embed(x)
            x = x + self.encoder.pos_embed[:, 1:, :]

            x = patch_masking(x, self.mask_ratio)

            cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
            cls_token = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.encoder.pos_drop(x)
            x = self.encoder.blocks(x)
            # x = self.encoder.norm(x) # everything is ok
        return x

    def forward(self, x):
        # x[0] is global feature, x[1:end] are local features
        x = self.forward_features(x)
        return x

# Check in 2023-03-05
# Flickr: RGB / NUSWIDE: RGB / COCO: BGR
class ImageLoader(tdata.Dataset):
    def __init__(self, data: np.ndarray, name):
        self.name = name
        self.data = data # (H,W,C,N)
        assert self.data.shape[0] == self.data.shape[1]
        print('[Dataloader]Data shape: ', self.data.shape)
        dp = tool.DataPreProcess()
        self.transform = dp.image_transform

    def __getitem__(self, item):
        # Numpy->PIL
        image_np = np.uint8(self.data[:, :, :, item].squeeze())
        if self.name == 'coco':
            image = Image.fromarray(image_np[:,:,::-1], mode='RGB')  # (H,W,C)
        elif self.name == 'flickr' or self.name == 'nuswide':
            image = Image.fromarray(image_np, mode='RGB') # (H,W,C)
        image = self.transform(image)
        return image.cuda()

    def __len__(self):
        return self.data.shape[3]

# Check in 2023-03-05
def vit_extractor(data, name):
    ''' An example about usage.
    model = ViT().eval().cuda()
    x = torch.randn([1, 3, 224, 224]).cuda()
    x = model(x)
    print(x.shape)
    '''
    model = ViT().eval().cuda()
    dataloader = tdata.DataLoader(ImageLoader(data, name),
                                  batch_size=512,
                                  shuffle=False)
    all_feature = []
    for batch_image in tqdm(dataloader):
        batch_local_feature = model(batch_image).detach().cpu()
        all_feature.append(batch_local_feature)

    final_feature = torch.cat(all_feature, dim=0).numpy()
    return final_feature

