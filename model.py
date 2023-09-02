from models.layers import *
from models.networks.vit_extractor import ViT, MaskedViT
from models.networks.glove_extractor import Glove
import torch.nn as nn
import torch
import time

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')

# Check in 2023-06-12
class Token2Concept(nn.Module):
    def __init__(self, args, modality_flag='image'):
        super(Token2Concept, self).__init__()
        if modality_flag == 'image':
          self.local_dim = args.image_local_dim
        elif modality_flag == 'text':
          self.local_dim = args.text_local_dim
        else:
          pass
        self.nbit = int(args.nbit)

        ordered_dict = OrderedDict()
        ordered_dict['conv'] = ReweightSparseFusion(patch_dim=self.local_dim, concept_lens=self.nbit)
        ordered_dict['bn'] = nn.BatchNorm1d(self.nbit) 
        ordered_dict['act'] = nn.Tanh() 
        self.agg = nn.Sequential(ordered_dict)

        # self._initialize()

    def _initialize(self):
        self.agg.conv.bias.data.fill_(0)

    def forward(self, token_sequence):
        concept = self.agg(token_sequence)
        return concept


# Check in 2022-1-4
class BSTH(nn.Module):
    def __init__(self, args):
        super(BSTH, self).__init__()
        ############################## Settings ##############################
        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        self.nbit = int(args.nbit)
        self.classes = args.classes
        self.batch_size = 0
        self.add_global = args.add_global

        # mask-1 args
        self.mask_concept = args.mask_concept
        self.mask_ratio = args.mask_ratio
        self.mask_flag = args.mask_flag # 'replace' / 'add'
        # mask-2 args
        self.mask_input = args.mask_input
  
        self.nhead = args.nhead
        self.act = args.trans_act
        self.dropout = args.dropout
        self.num_layer = args.num_layer

        ############################## Network ##############################
        self.image_backbone = ViT() # (N, S1+1, D1)
        if self.mask_input:
          print("Using masked ViT!")
          self.image_backbone_mask = MaskedViT(mask_ratio=self.mask_ratio)
        self.text_backbone = Glove(name=args.dataset) # (N, S2, D2)

        self.imageT2C = Token2Concept(args, modality_flag='image')
        self.textT2C = Token2Concept(args, modality_flag='text')

        self.imageMLP = FFN(hidden_dim=self.img_hidden_dim)
        self.textMLP = FFN(hidden_dim=self.txt_hidden_dim)

        # Self-attention + FFN
        self.imagePosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)
        self.textPosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)

        imageEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.common_dim,
                                                    activation=self.act,
                                                    dropout=self.dropout)
        imageEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=imageEncoderLayer, num_layers=self.num_layer, norm=imageEncoderNorm)

        textEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                   nhead=self.nhead,
                                                   dim_feedforward=self.common_dim,
                                                   activation=self.act,
                                                   dropout=self.dropout)
        textEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.textTransformerEncoder = TransformerEncoder(encoder_layer=textEncoderLayer, num_layers=self.num_layer, norm=textEncoderNorm)

        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=self.nbit * self.common_dim, out_channels=self.nbit * self.common_dim // 2, kernel_size=1, groups=self.nbit),
            nn.BatchNorm2d(self.nbit * self.common_dim // 2),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.nbit * self.common_dim // 2, out_channels=self.nbit, kernel_size=1, groups=self.nbit),
            nn.Tanh()
        )

        self.classify = nn.Linear(self.nbit, self.classes)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    # Check in 2023-06-05
    def add_noise(self, x: torch.Tensor, mean=0, var=0.1, mask_ratio=0.5, flag='replace')->torch.Tensor:
        # add / replace random gaussian noise on inputs
        N, K, D = x.shape
        noise = (var**0.5) * torch.randn(N, K, D) + mean
        noise = noise.cuda()
        mask = torch.zeros(N, K)
        num = int(mask_ratio * K)

        for i in range(N):
          indices = torch.randperm(K)
          mask[i, indices[:num]] = 1
        mask = mask.unsqueeze(-1).expand(-1, -1, D).cuda()

        if flag == 'add':
          x = x + noise * mask
        elif flag == 'replace':
          x = x * ~mask.bool() + noise * mask

        return x

    def forward(self, image, text, image_pfeat):
        self.batch_size = len(image)
        ##### Base branch #####
        with torch.no_grad():
            if not self.training: # test-time
                image_backbone_feature = self.image_backbone(image) # (N, S1+1, D)
            text_backbone_feature = self.text_backbone(text) # (N, S2, D)

        # extract sequence
        if self.add_global:
            image_feature = image_backbone_feature if not self.training else image_pfeat
            text_global_feature = torch.mean(text_backbone_feature, dim=1).unsqueeze(dim=1) # [N, 1, D]
            text_feature = torch.cat((text_global_feature, text_backbone_feature), dim=1)
        else:
            image_feature = image_backbone_feature[:, 1:, :] if not self.training else image_pfeat[:, 1:, :]
            text_feature = text_backbone_feature

        # generate concept using token
        imageC = self.imageT2C(image_feature)
        textC = self.textT2C(text_feature)
        imageC1 = self.imageMLP(imageC).permute(1, 0, 2) # [S, N, D]
        textC1 = self.textMLP(textC).permute(1, 0, 2)

        # transformer
        imageSrc = self.imagePosEncoder(imageC1)
        textSrc = self.textPosEncoder(textC1)
        imageMemory = self.imageTransformerEncoder(imageSrc)
        textMemory = self.textTransformerEncoder(textSrc)
        memory = imageMemory + textMemory # [S, N, D]

        # generate hash code
        code = self.hash(memory.permute(1, 0, 2).reshape(self.batch_size, self.nbit * self.common_dim, 1, 1)).squeeze()
        logit = self.classify(code)

        ##### Mask input patches branch #####
        if self.mask_input and self.training:
            with torch.no_grad():
              image_feature_mask = self.image_backbone_mask(image)[:, 1:, :] # w/o global
            imageC_mask = self.imageT2C(image_feature_mask)
            imageC1_mask = self.imageMLP(imageC_mask).permute(1, 0, 2) 
            
            imageSrc_mask = self.imagePosEncoder(imageC1_mask)
            imageMemory_mask = self.imageTransformerEncoder(imageSrc_mask)
            
            memory_mask = imageMemory_mask + textMemory

            code_mask = self.hash(memory_mask.permute(1, 0, 2).reshape(self.batch_size, self.nbit * self.common_dim, 1, 1)).squeeze()
            logit_mask = self.classify(code_mask)

            return code, logit, code_mask, logit_mask

        ##### Mask concept branch #####
        if self.mask_concept and self.training:
          # only performed in training stage
          # contrastive
          imageC_noise = self.add_noise(imageC, mask_ratio=self.mask_ratio, flag=self.mask_flag)
          textC_noise = self.add_noise(textC, mask_ratio=self.mask_ratio, flag=self.mask_flag)
          imageC1_noise = self.imageMLP(imageC_noise).permute(1, 0, 2) # [S, N, D]
          textC1_noise = self.textMLP(textC_noise).permute(1, 0, 2)

          imageSrc_noise = self.imagePosEncoder(imageC1_noise)
          textSrc_noise = self.textPosEncoder(textC1_noise)

          imageMemory_noise = self.imageTransformerEncoder(imageSrc_noise)
          textMemory_noise = self.textTransformerEncoder(textSrc_noise)
          memory_noise = imageMemory_noise + textMemory_noise # [S, N, D]

          code_noise = self.hash(memory_noise.permute(1, 0, 2).reshape(self.batch_size, self.nbit * self.common_dim, 1, 1)).squeeze()
          logit_noise = self.classify(code_noise)

          return code, logit, code_noise, logit_noise
        
        return code, logit, None, None

# Check in 2022-1-4
class L2H_Prototype(nn.Module):
    def __init__(self, args):
        super(L2H_Prototype, self).__init__()
        self.classes = args.classes
        self.nbit = args.nbit
        self.d_model = args.nbit
        self.num_layer = 1
        self.nhead = 1
        self.batch_size = 0

        self.labelEmbedding = nn.Embedding(self.classes + 1, self.d_model, padding_idx=0) # [N, S=args.classes, D=args.nbit]

        # [S, N, D]
        labelEncoderLayer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                       nhead=self.nhead,
                                                       dim_feedforward=self.d_model,
                                                       activation='gelu',
                                                       dropout=0.5)
        labelEncoderNorm = nn.LayerNorm(normalized_shape=self.d_model)
        self.labelTransformerEncoder = TransformerEncoder(encoder_layer=labelEncoderLayer, num_layers=self.num_layer, norm=labelEncoderNorm)

        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=self.classes * self.nbit, out_channels=self.classes * self.nbit, kernel_size=1, groups=self.classes),
            nn.Tanh()
        )
        self.classify = nn.Linear(self.nbit, self.classes)

    def forward(self, label):
        self.batch_size = label.size(0)

        index = torch.arange(1, self.classes+1).cuda().unsqueeze(dim=0) # [N=1, C]
        label_embedding = self.labelEmbedding(index) # (N=1, C, K) without padding index
        memory = self.labelTransformerEncoder(label_embedding.permute(1, 0, 2)) # (C, 1, K)

        prototype = self.hash(memory.permute(1, 0, 2).reshape(1, self.classes * self.nbit, 1, 1)).squeeze()
        prototype = prototype.squeeze().reshape(self.classes, self.nbit)

        code = torch.matmul(label, prototype)

        pred = self.classify(code)
        return prototype, code, pred

