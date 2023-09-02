import argparse
from easydict import EasyDict as edict
import json
import scipy.io as sio
import tqdm

import torch.nn.functional as F

from model import BSTH, L2H_Prototype
from models.loss.concept_CL import CL
from models.data_loader import *

def train(args, dset):
    print('=' * 30)
    print('Training Stage...')
    print('Train size: %d' % (dset.I_tr.shape[0]))
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]

    ## Defination
    loss_l2 = torch.nn.MSELoss()

    l2h = L2H_Prototype(args=args)
    l2h.train().cuda()

    bsth = BSTH(args=args)
    bsth.train().cuda()

    ## Optimizer
    optimizer_L2H = torch.optim.Adam(l2h.parameters(), lr=args.lr)
    backbone_params = [map(id, bsth.image_backbone.parameters()), map(id, bsth.text_backbone.parameters())]
    base_params = filter(lambda p: id(p) not in backbone_params, bsth.parameters())
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}])

    ## Preprocess
    _, COO_matrix = get_COO_matrix(dset.L_tr)
    COO_matrix = torch.Tensor(COO_matrix).cuda()

    ## Stage 1: learn the hash codes
    batch_label = torch.Tensor(dset.L_tr).cuda()

    for epoch in range(args.epochs_pre):
        optimizer_L2H.zero_grad()

        prototype, code, pred = l2h(batch_label)
        B = torch.sign(code)
        prototype_norm = F.normalize(prototype)

        recon_loss = loss_l2(torch.sigmoid(pred), batch_label) * args.param_recon_pre
        sign_loss = loss_l2(code, B) * args.param_sign_pre
        bal_loss = torch.sum(code) / code.size(0) * args.param_bal_pre
        static_loss = loss_l2(prototype_norm.mm(prototype_norm.t()), COO_matrix) * args.param_static_pre

        loss = recon_loss + sign_loss + bal_loss + static_loss

        loss.backward()
        optimizer_L2H.step()

        if (epoch + 1) % 2 == 0:
            print('Epoch [%3d/%3d], Loss: %.4f, Loss-R: %.4f, Loss-B: %.4f, Loss-S: %.4f'
                    % (epoch + 1, args.epochs_pre, loss.item(), recon_loss.item(), sign_loss.item(), static_loss.item()))

    l2h.eval() 

    with torch.no_grad():
        B_tr = l2h(batch_label)[1]     
    B_tr = np.sign(B_tr.data.cpu().numpy())

    del batch_label

    map_train = calculate_map(B_tr, B_tr, dset.L_tr, dset.L_tr)
    print('Training MAP: %.4f' % (map_train))
    print('=' * 30)

    ## Stage 2: learn the hash functions(Transformer)
    from models.networks.vit_extractor import vit_extractor
    I_tr_local = vit_extractor(dset.I_tr_image, args.dataset)
    assert I_tr_local.shape[1] == 197

    kwargs_image = dset.I_tr_image if args.mask_input else None
    train_loader = data.DataLoader(BSTH_dataset(dset.T_tr, dset.L_tr, args.dataset, image=kwargs_image, image_patch_feature=I_tr_local, B_tr=B_tr),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)

    cl_learner = CL().cuda()

    for epoch in range(args.epochs):
        for i, item_list in enumerate(train_loader):
            _, image, text, label, image_pfeat, B_gnd = item_list

            _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())
            aff_label = torch.from_numpy(aff_label).to('cuda:0', non_blocking=True)

            # (nonblocking and overlapping) from cpu to gpu
            image = image.to('cuda:0', non_blocking=True)
            text = text.to('cuda:0', non_blocking=True)
            label = label.to('cuda:0', non_blocking=True)
            image_pfeat = image_pfeat.to('cuda:0', non_blocking=True)
            B_gnd = B_gnd.to('cuda:0', non_blocking=True)

            optimizer.zero_grad()
            H, pred, H_noise, _ = bsth(image, text, image_pfeat)
            H_norm = F.normalize(H)

            clf_loss = loss_l2(torch.sigmoid(pred), label) * args.param_clf
            sign_loss = loss_l2(H, B_gnd) * args.param_sign
            similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)  * args.param_sim
            
            loss = clf_loss + sign_loss + similarity_loss

            if args.mask_concept or args.mask_input:
              contrastive_loss = cl_learner(H_noise, H.detach()) * args.param_cl
              loss = loss + contrastive_loss

            loss.backward()
            optimizer.step()

            if (i + 1) == len(train_loader) and (epoch + 1) % 2 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-B: %.4f, Loss-S: %.4f'
                      % (epoch + 1, args.epochs, loss.item(), clf_loss.item(), sign_loss.item(), similarity_loss.item()))

    # save model
    type_name = ''
    if args.mask_input:
        type_name = 'MaskIn'
    elif args.mask_concept:
        type_name = 'MaskReplace' if args.mask_flag == 'replace' else 'MaskAdd'
    else:
        pass
    path = 'params/DBSTH_' + args.dataset + '_' + type_name + '_' + str(args.nbit) + '.pt'
    torch.save(bsth, path)

    return bsth

def eval(model, dset, args):
    model.eval()
    ## Retrieval
    print('=' * 30)
    print('Testing Stage...')
    print('Retrieval size: %d' % (dset.I_db.shape[0]))
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]

    retrieval_loader = data.DataLoader(BSTH_dataset(dset.T_db, dset.L_db, args.dataset, image=dset.I_db_image),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)

    retrievalP = []
    for item_list in tqdm.tqdm(retrieval_loader):
        _, image, text, _ = item_list

        # (nonblocking and overlapping) from cpu to gpu
        image = image.to('cuda:0', non_blocking=True)
        text = text.to('cuda:0', non_blocking=True)
        
        with torch.no_grad():
            H, _, _, _ = model(image, text, None)
        retrievalP.append(H.data.cpu().numpy())

    retrievalH = np.concatenate(retrievalP)
    retrievalCode = np.sign(retrievalH)

    ## Query
    print('Query size: %d' % (dset.I_te.shape[0]))
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]

    val_loader = data.DataLoader(BSTH_dataset(dset.T_te, dset.L_te, args.dataset, image=dset.I_te_image),
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    valP = []
    for item_list in tqdm.tqdm(val_loader):
        _, image, text, _ = item_list

        # (nonblocking and overlapping) from cpu to gpu
        image = image.to('cuda:0', non_blocking=True)
        text = text.to('cuda:0', non_blocking=True)

        with torch.no_grad():
            H, _, _, _ = model(image, text, None)
        valP.append(H.data.cpu().numpy())

    valH = np.concatenate(valP)
    valCode = np.sign(valH)

    if args.save_flag:
        ## Save
        _dict = {
            'retrieval_B': retrievalCode.astype(np.int8),
            'val_B': valCode.astype(np.int8),
            'cateTrainTest': np.sign(dset.L_db @ dset.L_te.T).astype(np.int8),
            'L_db': dset.L_db,
            'L_te': dset.L_te
        }
        sava_path = 'Hashcode/BSTH_' + str(args.nbit) + '_' + args.dataset + '_bits.mat'
        sio.savemat(sava_path, _dict)
    else:
        return retrievalCode, valCode, dset.L_db, dset.L_te.T

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Net basic params
    parser.add_argument('--nbit', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')

    ## Loss params
    parser.add_argument('--param_clf', type=float, default=1) 
    parser.add_argument('--param_sign', type=float, default=0.01)
    parser.add_argument('--param_sim', type=float, default=1)
    parser.add_argument('--param_cl', type=float, default=0.1)
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    ## Flag params
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    seed_setting(args.seed)

    ## load static config
    if args.dataset == 'flickr':
        Config = 'config/flickr.json'
    elif args.dataset == 'nuswide':
        Config = 'config/nuswide.json'
    elif args.dataset == 'coco':
        Config = 'config/coco.json'
    else:
        raise Exception('Error dataset!')

    with open(Config, 'r') as f:
        config = edict(json.load(f))

    # dynamic config can cover the static config
    args = edict({**config, **vars(args)})

    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))
    print('Nbit: %d' % (args.nbit))

    args.L2H_hidden_dim.append(args.nbit)

    model = train(args, dset)
    eval(model, dset, args)

