import os
import os.path as osp
from types import SimpleNamespace

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
from ultralytics import YOLO
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1, FogPassFilterLoss, PairedFogPassFilterLoss
from dataset.paired_cityscapes import PairedCityscapes
from dataset.foggy_zurich import FoggyZurich
from plot import plot_cw_sf_layer
from test import convert_labels_to_ultralytics_format, test_model
from utils.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
import wandb
from kaggle_secrets import UserSecretsClient


def gram_matrix(tensor):
    """Compute Gram matrix for feature style comparison."""
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def compute_iou(boxes1, boxes2, K=300):
    """
    Compute mean IoU with Hungarian matching, keeping top-K detections by confidence.
    Args:
        boxes1: [N, 5] (x1, y1, x2, y2, conf)
        boxes2: [M, 5] (x1, y1, x2, y2, conf)
        K: number of top boxes to keep
    """
    # --- Filter top-K by confidence ---
    boxes1 = boxes1[boxes1[:, 4].argsort(descending=True)[:K], :4]
    boxes2 = boxes2[boxes2[:, 4].argsort(descending=True)[:K], :4]

    N, M = boxes1.shape[0], boxes2.shape[0]
    if N == 0 or M == 0:
        return torch.tensor(0.0, device=boxes1.device)

    # --- Compute IoU matrix ---
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)  # [N, M]

    # --- Hungarian matching ---
    iou_np = iou.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(-iou_np)
    matched_iou = iou[row_ind, col_ind]

    return matched_iou.mean()


def main():
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb")
    wandb.login(key=secret_value_0)

    args = get_arguments()

    yolo = YOLO("yolov8s.pt")
    yolo.to(args.gpu)
    yolo.model.args = SimpleNamespace(box=0.05, cls=0.5, dfl=1.5)

    # Initialize model
    model = YOLO('yolov8s.pt').model

    # load weights (strict=True will raise if any shape mismatches)
    model.load_state_dict(torch.load('cityscapes_finetuned.pth'), strict=True)
    model.train()
    model.to(args.gpu)

    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # Initialize logging
    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'
    wandb.init(project='FIFO', name=run_name)
    wandb.config.update(args)

    # Enable CuDNN
    cudnn.enabled = True

    # Initialize fog-pass filters (adjust input sizes based on YOLOv8n backbone)

    FogPassFilter2 = FogPassFilter_res1(8256)

    FogPassFilter2_optimizer = torch.optim.Adam(FogPassFilter2.parameters(), lr=1e-4)

    FogPassFilter2.to(args.gpu)
    paired_loss_fn = PairedFogPassFilterLoss(variant='cos', normalize=True).to(args.gpu)

    all_factors = {
        'layer0': {'CW': [], 'SF': []},
        'layer1': {'CW': [], 'SF': []}
    }

    # Data loaders
    cwsf_dataset = PairedCityscapes(args.data_dir, set=args.set, max_iters=args.num_steps * args.batch_size,
                                    img_size=args.img_size)
    cwsf_loader = data.DataLoader(
        cwsf_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
        collate_fn=cwsf_dataset.collate_fn
    )

    cwsf_loader_iter = iter(cwsf_loader)

    # Optimizer and scheduler
    optimiser = get_optimisers(model)
    scheduler = get_lr_schedulers(optimiser, args.num_steps)
    opts = make_list(optimiser)

    # Feature extraction hooks
    feature_layers = [4]  # Adjust based on YOLOv8s architecture
    features = {idx: [] for idx in feature_layers}
    handles = []

    def hook_fn(layer_idx):
        def hook(module, input, output):
            features[layer_idx].append(output.detach())

        return hook

    for layer_idx in feature_layers:
        handle = model.model[layer_idx].register_forward_hook(hook_fn(layer_idx))
        handles.append(handle)

    # Training loop
    for i_iter in tqdm(range(10000)):
        loss_det_cw_value = 0
        loss_det_sf_value = 0
        loss_fsm_value = 0
        loss_con_value = 0

        for opt in opts:
            opt.zero_grad()

        FogPassFilter2_optimizer.zero_grad()

        for sub_i in range(args.iter_size):
            # Step 1: Train fog-pass filters

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            for param in FogPassFilter2.parameters():
                param.requires_grad = True

            # Get batches
            batch_cwsf = next(cwsf_loader_iter, None)
            if batch_cwsf is None:
                cwsf_loader_iter = iter(cwsf_loader)
                batch_cwsf = next(cwsf_loader_iter)
            cw_img, sf_img, cw_label, sf_label, _, cw_domains, sf_domains = batch_cwsf
            cw_label = [{k: v.to(args.gpu) for k, v in label.items()} for label in cw_label]
            sf_label = [{k: v.to(args.gpu) for k, v in label.items()} for label in sf_label]

            sf_img, cw_img = (Variable(sf_img).to(args.gpu), Variable(cw_img).to(args.gpu))

            # Forward passes
            for key in features:
                features[key].clear()
            _ = model(cw_img)
            _ = model(sf_img)

            features_cw = {idx: features[idx][0] for idx in feature_layers}
            features_sf = {idx: features[idx][1] for idx in feature_layers}

            fsm_weights = {'layer1': 0.5}
            cw_features = {'layer1': features_cw[4]}
            sf_features = {'layer1': features_sf[4]}

            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):

                fogpassfilter = FogPassFilter2
                fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()
                fogpassfilter_optimizer.zero_grad()

                sf_gram = [0] * args.batch_size
                cw_gram = [0] * args.batch_size

                fog_factor_sf = [0] * args.batch_size
                fog_factor_cw = [0] * args.batch_size

                for batch_idx in range(args.batch_size):
                    sf_gram[batch_idx] = gram_matrix(sf_features[layer][batch_idx])
                    cw_gram[batch_idx] = gram_matrix(cw_features[layer][batch_idx])

                    vector_sf_gram = sf_gram[batch_idx][torch.triu(torch.ones_like(sf_gram[batch_idx])) == 1]
                    vector_cw_gram = cw_gram[batch_idx][torch.triu(torch.ones_like(cw_gram[batch_idx])) == 1]

                    fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram)
                    fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram)

                    # For plotting
                    # safe squeeze -> ensure shape (D,)
                    emb_cw = fog_factor_cw[batch_idx].detach().cpu().squeeze()
                    emb_sf = fog_factor_sf[batch_idx].detach().cpu().squeeze()

                    # choose layer key string consistent with your cw_features keys: 'layer0' or 'layer1'
                    layer_key = 'layer1'
                    all_factors[layer_key]['CW'].append(emb_cw.numpy())
                    all_factors[layer_key]['SF'].append(emb_sf.numpy())

                emb_cw = torch.stack([e.squeeze(0) if e.dim() == 2 else e for e in fog_factor_cw], dim=0).to(
                    args.gpu)  # (B, D)
                emb_sf = torch.stack([e.squeeze(0) if e.dim() == 2 else e for e in fog_factor_sf], dim=0).to(
                    args.gpu)  # (B, D)

                fog_factor_embeddings = torch.cat([emb_cw, emb_sf], dim=0)  # (2B, D)

                B = emb_cw.size(0)
                fog_factor_labels = torch.LongTensor([0] * B + [1] * B).to(args.gpu)

                # Compute loss
                fog_pass_filter_loss = paired_loss_fn(emb_cw, emb_sf)
                total_fpf_loss += fog_pass_filter_loss

            total_fpf_loss.backward()

            if i_iter < 5000:
                FogPassFilter2_optimizer.step()

            if i_iter >= 5000:
                # Step 2: Train YOLO model
                model.train()
                for param in model.parameters():
                    param.requires_grad = True

                for param in FogPassFilter2.parameters():
                    param.requires_grad = False

                loss_det_cw, loss_det_sf, loss_con, loss_fsm = 0, 0, 0, 0
                for key in features:
                    features[key].clear()

                det_cw = model(cw_img)
                det_sf = model(sf_img)

                batch_cw = convert_labels_to_ultralytics_format(cw_label)
                loss_components, _ = yolo.loss(batch_cw, det_cw)
                loss_det_cw = loss_components.sum() / 10

                batch_sf = convert_labels_to_ultralytics_format(sf_label)
                loss_components, _ = yolo.loss(batch_sf, det_sf)
                loss_det_sf = loss_components.sum() / 10

                det_cw_processed = yolo(cw_img, verbose=False)
                det_sf_processed = yolo(sf_img, verbose=False)
                # Consistency loss with matched IoU
                if det_cw[0].numel() > 0 and det_sf[0].numel() > 0:
                    boxes_cw = torch.cat([det_cw_processed[0].boxes.xyxy, det_cw_processed[0].boxes.conf[:, None]],
                                         dim=1)
                    boxes_sf = torch.cat([det_sf_processed[0].boxes.xyxy, det_sf_processed[0].boxes.conf[:, None]],
                                         dim=1)
                    loss_con = 1 - compute_iou(boxes_cw, boxes_sf)  # Maximize IoU
                else:
                    loss_con = 0

                cw_features = {'layer1': features[4][0]}
                sf_features = {'layer1': features[4][1]}
                a_features, b_features = cw_features, sf_features

                for idx, layer in enumerate(fsm_weights):

                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                    fogpassfilter.eval()
                    layer_fsm_loss = 0

                    for batch_idx in range(args.batch_size):
                        a_gram = gram_matrix(a_features[layer][batch_idx])
                        b_gram = gram_matrix(b_features[layer][batch_idx])
                        _, _, ha, wa = a_features[layer].size()
                        _, _, hb, wb = b_features[layer].size()

                        vector_a = a_gram[torch.triu(torch.ones_like(a_gram)) == 1]
                        vector_b = b_gram[torch.triu(torch.ones_like(b_gram)) == 1]

                        fog_factor_a = fogpassfilter(vector_a)
                        fog_factor_b = fogpassfilter(vector_b)
                        half = int(fog_factor_b.shape[0] / 2)

                        layer_fsm_loss += fsm_weights[layer] * torch.mean(
                            (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half

                    loss_fsm += layer_fsm_loss / args.batch_size

                loss = (
                                   loss_det_sf + loss_det_cw + args.lambda_fsm * loss_fsm + args.lambda_con * loss_con) / args.iter_size
                if loss.requires_grad and loss != 0:
                    loss.backward()

                    if loss_det_cw != 0:
                        loss_det_cw_value += loss_det_cw.item() / args.iter_size
                    if loss_det_sf != 0:
                        loss_det_sf_value += loss_det_sf.item() / args.iter_size
                    if loss_fsm != 0:
                        loss_fsm_value += loss_fsm.item() / args.iter_size
                    if loss_con != 0:
                        loss_con_value += loss_con.item() / args.iter_size

                    for opt in opts:
                        opt.step()

                    FogPassFilter2_optimizer.step()
                    scheduler.step()
                else:
                    print(f"Skipped backward at iter {i_iter}, sub {sub_i}: Zero loss (empty batch?)")

                wandb.log({
                    "total_fpf_loss": total_fpf_loss,
                    "loss_det_cw": loss_det_cw_value,
                    "loss_det_sf": loss_det_sf_value,
                    "fsm_loss": args.lambda_fsm * loss_fsm_value,
                    "consistency_loss": args.lambda_con * loss_con_value,
                    "total_loss": loss
                }, step=i_iter)

                if i_iter % 500 == 0 and i_iter > 0:
                    metrics = test_model(args, model, yolo, FogPassFilter2)
                    print(f"Iter {i_iter} Metrics:", metrics)

        if i_iter % 1000 == 999:
            plot_cw_sf_layer(all_factors, out_file='layer1.png')
            all_factors = {
                'layer0': {'CW': [], 'SF': []},
                'layer1': {'CW': [], 'SF': []}
            }

    # Cleanup hooks
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()