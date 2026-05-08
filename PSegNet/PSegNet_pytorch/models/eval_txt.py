"""
Evaluate PSegNet on a single txt point cloud file.

Input txt format: x y z instance_label (4 columns, space-separated)
where instance_label = 0 for stem, >0 for individual leaf instances.

No h5 conversion needed — data is loaded directly into memory.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PSEGNET_DIR = os.path.dirname(BASE_DIR)  # PSegNet_pytorch/
sys.path.append(BASE_DIR)
sys.path.append(PSEGNET_DIR)

from model_pytorch import plantnet_model
from utils.clustering import cluster
from utils.pointnet2_util_pytorch import farthest_point_sample, index_points


def parse_args():
    parser = argparse.ArgumentParser('PSegNet txt eval')
    parser.add_argument('--input', type=str, required=True, help='Path to txt file (x y z instance_label)')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_epoch199.pth')
    parser.add_argument('--num_point', type=int, default=4096)
    parser.add_argument('--bandwidth', type=float, default=0.6)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='out')
    parser.add_argument('--propagate_k', type=int, default=1,
                        help='If >0, propagate labels to all original points via KNN (k neighbors). 0 = disabled.')
    return parser.parse_args()


def normalize(xyz):
    """Center and scale to [-1, 1] range. Mirrors 003TXT2H5.py."""
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    xyz = xyz - (xyz_min + (xyz_max - xyz_min) / 2)
    scale = np.max(np.abs(xyz))
    return xyz / scale


def fps_subsample(xyz, gt_ins, npoint, device):
    """FPS downsample xyz and carry the GT instance labels along. Returns indices too."""
    xyz_t = torch.from_numpy(xyz).float().unsqueeze(0).to(device)  # (1, N, 3)
    idx = farthest_point_sample(npoint, xyz_t)                      # (1, npoint)
    idx_np = idx.squeeze(0).cpu().numpy()                           # (npoint,)
    xyz_fps = index_points(xyz_t, idx).squeeze(0).cpu().numpy()     # (npoint, 3)
    gt_fps = gt_ins[idx_np]                                         # (npoint,)
    return xyz_fps, gt_fps, idx_np


def propagate_labels(xyz_all, xyz_sampled, labels_sampled, k):
    """KNN label propagation from sampled points to all original points."""
    tree = cKDTree(xyz_sampled)
    _, nn_idx = tree.query(xyz_all, k=k)  # (N,) if k=1 else (N, k)
    if k == 1:
        return labels_sampled[nn_idx]
    neighbor_labels = labels_sampled[nn_idx]  # (N, k)
    return stats.mode(neighbor_labels, axis=1, keepdims=False)[0].astype(int)


def compute_instance_metrics(pred_ins, gt_ins, iou_thresh=0.5):
    """Class-agnostic instance segmentation metrics."""
    pred_ids = np.unique(pred_ins)
    gt_ids = np.unique(gt_ins)

    # Build boolean masks
    pred_masks = [pred_ins == g for g in pred_ids if g != -1]
    gt_masks = [gt_ins == g for g in gt_ids]

    # MUCov and MWCov
    sum_cov, sum_weighted_cov, total_gt_pts = 0.0, 0.0, 0
    for ins_gt in gt_masks:
        n_gt = np.sum(ins_gt)
        total_gt_pts += n_gt
        best_iou = 0.0
        for ins_pred in pred_masks:
            iou = np.sum(ins_pred & ins_gt) / np.sum(ins_pred | ins_gt)
            if iou > best_iou:
                best_iou = iou
        sum_cov += best_iou
        sum_weighted_cov += best_iou * n_gt

    mucov = sum_cov / len(gt_masks) if gt_masks else 0.0
    mwcov = sum_weighted_cov / total_gt_pts if total_gt_pts > 0 else 0.0

    # Precision and Recall @ iou_thresh
    tp = fp = 0
    for ins_pred in pred_masks:
        best_iou = max(
            (np.sum(ins_pred & ins_gt) / np.sum(ins_pred | ins_gt) for ins_gt in gt_masks),
            default=0.0,
        )
        if best_iou >= iou_thresh:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(gt_masks) if gt_masks else 0.0

    return mucov, mwcov, precision, recall


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- Load txt file ---
    data = np.loadtxt(args.input)
    xyz = data[:, :3]
    gt_ins = data[:, 3].astype(int)

    print(f'Loaded {xyz.shape[0]} points from {args.input}')

    # --- Normalize (model input only) ---
    xyz_orig = xyz.copy()
    xyz = normalize(xyz)

    # --- FPS to num_point ---
    if xyz.shape[0] < args.num_point:
        raise ValueError(f'File has {xyz.shape[0]} points, need at least {args.num_point}')
    xyz_fps, gt_fps, fps_idx = fps_subsample(xyz, gt_ins, args.num_point, device)
    xyz_orig_fps = xyz_orig[fps_idx]  # original-scale coords for the same sampled points
    print(f'FPS -> {xyz_fps.shape[0]} points')

    # --- Load model ---
    NUM_CLASSES = 6
    model = plantnet_model(NUM_CLASSES).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Model loaded.')

    mean_ins_size = np.loadtxt(os.path.join(BASE_DIR, 'mean_ins_size.txt'))

    # --- Inference ---
    pts_t = torch.from_numpy(xyz_fps).float().unsqueeze(0).to(device)  # (1, 4096, 3)
    with torch.no_grad():
        pred_sem_logits, pred_ins_emb, _ = model(pts_t)

    pred_sem = torch.argmax(F.softmax(pred_sem_logits, dim=2), dim=2)
    pred_sem = pred_sem.squeeze(0).cpu().numpy()          # (4096,)
    pred_ins_emb = pred_ins_emb.squeeze(0).cpu().numpy()  # (4096, 5)

    # --- Clustering ---
    num_clusters, cluster_labels, _ = cluster(pred_ins_emb, args.bandwidth)
    print(f'Clustering found {num_clusters} raw clusters.')

    # Filter small clusters (mirrors 02test.py:140-149)
    pred_ins_final = -1 * np.ones_like(cluster_labels)
    counter = 0
    for g in np.unique(cluster_labels):
        if g == -1:
            continue
        mask = cluster_labels == g
        sem_g = int(stats.mode(pred_sem[mask], keepdims=False)[0])
        if np.sum(mask) > 0.01 * mean_ins_size[sem_g]:
            pred_ins_final[mask] = counter
            counter += 1

    print(f'After size filtering: {counter} instances.')

    # --- Metrics ---
    mucov, mwcov, precision, recall = compute_instance_metrics(pred_ins_final, gt_fps)
    print(f'\n=== Instance Segmentation Metrics ===')
    print(f'MUCov:        {mucov:.4f}')
    print(f'MWCov:        {mwcov:.4f}')
    print(f'Precision@0.5: {precision:.4f}')
    print(f'Recall@0.5:    {recall:.4f}')

    # --- Save visual output ---
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output_dir, f'{stem}_pred.txt')

    if args.propagate_k > 0:
        print(f'\nPropagating labels to {xyz_orig.shape[0]} points (k={args.propagate_k})...')
        pred_ins_full = propagate_labels(xyz_orig, xyz_orig_fps, pred_ins_final, k=args.propagate_k)
        out_data = np.column_stack([xyz_orig, pred_ins_full, gt_ins])
    else:
        out_data = np.column_stack([xyz_orig_fps, pred_ins_final, gt_fps])

    np.savetxt(out_path, out_data, fmt='%f %f %f %d %d',
               header='x y z pred_instance gt_instance')
    print(f'Visual output saved to: {out_path}')


if __name__ == '__main__':
    main()
