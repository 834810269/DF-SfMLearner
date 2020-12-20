import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import numpy as np

def robust_rand_sample(match, mask, num):
    # match: [b, 4, -1] mask: [b, 1, -1]
    b, n = match.shape[0], match.shape[2]
    nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1))  # []
    if nonzeros_num.detach().cpu().numpy() == n:
        rand_int = torch.randint(0, n, [num])
        select_match = match[:, :, rand_int]
    else:
        # If there is zero score in match, sample the non-zero matches.
        num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
        select_idxs = []
        for i in range(b):
            nonzero_idx = torch.nonzero(mask[i, 0, :])  # [nonzero_num,1]
            rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
            select_idx = nonzero_idx[rand_int, :]  # [num, 1]
            select_idxs.append(select_idx)
        select_idxs = torch.stack(select_idxs, 0)  # [b,num,1]
        select_match = torch.gather(match.transpose(1, 2), index=select_idxs.repeat(1, 1, 4), dim=1).transpose(1,2)  # [b, 4, num]
    return select_match, num

def top_ratio_sample(match, mask, ratio):
    # match: [b, 4, -1] mask: [b, 1, -1]
    b, total_num = match.shape[0], match.shape[-1]
    scores, indices = torch.topk(mask, int(ratio * total_num), dim=-1)  # [B, 1, ratio*tnum]
    select_match = torch.gather(match.transpose(1, 2), index=indices.squeeze(1).unsqueeze(-1).repeat(1, 1, 4),
                                dim=1).transpose(1, 2)  # [b, 4, ratio*tnum]
    return select_match, scores

def rand_sample(match, num):
    b, c, n = match.shape[0], match.shape[1], match.shape[2]
    rand_int = torch.randint(0, match.shape[-1], size=[num])
    select_pts = match[:, :, rand_int]
    return select_pts

def filt_negative_depth(point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord):
    # Filter out the negative projection depth.
    # point2d_1_depth: [b, n, 1]
    b, n = point2d_1_depth.shape[0], point2d_1_depth.shape[1]
    mask = (point2d_1_depth > 0.01).float() * (point2d_2_depth > 0.01).float()
    select_idxs = []
    flag = 0
    for i in range(b):
        if torch.sum(mask[i, :, 0]) == n:
            idx = torch.arange(n).to(mask.get_device())
        else:
            nonzero_idx = torch.nonzero(mask[i, :, 0]).squeeze(1)  # [k]
            if nonzero_idx.shape[0] < 0.1 * n:
                idx = torch.arange(n).to(mask.get_device())
                flag = 1
            else:
                res = torch.randint(0, nonzero_idx.shape[0], size=[n - nonzero_idx.shape[0]]).to(
                    mask.get_device())  # [n-nz]
                idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
        select_idxs.append(idx)
    select_idxs = torch.stack(select_idxs, dim=0)  # [b,n]
    point2d_1_depth = torch.gather(point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1)  # [b,n,1]
    point2d_2_depth = torch.gather(point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1)  # [b,n,1]
    point2d_1_coord = torch.gather(point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2),
                                       dim=1)  # [b,n,2]
    point2d_2_coord = torch.gather(point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2),
                                       dim=1)  # [b,n,2]
    return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

def filt_invalid_coord(point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h, max_w):
    # Filter out the negative projection depth.
    # point2d_1_depth: [b, n, 1]
    b, n = point2d_1_coord.shape[0], point2d_1_coord.shape[1]
    max_coord = torch.Tensor([max_w, max_h]).to(point2d_1_coord.get_device())
    mask = (point2d_1_coord > 0).all(dim=-1, keepdim=True).float() * (point2d_2_coord > 0).all(dim=-1, keepdim=True).float() * \
               (point2d_1_coord < max_coord).all(dim=-1, keepdim=True).float() * (point2d_2_coord < max_coord).all(dim=-1, keepdim=True).float()

    flag = 0
    if torch.sum(1.0 - mask) == 0:
        return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

    select_idxs = []
    for i in range(b):
        if torch.sum(mask[i, :, 0]) == n:
            idx = torch.arange(n).to(mask.get_device())
        else:
            nonzero_idx = torch.nonzero(mask[i, :, 0]).squeeze(1)  # [k]
            if nonzero_idx.shape[0] < 0.1 * n:
                idx = torch.arange(n).to(mask.get_device())
                flag = 1
            else:
                res = torch.randint(0, nonzero_idx.shape[0], size=[n - nonzero_idx.shape[0]]).to(mask.get_device())
                idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
        select_idxs.append(idx)
    select_idxs = torch.stack(select_idxs, dim=0)  # [b,n]
    point2d_1_depth = torch.gather(point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1)  # [b,n,1]
    point2d_2_depth = torch.gather(point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1)  # [b,n,1]
    point2d_1_coord = torch.gather(point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2), dim=1)  # [b,n,2]
    point2d_2_coord = torch.gather(point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2), dim=1)  # [b,n,2]
    return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

def ray_angle_filter(match, P1, P2, return_angle=False):
    # match: [b, 4, n] P: [B, 3, 4]
    b, n = match.shape[0], match.shape[2]
    K = P1[:, :, :3]  # P1 with identity rotation and zero translation
    K_inv = torch.inverse(K)
    RT1 = K_inv.bmm(P1)  # [b, 3, 4]
    RT2 = K_inv.bmm(P2)
    ones = torch.ones([b, 1, n]).to(match.get_device())
    pts1 = torch.cat([match[:, :2, :], ones], 1)
    pts2 = torch.cat([match[:, 2:, :], ones], 1)

    ray1_dir = (RT1[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts1)  # [b,3,n]
    ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
    ray1_origin = (-1) * RT1[:, :, :3].transpose(1, 2).bmm(RT1[:, :, 3].unsqueeze(-1))  # [b, 3, 1]
    ray2_dir = (RT2[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts2)  # [b,3,n]
    ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
    ray2_origin = (-1) * RT2[:, :, :3].transpose(1, 2).bmm(RT2[:, :, 3].unsqueeze(-1))  # [b, 3, 1]

    # We compute the angle betwwen vertical line from ray1 origin to ray2 and ray1.
    p1p2 = (ray1_origin - ray2_origin).repeat(1, 1, n)
    verline = ray2_origin.repeat(1, 1, n) + torch.sum(p1p2 * ray2_dir, dim=1,
                                                          keepdim=True) * ray2_dir - ray1_origin.repeat(1, 1,
                                                                                                        n)  # [b,3,n]
    cosvalue = torch.sum(ray1_dir * verline, dim=1, keepdim=True) / \
                   ((torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12) * (
                               torch.norm(verline, dim=1, keepdim=True, p=2) + 1e-12))  # [b,1,n]

    mask = (cosvalue > 0.001).float()  # we drop out angles less than 1' [b,1,n]
    flag = 0
    num = torch.min(torch.sum(mask, -1)).int()
    if num.cpu().detach().numpy() == 0:
        flag = 1
        filt_match = match[:, :, :100]
        if return_angle:
            return filt_match, flag, torch.zeros_like(mask).to(filt_match.get_device())
        else:
            return filt_match, flag
    nonzero_idx = []
    for i in range(b):
        idx = torch.nonzero(mask[i, 0, :])[:num]  # [num,1]
        nonzero_idx.append(idx)
    nonzero_idx = torch.stack(nonzero_idx, 0)  # [b,num,1]
    filt_match = torch.gather(match.transpose(1, 2), index=nonzero_idx.repeat(1, 1, 4), dim=1).transpose(1,
                                                                                                             2)  # [b,4,num]
    if return_angle:
        return filt_match, flag, mask
    else:
        return filt_match, flag

def midpoint_triangulate(match, K_inv, P1, P2):
    # match: [b, 4, num] P1: [b, 3, 4]
    # Match is in the image coordinates. P1, P2 is camera parameters. [B, 3, 4] match: [B, 4, M]
    b, n = match.shape[0], match.shape[2]
    RT1 = K_inv.bmm(P1)  # [b, 3, 4]
    RT2 = K_inv.bmm(P2)
    ones = torch.ones([b, 1, n]).to(match.get_device())
    pts1 = torch.cat([match[:, :2, :], ones], 1) # [b,3,n]
    pts2 = torch.cat([match[:, 2:, :], ones], 1) # [b,3,n]

    ray1_dir = (RT1[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts1)  # [b,3,n]
    ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
    ray1_origin = (-1) * RT1[:, :, :3].transpose(1, 2).bmm(RT1[:, :, 3].unsqueeze(-1))  # [b, 3, 1]
    ray2_dir = (RT2[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts2)  # [b,3,n]
    ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
    ray2_origin = (-1) * RT2[:, :, :3].transpose(1, 2).bmm(RT2[:, :, 3].unsqueeze(-1))  # [b, 3, 1]

    dir_cross = torch.cross(ray1_dir, ray2_dir, dim=1)  # [b,3,n]
    denom = 1.0 / (torch.sum(dir_cross * dir_cross, dim=1, keepdim=True) + 1e-12)  # [b,1,n]
    origin_vec = (ray2_origin - ray1_origin).repeat(1, 1, n)  # [b,3,n]
    a1 = origin_vec.cross(ray2_dir, dim=1)  # [b,3,n]
    a1 = torch.sum(a1 * dir_cross, dim=1, keepdim=True) * denom  # [b,1,n]
    a2 = origin_vec.cross(ray1_dir, dim=1)  # [b,3,n]
    a2 = torch.sum(a2 * dir_cross, dim=1, keepdim=True) * denom  # [b,1,n]
    p1 = ray1_origin + a1 * ray1_dir
    p2 = ray2_origin + a2 * ray2_dir
    point = (p1 + p2) / 2.0  # [b,3,n]
    # Convert to homo coord to get consistent with other functions.
    point_homo = torch.cat([point, ones], dim=1).transpose(1, 2)  # [b,n,4] [x,y,z,1]
    return point_homo

def reproject(P, point3d):
    # P: [b,3,4] point3d: [b,n,4]
    point2d = P.bmm(point3d.transpose(1, 2))  # [b,4,n]
    point2d_coord = (point2d[:, :2, :] / (point2d[:, 2, :].unsqueeze(1) + 1e-12)).transpose(1, 2)  # [b,n,2]
    point2d_depth = point2d[:, 2, :].unsqueeze(1).transpose(1, 2)  # [b,n,1]
    return point2d_coord, point2d_depth

def scale_adapt(depth1, depth2, eps=1e-12):
    with torch.no_grad():
        A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1)  # [b,1]
        C = torch.sum(depth1 / (depth2 + eps), dim=1)  # [b,1]
        a = C / (A + eps)
    return a

def affine_adapt(depth1, depth2, use_translation=True, eps=1e-12):
    a_scale = scale_adapt(depth1, depth2, eps=eps)
    if not use_translation:  # only fit the scale parameter
        return a_scale, torch.zeros_like(a_scale)
    else:
        with torch.no_grad():
            A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1)  # [b,1]
            B = torch.sum(depth1 / (depth2 ** 2 + eps), dim=1)  # [b,1]
            C = torch.sum(depth1 / (depth2 + eps), dim=1)  # [b,1]
            D = torch.sum(1.0 / (depth2 ** 2 + eps), dim=1)  # [b,1]
            E = torch.sum(1.0 / (depth2 + eps), dim=1)  # [b,1]
            a = (B * E - D * C) / (B * B - A * D + 1e-12)
            b = (B * C - A * E) / (B * B - A * D + 1e-12)

            # check ill condition
            cond = (B * B - A * D)
            valid = (torch.abs(cond) > 1e-4).float()
            a = a * valid + a_scale * (1 - valid)
            b = b * valid
        return a, b

def register_depth(depth_pred, coord_tri, depth_tri):
    # depth_pred: [b, 1, h, w] coord_tri: [b,n,2] depth_tri: [b,n,1]
    batch, _, h, w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2], depth_pred.shape[3]
    n = depth_tri.shape[1]
    coord_tri_nor = torch.stack(
            [2.0 * coord_tri[:, :, 0] / (w - 1.0) - 1.0, 2.0 * coord_tri[:, :, 1] / (h - 1.0) - 1.0], -1)
    depth_inter = F.grid_sample(depth_pred, coord_tri_nor.view([batch, n, 1, 2]),
                                    padding_mode='reflection').squeeze(-1).transpose(1, 2)  # [b,n,1]

    # Normalize
    scale = torch.median(depth_inter, 1)[0] / (torch.median(depth_tri, 1)[0] + 1e-12)
    scale = scale.detach()  # [b,1]
    scale_depth_inter = depth_inter / (scale.unsqueeze(-1) + 1e-12)
    scale_depth_pred = depth_pred / (scale.unsqueeze(-1).unsqueeze(-1) + 1e-12)

    # affine adapt
    a, b = affine_adapt(scale_depth_inter, depth_tri, use_translation=False)
    affine_depth_inter = a.unsqueeze(1) * scale_depth_inter + b.unsqueeze(1)  # [b,n,1]
    affine_depth_pred = a.unsqueeze(-1).unsqueeze(-1) * scale_depth_pred + b.unsqueeze(-1).unsqueeze(-1)  # [b,1,h,w]
    return affine_depth_pred, affine_depth_inter

def triangulation_forward(K, pose, valid_mask, fwd_match, depth_sample_ratio, depth_match_num):
    # Get masks
    b = fwd_match.shape[0]
    img_h = valid_mask.size(2)
    img_w = valid_mask.size(3)

    K_inv = torch.inverse(K)
    top_ratio_match, top_ratio_mask = top_ratio_sample(fwd_match.view([b, 4, -1]), valid_mask.view([b, 1, -1]), ratio=depth_sample_ratio)  # [b, 4, ratio*h*w]
    depth_match, depth_match_num = robust_rand_sample(top_ratio_match, top_ratio_mask, num=depth_match_num)

    iden = torch.cat([torch.eye(3), torch.zeros([3, 1])], -1).unsqueeze(0).repeat(b, 1, 1).to(K.get_device())  # [b,3,4]
    P1 = K @ iden
    P1 = P1.detach()
    P2 = K @ pose
    P2 = P2.detach()

    # Get triangulated points
    filt_depth_match, flag1 = ray_angle_filter(depth_match, P1, P2, return_angle=False)  # [b, 4, filt_num]

    point3d_1 = midpoint_triangulate(filt_depth_match, K_inv, P1, P2)
    point2d_1_coord, point2d_1_depth = reproject(P1, point3d_1)  # [b,n,2], [b,n,1]
    point2d_2_coord, point2d_2_depth = reproject(P2, point3d_1)

    # Filter out some invalid triangulation results to stablize training.
    point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag2 = filt_negative_depth(point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord)
    point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag3 = filt_invalid_coord(point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h=img_h, max_w=img_w)

    return point2d_1_coord, point2d_1_depth, point2d_2_coord, point2d_2_depth, flag1+flag2+flag3

