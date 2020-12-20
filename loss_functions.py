from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp2, compute_dynamic_mask, get_consistent_mask, warp_flow, get_flow_norm
from triangulation import triangulation_forward, register_depth
import math
from img_show import show_flow, img_show_singleimage, img_show_singleimage_numpy
import matplotlib.pyplot as plt
from utils import tensor2array

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

compute_ssim_loss = SSIM().to(device)

# photometric loss
# geometry consistency loss
# cross consistency loss
# triangulation loss
def compute_losses(tgt_img, ref_imgs, intrinsics,
                   tgt_depth, ref_depths,
                   flows_tgt2ref, flows_ref2tgt,
                   poses_tgt2ref, poses_ref2tgt, F_tgt2ref, F_ref2tgt,
                   with_ssim, with_mask, with_triangulation,
                   max_scales, padding_mode,log_output):
    log_output = False

    photo_loss = 0
    geometry_loss = 0
    cross_loss = 0
    triangulation_loss = 0
    num_scales = min(min(len(tgt_depth), max_scales),len(flows_tgt2ref[0]))
    index = 0 
    scaled_tgt_depth = []
    scaled_ref_depth = []
    for ref_img, ref_depth, flow_t2r, flow_r2t in zip(ref_imgs, ref_depths, flows_tgt2ref, flows_ref2tgt):
        pose = poses_tgt2ref[:,index,:,:].detach()
        # print(pose[:,:,-1].norm(dim=1))
        pose_inv = poses_ref2tgt[:,index,:,:].detach()
        F_t2r = F_tgt2ref[:,index,:,:].detach()
        F_r2t = F_ref2tgt[:,index,:,:].detach()
        # print(pose[0],pose_inv[0],F_t2r[0],F_r2t[0],intrinsics[0])
        index += 1
        for s in range(num_scales):
            # upsample depth
            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics
            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
                flow_t2r_scaled = flow_t2r[s]
                flow_r2t_scaled = flow_r2t[s]
            else:
                tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode='bilinear',align_corners=False)
                ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode='bilinear',align_corners=False)
                x_scale = w/flow_t2r[s].size(3)
                y_scale = h/flow_t2r[s].size(2)
                scaled = torch.tensor([x_scale,y_scale]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float().to(device).detach()
                flow_t2r_scaled =  scaled*F.interpolate(flow_t2r[s], (h,w), mode='bilinear',align_corners=False) # todo: the coefficient
                flow_r2t_scaled =  scaled*F.interpolate(flow_r2t[s], (h,w), mode='bilinear',align_corners=False)

            if with_mask:
                forward_mask = compute_dynamic_mask(F_t2r, flow_t2r_scaled).detach()
                backward_mask = compute_dynamic_mask(F_r2t, flow_r2t_scaled).detach()
            else:
                forward_mask = torch.ones_like(tgt_depth_scaled).detach()
                backward_mask = torch.ones_like(tgt_depth_scaled).detach()
            # print(forward_mask.sum(dim=(1,2,3)))
            fwd_consistent_mask, bwd_consistent_mask, fwd_flow_diff, bwd_flow_diff, fwd_matches, mask_f, mask_b = get_consistent_mask(
                flow_t2r_scaled, flow_r2t_scaled)
            #print("==========================================")
            fwd_consistent_mask = fwd_consistent_mask.detach()
            fwd_matches = fwd_matches.detach()
            mask_f = mask_f.detach()
            #print(tgt_depth_scaled[0])
            if with_triangulation:
                # todo: forward_backward mask
                fwd_valid_mask =  forward_mask.float()*fwd_consistent_mask.float()*mask_f.float()
                #if(log_output):
                #    img_show_singleimage(forward_mask[0]*255)
                #    img_show_singleimage(fwd_consistent_mask[0]*255)
                #    img_show_singleimage(fwd_valid_mask[0]*255)
                #    plt.show()

                point2d_1_coord, point2d_1_depth, point2d_2_coord, point2d_2_depth, flag = triangulation_forward(intrinsics, pose, fwd_valid_mask, fwd_matches, 0.2, 6000)

                if log_output:
                    y_coord = point2d_1_coord[0,:,0].type(torch.long)
                    x_coord = point2d_1_coord[0,:,1].type(torch.long)
                    empty_depth = torch.zeros(tgt_img.size(2),tgt_img.size(3)).to(device)
                    empty_depth[x_coord,y_coord]=point2d_1_depth[0].squeeze(-1)
                    tri_map = tensor2array(empty_depth.squeeze(0),(empty_depth.sum()/float(y_coord.size(0))).item(),colormap='magma').transpose(1,2,0)
                    img_show_singleimage_numpy(tri_map[:,:,:-1])
                #    plt.show()

                if flag == 0:
                    tgt_depth_scaled, inter_pred1 = register_depth(tgt_depth_scaled, point2d_1_coord, point2d_1_depth)
                    ref_depth_scaled, inter_pred2 = register_depth(ref_depth_scaled, point2d_2_coord, point2d_2_depth)
                # Get Losses
                    triangulation_loss += get_trian_loss(point2d_1_depth, inter_pred1) + get_trian_loss(point2d_2_depth, inter_pred2)
                else:
                    return torch.zeros(4).to(device).requires_grad_()
            #print(tgt_depth_scaled[0])

            photo_loss1, geometry_loss1, cross_loss1, forward_mask = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled,
                                                                                           pose, flow_t2r_scaled, intrinsic_scaled, forward_mask, with_ssim, padding_mode,(log_output and s==0))
            photo_loss2, geometry_loss2, cross_loss2, backward_mask = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled,
                                                                                            pose_inv, flow_r2t_scaled, intrinsic_scaled, backward_mask, with_ssim, padding_mode,(log_output and s==0))

            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)
            cross_loss += (cross_loss1 + cross_loss2)
            # print(triangulation_loss)

    return photo_loss, geometry_loss, cross_loss, triangulation_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, flow, intrinsic, weight_mask, with_ssim, padding_mode,log_output):

    ref_img_warped, valid_mask, projected_depth, computed_depth, rigid_flow = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs()
    
    
    #print(rigid_flow[0])
    #print(flow[0])
    #print("================================================================")
    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth).abs()).clamp(0, 1)

    diff_flow = (rigid_flow - flow).abs()*valid_mask

    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if log_output:
        img_show_singleimage(tgt_img[0])
        img_show_singleimage(ref_img_warped[0])
        img_show_singleimage(diff_img[0])
        #img_show_singleimage(ssim_map[0])
        #print(ssim_map[0])
        de = tensor2array(tgt_depth[0]).transpose(1,2,0)
        img_show_singleimage_numpy(de[:,:,:-1])
    #    show_flow(rigid_flow[0])
        plt.show()
    #print(tgt_depth[0])
    # compute all loss
    diff_img = weight_mask * diff_img
    diff_depth = weight_mask * diff_depth
    diff_flow = weight_mask * diff_flow

    # comppute loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
    diff_flow_norm = torch.sqrt(get_flow_norm(diff_flow)+ 1e-8)
    cross_loss = mean_on_mask(diff_flow_norm, valid_mask)
    #print(cross_loss)

    return reconstruction_loss, geometry_consistency_loss, cross_loss, weight_mask*valid_mask

# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / (mask.sum().clamp(min=1.0))
    return mean_value


def get_trian_loss(tri_depth, pred_tri_depth):
    # depth: [b,n,1]
    loss = torch.pow(1.0 - pred_tri_depth / (tri_depth + 1e-12), 2).mean()
    return loss

def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def get_edge_smoothness(img, pred):
        img_dx, img_dy = gradient(img)
        disp_dx, disp_dy = gradient(pred)
        disp_dx2, disp_dxdy = gradient(disp_dx)
        disp_dydx, disp_dy2 = gradient(disp_dy)

        weights_x = torch.exp(-torch.mean(torch.abs(img_dx),1, keepdim=True)) #[:,:,:,1:]
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy),1, keepdim=True)) #[:,:,1:]

        smoothness_x = torch.abs(disp_dx) * weights_x
        smoothness_y = torch.abs(disp_dy) * weights_y

        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = F.interpolate(img, (h, w),mode='area')
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 2.3

    return loss

def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs, max_scales=1):
    loss = edge_aware_smoothness_loss(tgt_depth, tgt_img, max_scales)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += edge_aware_smoothness_loss(ref_depth, ref_img, max_scales)

    return loss



@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

