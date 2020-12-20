from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
pixel_coords = None
from img_show import img_show_singleimage,show_flow
import matplotlib.pyplot as plt


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)

def mycam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics, padding_mode):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]

    if proj_c2p_rot is not None:
        coords = proj_c2p_rot @ cam_coords_flat
    else:
        coords = cam_coords_flat
    if proj_c2p_tr is not None:
        coords = coords + proj_c2p_tr  # [B, 3, H*W]


    depth = coords[:,2].clamp(min=1e-3) # [b,h*w]
    depth = depth.unsqueeze(1) # [b,1,h*w]
    coords_one_scale = coords/depth # [B, 3, h*w]
    pcoords = intrinsics @ coords_one_scale

    #pcoords_mask = (pcoords[:,2] == 1).reshape(h,w).unsqueeze(0).unsqueeze(0)

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    X_norm = 2*X/(w-1) - 1
    Y_norm = 2*Y/(h-1) - 1
    if padding_mode == 'zeros':
        X_mask = ((X_norm>1)+(X_norm<-1)).detach()
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm<-1)+(Y_norm>1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)

    return pixel_coords.reshape(b,h,w,2), depth.reshape(b,1,h,w), pcoords.reshape(b, 3, h, w)

def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    rot, tr = pose[:,:,:3], pose[:,:,-1:]

    src_pixel_coords, computed_depth, rigid_pixels = mycam2pixel(cam_coords, rot, tr, intrinsics, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()
    rigid_flow = valid_mask*(rigid_pixels - pixel_coords)[:,:2]
    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode).clamp(min=1e-3)
    
    #img_show_singleimage(img[0])
    #img_show_singleimage(projected_img[0])
    #plt.show()

    return projected_img, valid_mask, projected_depth, computed_depth, rigid_flow

def compute_dynamic_mask(F_matrix, flow):
    """
    Generate a mask using pose and optical flow matches.
    Args:
        F_matrix: fundamental matrix from target to source -- [B, 3, 3]
        flow: optical flow from target to source -- [B, 2, h, w]
    Returns:
        valid_mask: dynamic_mask -- [B, 1, h, 2]
    """
    global pixel_coords
    batch_size, _, img_height, img_width = flow.size() # [b,2,h,w]
    if (pixel_coords is None) or pixel_coords.size(2) < img_height:
        set_id_grid(flow[:, 0])
    pixel_coords_uv = pixel_coords[:,:2] # [b, 2, h, w]
    pixel_coords_pred = (pixel_coords_uv + flow).reshape(batch_size,2,-1) # [b,2,h*w]
    
    # edge validation
    X = pixel_coords_pred[:, 0]  # TODO:[B,H*W]
    Y = pixel_coords_pred[:, 1]  # TODO:[B,H*W]
    X_norm = 2 * X / (img_width - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (img_height - 1) - 1  # Idem [B, H*W]
    X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
    X_norm[X_mask] = 2
    Y_mask = ((Y_norm < -1) + (Y_norm > 1)).detach()
    Y_norm[Y_mask] = 2
    mask= (X_mask * Y_mask).reshape(batch_size, img_height, img_width).unsqueeze(1).float()
    mask = torch.ones_like(mask)-mask

    pre_uv = pixel_coords_pred.transpose(1,2) # [b,h*w,2]
    ones = torch.ones(batch_size,img_height*img_width,1).type_as(pixel_coords_pred) # [b,h*w,1]
    pre_pixels = torch.cat([pre_uv, ones], dim=2) # [b,h*w,3]

    #print(pixel_coords.size())
    cur_pixels = pixel_coords.expand([batch_size, 3, img_height, img_width]).reshape(batch_size,3,-1).transpose(1,2) # [b, h*w, 3]
    epipolar_line = F_matrix.unsqueeze(1) @ cur_pixels.unsqueeze(-1) # [b,h*w,3,1]
    a = epipolar_line[:,:,0,:]
    b = epipolar_line[:,:,1,:]
    dist_div = torch.sqrt( a*a + b*b ) + 1e-6 # [b, h*w, 1]
    # TODO:极线约束 [B,H*W,1,3]@[B,H*W,3,1]
    epipolar_mask = ( pre_pixels.unsqueeze(2) @ epipolar_line).squeeze(-1)
    dist_map = (epipolar_mask.abs()/ dist_div) # [b, h*w, 1]
    #epipolar_mask = (dist_map.reshape(batch_size, -1, img_height, img_width)  < 1).float()    # print(dist_map.reshape(batch_size, -1, img_height, img_width))
    epipolar_mask = torch.exp(-0.1*dist_map.abs().squeeze(-1).unsqueeze(1).reshape(batch_size, -1, img_height, img_width))
    #print(epipolar_mask[0])
    #show_flow(flow[0])
    # img_show_singleimage(epipolar_mask[0]*255)

    return epipolar_mask


def warp_flow(x, flow, padding_mode = 'zeros'):
    global pixel_coords
    batch_size, _, img_height, img_width = x.size()
    if (pixel_coords is None) or pixel_coords.size(2) < img_height:
        set_id_grid(x[:,0])

    pixel_coords_uv = pixel_coords[:,:2].expand([batch_size,2,img_height,img_width]) # [1,2,h,w]
    pixel_coords_src = (pixel_coords_uv + flow).reshape(batch_size,2,-1)
    X = pixel_coords_src[:, 0]  # TODO:[B,H*W]
    Y = pixel_coords_src[:, 1]  # TODO:[B,H*W]
    X_norm = 2 * X / (img_width - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * Y / (img_height - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm < -1) + (Y_norm > 1)).detach()
        Y_norm[Y_mask] = 2
    pixel_coords_flow = torch.stack([X_norm, Y_norm], dim=2)
    pixel_coords_flow = pixel_coords_flow.reshape(batch_size, img_height, img_width, 2)# [B, H*W, 2]
    # projected_img = F.grid_sample(img, pixel_coords_flow, padding_mode=padding_mode)
    valid_points = pixel_coords_flow.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()
    projected_flow_backward = F.grid_sample(x, pixel_coords_flow, padding_mode=padding_mode)
    matches = torch.cat([pixel_coords_uv, pixel_coords_src.reshape(batch_size,2,img_height,img_width)],1) # [b,4,h,w]
    return projected_flow_backward, matches, valid_mask

def get_consistent_mask(flow_forward, flow_backward, beta=0.05, alpha=3):
    batch_size, _, img_height, img_width = flow_forward.size()
    #print(flow_forward)
    #print(flow_backward)
    #show_flow(flow_forward[0])
    #show_flow(flow_backward[0])
    #plt.show()


    bwd2fwd_flow, fwd_matches, mask_f = warp_flow(flow_backward, flow_forward)  # -
    fwd2bwd_flow, bwd_matches, mask_b = warp_flow(flow_forward, flow_backward)  # +

    fwd_flow_diff = torch.abs(bwd2fwd_flow + flow_forward)
    bwd_flow_diff = torch.abs(fwd2bwd_flow + flow_backward)

    #print(fwd_flow_diff)
    #print(bwd_flow_diff)
    # flow consistency condition
    bwd_consist_bound = torch.max(beta * get_flow_norm(flow_backward),torch.from_numpy(np.array([alpha])).float().to(flow_backward.get_device()))
    fwd_consist_bound = torch.max(beta * get_flow_norm(flow_forward), torch.from_numpy(np.array([alpha])).float().to(flow_forward.get_device()))

    noc_masks_img2 = mask_b*(get_flow_norm(bwd_flow_diff) < bwd_consist_bound).float()
    noc_masks_img1 = mask_f*(get_flow_norm(fwd_flow_diff) < fwd_consist_bound).float()

    #print(mask_f,mask_b)
    #img_show_singleimage(mask_f[0]*255)
    #plt.show()
    return noc_masks_img1, noc_masks_img2, fwd_flow_diff, bwd_flow_diff, fwd_matches, mask_f, mask_b # mask_f,mask_b 表前向/后向采样有效点掩码

def get_flow_norm(flow, p=2):
    flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
    return flow_norm