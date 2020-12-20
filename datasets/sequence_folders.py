import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            poses = np.genfromtxt(scene/'poses.txt').astype(np.float32) # TOOD: [N,12]
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:
                continue
            if poses.shape[0] != len(imgs):
                print("the number of the image and the number of poses is not equal")   
                continue
                
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 
                          'pose_tgt2ref': [], 'pose_ref2tgt': [], 'essential_tgt2ref': [], 'essential_ref2tgt': []} # todo: add ground truth poses and essential matrix
                pose_tgt = poses[i].reshape(3, 4)
                R_tgt = pose_tgt[:, :3]
                R_tgt_inv = np.linalg.inv(R_tgt)
                t_tgt = pose_tgt[:, -1].reshape(3, 1)
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    pose_ref = poses[i + j].reshape(3, 4)
                    R_ref = pose_ref[:, :3]
                    R_ref_inv = np.linalg.inv(R_ref)
                    t_ref = pose_ref[:, -1].reshape(3, 1)
                    R_ref2tgt = np.dot(R_tgt_inv, R_ref)
                    t_ref2tgt = np.dot(R_tgt_inv, (t_ref - t_tgt))
                    essential_ref2tgt = np.cross(t_ref2tgt.T, R_ref2tgt.T).T
                    pose_ref2tgt = np.hstack((R_ref2tgt, t_ref2tgt))
                    R_tgt2ref = np.dot(R_ref_inv, R_tgt)
                    t_tgt2ref = np.dot(R_ref_inv, (t_tgt - t_ref))
                    essential_tgt2ref = np.cross(t_tgt2ref.T, R_tgt2ref.T).T
                    pose_tgt2ref = np.hstack((R_tgt2ref, t_tgt2ref))
                    sample['pose_ref2tgt'].append(pose_ref2tgt)
                    sample['pose_tgt2ref'].append(pose_tgt2ref)
                    sample['essential_ref2tgt'].append(essential_ref2tgt)
                    sample['essential_tgt2ref'].append(essential_tgt2ref)
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        pose_ref2tgt = np.array([pose for pose in np.copy(sample['pose_ref2tgt'])])  # TODO: (ref_num×3*4)
        pose_tgt2ref = np.array([pose for pose in np.copy(sample['pose_tgt2ref'])])  # TODO: (ref_num×3*4)

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        intrinsics_inv = np.linalg.inv(intrinsics)

        fundamental_ref2tgt = np.array([intrinsics_inv.T.dot(e).dot(intrinsics_inv) for e in np.copy(sample['essential_ref2tgt'])])
        fundamental_tgt2ref = np.array([intrinsics_inv.T.dot(e).dot(intrinsics_inv) for e in np.copy(sample['essential_tgt2ref'])])

        return tgt_img, ref_imgs, intrinsics, intrinsics_inv, pose_tgt2ref, pose_ref2tgt, fundamental_tgt2ref, fundamental_ref2tgt

    def __len__(self):
        return len(self.samples)
