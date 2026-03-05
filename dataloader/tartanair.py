import numpy as np
import os
import os.path as osp
from glob import glob

from dataloader.template import FlowDataset


class TartanAir(FlowDataset):
    def __init__(self, root='datasets/TartanAir'):
        super(TartanAir, self).__init__()
        self.root = root
        self._build_dataset()

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, '*/*/*/'))
        for scene in sorted(scenes):
            images = sorted(glob(osp.join(scene, 'image_left/*.png')))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_flow.npy"))
                self.mask_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_mask.npy"))

    def read_flow(self, index):
        flow = np.load(self.flow_list[index])
        valid = np.load(self.mask_list[index])
        valid = 1 - valid / 100
        return flow, valid
