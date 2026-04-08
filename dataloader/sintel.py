import numpy as np

import os
import os.path as osp
from glob import glob

from utils import frame_utils
from dataloader.template import FlowDataset
from config import get_dataset_root


class MpiSintel(FlowDataset):
    def __init__(self, split="training", root=None, dstype="clean"):
        root = root or get_dataset_root("sintel", "datasets/Sintel")
        super(MpiSintel, self).__init__()
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

    def read_flow(self, index):
        flow = frame_utils.read_gen(self.flow_list[index])
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid
