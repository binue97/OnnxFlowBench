import numpy as np

from glob import glob
import os.path as osp

from utils import frame_utils
from dataloader.template import FlowDataset
from config import get_dataset_root


class FlyingChairs(FlowDataset):
    def __init__(self, split="training", root=None):
        root = root or get_dataset_root("chairs", "datasets/FlyingChairs/")
        super(FlyingChairs, self).__init__()

        images = sorted(glob(osp.join(root, "FlyingChairs_release/data", "*.ppm")))
        flows = sorted(glob(osp.join(root, "FlyingChairs_release/data", "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt(osp.join(root, "chairs_split.txt"), dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]

    def read_flow(self, index):
        flow = frame_utils.read_gen(self.flow_list[index])
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid
