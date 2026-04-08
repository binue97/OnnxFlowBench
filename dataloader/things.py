import numpy as np

from glob import glob
import os.path as osp

from utils import frame_utils
from dataloader.template import FlowDataset
from config import get_dataset_root


class FlyingThings(FlowDataset):
    def __init__(self, root=None, dstype="frames_cleanpass"):
        root = root or get_dataset_root("things", "datasets/FlyingThings")
        super(FlyingThings, self).__init__()
        for cam in ["left"]:
            for direction in ["into_future", "into_past"]:
                image_dirs = sorted(glob(osp.join(root, dstype, "TRAIN/*/*")))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
                flow_dirs = sorted(glob(osp.join(root, "optical_flow/TRAIN/*/*")))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, "*.png")))
                    flows = sorted(glob(osp.join(fdir, "*.pfm")))
                    for i in range(len(flows) - 1):
                        if direction == "into_future":
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == "into_past":
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]

    def read_flow(self, index):
        flow = frame_utils.read_gen(self.flow_list[index])
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid
