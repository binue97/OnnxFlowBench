import os
import os.path as osp
from glob import glob

from utils import frame_utils
from dataloader.template import FlowDataset


class HD1K(FlowDataset):
    def __init__(self, root="datasets/HD1K"):
        super(HD1K, self).__init__(sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(osp.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(osp.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1

    def read_flow(self, index):
        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        return flow, valid
