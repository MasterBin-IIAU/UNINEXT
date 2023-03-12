import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class TNL2KDataset(BaseDataset):
    """
    TNL-2K test set consisting of 700 videos
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)

        frames_list = [os.path.join(frames_path, x) for x in sorted(os.listdir(frames_path))]

        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('datasets/data_specs/tnl2k.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list