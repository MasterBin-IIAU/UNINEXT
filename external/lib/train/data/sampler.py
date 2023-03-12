import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_sample_frames, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_sample_frames = num_sample_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.sample(valid_ids, num_ids) # without replacement

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                frame_ids = self._sample_visible_ids(visible, num_ids=self.num_sample_frames, min_id=0, max_id=len(visible))
                frame_ids.sort()
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                frame_ids = [1] * self.num_sample_frames
            try:
                frames, anno, meta_obj = dataset.get_frames(seq_id, frame_ids, seq_info_dict)

                H, W, _ = frames[0].shape
                masks = anno['mask'] if 'mask' in anno else [torch.zeros((H, W))] * self.num_sample_frames

                data = TensorDict({'search_images': frames,
                                    'search_anno': anno['bbox'],
                                    'search_masks': masks,
                                    'dataset': dataset.get_name()})
                # make data augmentation
                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (self.num_sample_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict