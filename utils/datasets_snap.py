"""SnapMoGen datasets used by direct motion-generation models."""

import json
import random
from os.path import join as pjoin

import numpy as np
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def snap_collate_fn(batch):
    batch.sort(key=lambda item: item[2], reverse=True)
    return default_collate(batch)


class SnapText2MotionDataset(data.Dataset):
    """Load the segmented 296-D motions and JSON captions from SnapMoGen."""

    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,
        text_file,
        unit_length,
        max_motion_length,
        min_motion_length=128,
    ):
        self.mean = mean
        self.std = std
        self.unit_length = int(unit_length)
        self.max_motion_length = int(max_motion_length)
        self.min_motion_length = int(min_motion_length)

        with open(split_file, encoding="utf-8") as split_handle:
            segment_ids = [line.strip() for line in split_handle if line.strip()]
        with open(text_file, encoding="utf-8") as text_handle:
            captions = json.load(text_handle)

        self.data_dict = {}
        self.name_list = []
        for segment_id in tqdm(segment_ids):
            motion_name, start_frame, end_frame = segment_id.split("#")
            motion = np.load(pjoin(motion_dir, f"{motion_name}.npy"))
            motion = motion[int(start_frame) : int(end_frame)]
            if len(motion) < self.min_motion_length:
                continue
            if np.isnan(motion).any() or np.isinf(motion).any():
                continue

            text_data = captions[segment_id]["manual"] + captions[segment_id]["gpt"]
            self.data_dict[segment_id] = {
                "motion": np.expand_dims(motion, axis=1),
                "text": text_data,
            }
            self.name_list.append(segment_id)

        print(f"Loaded {len(self.name_list)} SnapMoGen motion-caption segments")

    def transform(self, motion, mean=None, std=None):
        target_mean = self.mean if mean is None else mean
        target_std = self.std if std is None else std
        return (motion - target_mean) / target_std

    def inv_transform(self, motion, mean=None, std=None):
        target_mean = self.mean if mean is None else mean
        target_std = self.std if std is None else std
        return motion * target_std + target_mean

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        data = self.data_dict[self.name_list[index]]
        motion = data["motion"]
        caption = random.choice(data["text"])

        motion_length = min(len(motion), self.max_motion_length)
        motion_length = (motion_length // self.unit_length) * self.unit_length
        start = random.randint(0, len(motion) - motion_length)
        motion = motion[start : start + motion_length]
        motion = self.transform(motion)

        if motion_length < self.max_motion_length:
            padding = np.zeros(
                (
                    self.max_motion_length - motion_length,
                    motion.shape[1],
                    motion.shape[2],
                ),
                dtype=motion.dtype,
            )
            motion = np.concatenate((motion, padding), axis=0)

        return caption, motion, motion_length
