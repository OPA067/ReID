"""
RSTPReid dataset loader.

This dataset represents a person ReID benchmark where each data sample
consists of three elements:
    - id:          Unique person identity label.
    - tar_path:    Path to the target (query) image.
    - can_path:    Path to the candidate (gallery) image.

The dataset assumes a JSON annotation format:
    [
        {"id": 0,  "tar_path": ".../person_001_a.jpg",  "can_path": ".../person_001_b.jpg"},
        ...
    ]
"""

import logging
import os
import os.path as op
from typing import List
from utils.iotools import read_json


class RSTPReid:
    """
    RSTPReid dataset for image-based person re-identification.

    Attributes:
        dataset_dir (str): Relative directory name for the dataset on disk.
    """
    dataset_dir = 'RSTPReid'

    def __init__(self, root=''):
        """
        Initialize the RSTPReid dataset.

        Args:
            root (str): Root directory containing the RSTPReid folder.
        """
        super(RSTPReid, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)

        # ------------------------------------------------------------------
        # Define paths to training and testing annotation JSONs.
        # Each JSON contains (tar_path, can_path, id) pairs.
        # ------------------------------------------------------------------
        self.train_person_json_path_list = [
            [os.path.join(self.dataset_dir, 'train_pairs.json')],
        ]
        self.train_person_pairs = self._split_json(self.train_person_json_path_list)
        self.train_person, _ = self._process_json(self.train_person_pairs)

        self.test_person_json_path_list = [
            [op.join(self.dataset_dir, 'test_pairs.json')],
        ]
        self.test_person_pairs = self._split_json(self.test_person_json_path_list)
        self.test_person, _ = self._process_json(self.test_person_pairs)

        # Expose unified train / test attributes.
        self.train = self.train_person
        self.test = self.test_person

        self.logger = logging.getLogger("reid")
        self.logger.info("total training samples: {}".format(len(self.train)))
        self.logger.info("total testing samples: {}".format(len(self.test)))

    def _split_json(self, json_path_lists):
        """
        Load and flatten a list of JSON annotation paths.

        Args:
            json_path_lists: Nested list of JSON file paths.

        Returns:
            list: A flat list of annotation dictionaries.
        """
        img_pairs_list = []
        for json_path_list in json_path_lists:
            json_path = json_path_list[0]
            anno_json = read_json(json_path)
            for img_pair in anno_json:
                img_pairs_list.append(img_pair)
        return img_pairs_list

    def _process_json(self, annos: List[dict]):
        """
        Parse annotation dictionaries into a dataset format and collect IDs.

        Args:
            annos (List[dict]): List of annotation items.

        Returns:
            tuple: (dataset, id_container)
                - dataset: List of (id, tar_img_path, can_img_path) tuples.
                - id_container: Set of unique person IDs in this split.
        """
        id_container = set()
        dataset = []

        for anno in annos:
            pid = int(anno['id'])
            id_container.add(pid)
            tar_img_path = anno['tar_path']
            can_img_path = anno['can_path']
            dataset.append((pid, tar_img_path, can_img_path))

        return dataset, id_container