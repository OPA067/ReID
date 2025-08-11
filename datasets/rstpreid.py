import logging
import os
import os.path as op
from typing import List
from utils.iotools import read_json

class RSTPReid:
    dataset_dir = 'reid_datasets'

    def __init__(self, root=''):
        super(RSTPReid, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)

        # person training and testing
        self.train_person_json_path_list = [
            [os.path.join(self.dataset_dir, 'person_jsons/cam_a_b.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/CUHK01.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/CUHK03.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/ICFG-PEDES-test.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/ICFG-PEDES-train.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/Market.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/RSTP-Reid.json')],
            [os.path.join(self.dataset_dir, 'person_jsons/train_query.json')],
            ]
        self.train_person_pairs = self._split_json(self.train_person_json_path_list)
        self.train_person, _ = self._process_json(self.train_person_pairs, type="person")

        self.test_person_json_path_list = [[op.join(self.dataset_dir, 'person_jsons/test_query.json')], ]
        self.test_person_pairs = self._split_json(self.test_person_json_path_list)
        self.test_person, _ = self._process_json(self.test_person_pairs, type="person")


        # car training and testing
        self.train_car_json_path_list = [[os.path.join(self.dataset_dir, f'car_jsons/video_{i}.json')] for i in range(1, 72)]
        # self.train_json_path_list_car = [[os.path.join(self.dataset_dir, 'train_car_jsons/train_video_1.json')],
        #                                  ......
        #                                  [os.path.join(self.dataset_dir, 'train_car_jsons/train_video_71.json')],]
        self.train_car_pairs = self._split_json(self.train_car_json_path_list)
        self.train_car, _ = self._process_json(self.train_car_pairs, type="car")

        self.test_car_json_path_list = [[op.join(self.dataset_dir, 'car_jsons/test_car.json')], ]
        self.test_car_pairs = self._split_json(self.test_car_json_path_list)
        self.test_car, _ = self._process_json(self.test_car_pairs, type="car")

        self.train = self.train_car + self.train_person
        self.test = self.test_car + self.test_person

        self.logger = logging.getLogger("ReID")
        self.logger.info("total train len is: {}".format(len(self.train)))
        self.logger.info("train_person len is: {}, train_car len is: {}".format(len(self.train_person), len(self.train_car)))
        self.logger.info("total test len is: {}".format(len(self.test)))
        self.logger.info("test_person len is: {}, test_car len is: {}".format(len(self.test_person), len(self.test_car)))

    def _split_json(self, json_path_lists):
        img_pairs_list = []
        for json_path_list in json_path_lists:
            json_path = json_path_list[0]
            json = read_json(json_path)
            for img_pair in json:
                img_pairs_list.append(img_pair)
        return img_pairs_list

    def _process_json(self, annos: List[dict], type: str):
        id_container = set()
        dataset = []
        if type == "person":
            for anno in annos:
                id = int(anno['id'])
                id_container.add(id)
                tar_img_path = op.join(self.dataset_dir + "/person_images", anno['tar_path'])
                can_img_path = op.join(self.dataset_dir + "/person_images", anno['can_path'])
                dataset.append((id, tar_img_path, can_img_path))
        else:
            for anno in annos:
                id = int(anno['id'])
                id_container.add(id)
                tar_img_path = op.join(self.dataset_dir + "/car_images", anno['tar_path'])
                can_img_path = op.join(self.dataset_dir + "/car_images", anno['can_path'])
                dataset.append((id, tar_img_path, can_img_path))

        return dataset, id_container