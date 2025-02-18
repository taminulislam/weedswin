# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class WeedDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('ABUTH_week_10', 'ABUTH_week_11', 'ABUTH_week_1', 'ABUTH_week_2', 'ABUTH_week_3', 'ABUTH_week_4', 'ABUTH_week_5', 'ABUTH_week_6', 'ABUTH_week_7', 'ABUTH_week_8', 'ABUTH_week_9', 'AMAPA_week_10', 'AMAPA_week_11', 'AMAPA_week_1', 
         'AMAPA_week_2', 'AMAPA_week_3', 'AMAPA_week_4', 'AMAPA_week_5', 'AMAPA_week_6', 'AMAPA_week_7', 'AMAPA_week_8', 'AMAPA_week_9', 'AMARE_week_10', 'AMARE_week_11', 'AMARE_week_1', 'AMARE_week_2', 'AMARE_week_3', 'AMARE_week_4', 
         'AMARE_week_5', 'AMARE_week_6', 'AMARE_week_7', 'AMARE_week_8', 'AMARE_week_9', 'AMATA_week_10', 'AMATA_week_11', 'AMATA_week_1', 'AMATA_week_2', 'AMATA_week_3', 'AMATA_week_4', 'AMATA_week_5', 'AMATA_week_6', 'AMATA_week_7', 
         'AMATA_week_8', 'AMATA_week_9', 'AMBEL_week_10', 'AMBEL_week_11', 'AMBEL_week_1', 'AMBEL_week_2', 'AMBEL_week_3', 'AMBEL_week_4', 'AMBEL_week_5', 'AMBEL_week_6', 'AMBEL_week_7', 'AMBEL_week_8', 'AMBEL_week_9', 'CHEAL_week_10', 
         'CHEAL_week_11', 'CHEAL_week_1', 'CHEAL_week_2', 'CHEAL_week_3', 'CHEAL_week_4', 'CHEAL_week_5', 'CHEAL_week_6', 'CHEAL_week_7', 'CHEAL_week_8', 'CHEAL_week_9', 'CYPES_week_10', 'CYPES_week_11', 'CYPES_week_1', 'CYPES_week_2', 
         'CYPES_week_3', 'CYPES_week_4', 'CYPES_week_5', 'CYPES_week_6', 'CYPES_week_7', 'CYPES_week_8', 'CYPES_week_9', 'DIGSA_week_10', 'DIGSA_week_11', 'DIGSA_week_1', 'DIGSA_week_2', 'DIGSA_week_3', 'DIGSA_week_4', 'DIGSA_week_5', 
         'DIGSA_week_6', 'DIGSA_week_7', 'DIGSA_week_8', 'DIGSA_week_9', 'ECHCG_week_10', 'ECHCG_week_11', 'ECHCG_week_1', 'ECHCG_week_2', 'ECHCG_week_3', 'ECHCG_week_4', 'ECHCG_week_5', 'ECHCG_week_6', 'ECHCG_week_7', 'ECHCG_week_8', 
         'ECHCG_week_9', 'ERICA_week_10', 'ERICA_week_11', 'ERICA_week_1', 'ERICA_week_2', 'ERICA_week_3', 'ERICA_week_4', 'ERICA_week_5', 'ERICA_week_6', 'ERICA_week_7', 'ERICA_week_8', 'ERICA_week_9', 'PANDI_week_10', 'PANDI_week_11', 
         'PANDI_week_1', 'PANDI_week_2', 'PANDI_week_3', 'PANDI_week_4', 'PANDI_week_5', 'PANDI_week_6', 'PANDI_week_7', 'PANDI_week_8', 'PANDI_week_9', 'SETFA_week_10', 'SETFA_week_11', 'SETFA_week_1', 'SETFA_week_2', 'SETFA_week_3', 
         'SETFA_week_4', 'SETFA_week_5', 'SETFA_week_6', 'SETFA_week_7', 'SETFA_week_8', 'SETFA_week_9', 'SETPU_week_10', 'SETPU_week_11', 'SETPU_week_1', 'SETPU_week_2', 'SETPU_week_3', 'SETPU_week_4', 'SETPU_week_5', 'SETPU_week_6', 
         'SETPU_week_7', 'SETPU_week_8', 'SETPU_week_9', 'SIDSP_week_10', 'SIDSP_week_11', 'SIDSP_week_1', 'SIDSP_week_2', 'SIDSP_week_3', 'SIDSP_week_4', 'SIDSP_week_5', 'SIDSP_week_6', 'SIDSP_week_7', 'SIDSP_week_8', 'SIDSP_week_9', 
         'SORHA_week_10', 'SORHA_week_11', 'SORHA_week_3', 'SORHA_week_4', 'SORHA_week_5', 'SORHA_week_6', 'SORHA_week_7', 'SORHA_week_8', 'SORHA_week_9', 'SORVU_week_10', 'SORVU_week_11', 'SORVU_week_1', 'SORVU_week_2', 'SORVU_week_3', 
         'SORVU_week_4', 'SORVU_week_5', 'SORVU_week_6', 'SORVU_week_7', 'SORVU_week_8', 'SORVU_week_9'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
    [(102, 179, 92),
 (14, 106, 71),
 (188, 20, 102),
 (121, 210, 214),
 (74, 202, 87),
 (116, 99, 103),
 (151, 130, 149),
 (52, 1, 87),
 (235, 157, 37),
 (129, 191, 187),
 (20, 160, 203),
 (57, 21, 252),
 (235, 88, 48),
 (218, 58, 254),
 (169, 255, 219),
 (187, 207, 14),
 (189, 189, 174),
 (189, 50, 107),
 (54, 243, 63),
 (248, 130, 228),
 (50, 134, 20),
 (72, 166, 17),
 (131, 88, 59),
 (13, 241, 249),
 (8, 89, 52),
 (129, 83, 91),
 (110, 187, 198),
 (171, 252, 7),
 (174, 34, 205),
 (80, 163, 49),
 (103, 131, 1),
 (253, 133, 53),
 (105, 3, 53),
 (220, 190, 145),
 (217, 43, 161),
 (201, 189, 227),
 (13, 94, 47),
 (14, 199, 205),
 (214, 251, 248),
 (189, 39, 212),
 (207, 236, 81),
 (110, 52, 23),
 (153, 216, 251),
 (187, 123, 236),
 (40, 156, 14),
 (44, 64, 88),
 (70, 8, 87),
 (128, 235, 135),
 (215, 62, 138),
 (242, 80, 135),
 (162, 162, 32),
 (122, 4, 233),
 (230, 249, 40),
 (27, 134, 200),
 (71, 11, 161),
 (32, 47, 246),
 (150, 61, 215),
 (36, 98, 171),
 (103, 213, 218),
 (34, 192, 226),
 (100, 174, 205),
 (130, 0, 4),
 (217, 246, 254),
 (141, 102, 26),
 (136, 206, 14),
 (89, 41, 123),
 (204, 178, 62),
 (95, 230, 240),
 (51, 252, 95),
 (131, 221, 228),
 (150, 230, 236),
 (142, 170, 28),
 (35, 12, 159),
 (70, 186, 242),
 (85, 27, 65),
 (169, 44, 61),
 (184, 244, 133),
 (27, 27, 107),
 (43, 83, 29),
 (189, 74, 127),
 (249, 246, 91),
 (216, 230, 189),
 (224, 128, 120),
 (26, 189, 120),
 (115, 204, 232),
 (2, 102, 197),
 (199, 154, 136),
 (61, 164, 224),
 (50, 233, 171),
 (151, 206, 58),
 (117, 159, 95),
 (215, 232, 179),
 (112, 61, 240),
 (185, 51, 11),
 (253, 38, 129),
 (130, 112, 100),
 (112, 183, 80),
 (186, 112, 1),
 (129, 219, 53),
 (86, 228, 223),
 (224, 128, 146),
 (125, 129, 52),
 (171, 217, 159),
 (197, 159, 246),
 (67, 182, 202),
 (183, 122, 144),
 (254, 37, 23),
 (68, 115, 97),
 (197, 213, 138),
 (254, 239, 143),
 (96, 200, 123),
 (186, 69, 207),
 (92, 2, 147),
 (251, 186, 163),
 (146, 89, 194),
 (254, 146, 147),
 (95, 198, 51),
 (232, 160, 167),
 (127, 38, 81),
 (103, 128, 10),
 (219, 184, 216),
 (177, 150, 158),
 (221, 41, 98),
 (6, 251, 143),
 (89, 111, 248),
 (243, 59, 112),
 (1, 128, 47),
 (253, 139, 196),
 (36, 159, 250),
 (246, 8, 232),
 (98, 146, 47),
 (207, 130, 147),
 (151, 53, 119),
 (160, 151, 115),
 (74, 112, 199),
 (163, 165, 103),
 (83, 253, 226),
 (111, 253, 216),
 (98, 152, 92),
 (145, 127, 109),
 (81, 193, 53),
 (162, 207, 188),
 (168, 227, 160),
 (67, 32, 141),
 (20, 47, 147),
 (247, 127, 135),
 (134, 194, 144),
 (127, 32, 175),
 (203, 186, 114),
 (213, 118, 21),
 (237, 157, 37),
 (229, 108, 50),
 (181, 7, 26),
 (26, 225, 20),
 (29, 96, 27),
 (110, 191, 224),
 (196, 251, 60),
 (47, 146, 3),
 (34, 191, 48),
 (255, 16, 171),
 (219, 157, 220),
 (45, 116, 5),
 (98, 123, 232),
 (36, 23, 92),
 (240, 45, 180),
 (94, 98, 187),
 (224, 115, 190),
 (252, 212, 159),
 (214, 160, 255),
 (66, 127, 17),
 (24, 233, 222),
 (53, 57, 66),
 (103, 173, 23),
 (113, 31, 174)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
