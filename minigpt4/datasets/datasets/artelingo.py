import pandas as pd
import os
from PIL import Image
import sys
from collections import OrderedDict
from minigpt4.datasets.datasets.base_dataset import BaseDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

class ArtelingoMulti(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, percent=1.0, language=None):
        self.vis_root = vis_root
        for ann_path in ann_paths:
            print('Dataset Percent:', percent)
            self.annotation = pd.read_csv(ann_path).sample(frac=percent, random_state=42)
            if language is not None:
                print('Filtering language:', language)
                self.annotation = self.annotation[self.annotation['language'] == language]
            print('Dataset Size:', len(self.annotation))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        img_ids = self.annotation['image_id'].unique()
        self.img_ids = {img_id: i for i, img_id in enumerate(img_ids)}

    def _add_instance_ids(self, key="instance_id"):
        self.annotation[key] = self.annotation.index
        
    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        #print(ann)
        img_file = ann["image_name"]
        image_path = os.path.join(self.vis_root, img_file)

        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            return None

        ann = self.annotation.iloc[index]

        img_file = ann["image_name"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "art_style": ann["art_style"],
            "painting": ann["painting"],
            "emotion": ann["emotion"],
            "language": ann["language"],
        }