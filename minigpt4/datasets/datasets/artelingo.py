import pandas as pd
import os
from PIL import Image
import sys
from collections import OrderedDict
import random

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class ArtelingoMulti(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, percent=1.0, language=None):
        self.vis_root = vis_root
        for ann_path in ann_paths:
            self.annotation = pd.read_csv(ann_path).sample(frac=percent, random_state=42)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()
        img_ids = self.annotation['image_id'].unique()
        self.img_ids = {img_id: i for i, img_id in enumerate(img_ids)}
        
        # Define question templates
        self.question_templates = [
            "The given image is described in {LANG} as '{caption}'. What is the emotion associated with the image with respect to the culture of the language? Your options are amusement, excitement, awe, contentment, sadness, fear, disgust, anger, and other.",
            "The given image is described in {LANG} as '{caption}'. Would you say that this image reflects the emotion of amusement, excitement, awe, contentment, sadness, fear, disgust, anger, or other?",
            "An image is captioned in {LANG} as '{caption}'. What emotion does it convey? Options: amusement, excitement, awe, contentment, sadness, fear, disgust, anger, or other.",
            "Consider an image described in {LANG} with '{caption}'. Which emotion best describes the image? Choose from amusement, excitement, awe, contentment, sadness, fear, disgust, anger, or other.",
            "The image is described in {LANG} as '{caption}'. Based on this, identify the emotion depicted in the image. Possible emotions are amusement, excitement, awe, contentment, sadness, fear, disgust, anger, or other.",
            "Given the description '{caption}' in {LANG}, what is the predominant emotion in the image? Options include amusement, excitement, awe, contentment, sadness, fear, disgust, anger, and other.",
            "An image described as '{caption}' in {LANG} reflects which emotion? Select one: amusement, excitement, awe, contentment, sadness, fear, disgust, anger, or other.",
        ]

    def _add_instance_ids(self, key="instance_id"):
        self.annotation[key] = self.annotation.index

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        img_file = ann["image_name"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = ann["caption"]
        language = ann["language"]
        emotion = ann["emotion"]

        # Randomly select a question template and format it
        question_template = random.choice(self.question_templates)
        question = question_template.format(LANG=language, caption=caption)

        return {
            "image": image,
            "instruction_input": question,
            "answer": emotion
        }