import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from langcodes import Language

class WIT(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        base, ext = os.path.splitext(ann_path)
        index_path = f"{base}_index{ext}"
        self.index = {}
        with open(index_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.index[entry['index']] = entry['offset']
        self.data_file = ann_path
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.question_templates = [
            "<Img><ImageHere></Img> Consider the following two texts. Text 1: {}. Text 2: {}. Which text is most associated with the given image? Answer with 'Text 1' or 'Text 2'.",
            "<Img><ImageHere></Img> Here are two descriptions. Text 1: {}. Text 2: {}. Based on the image, which text corresponds to the image? Answer with 'Text 1' or 'Text 2'.",
            "<Img><ImageHere></Img> Look at the following texts. Text 1: {}. Text 2: {}. Which text best matches the given image? Respond with 'Text 1' or 'Text 2'.",
            "<Img><ImageHere></Img> Given the image, compare these texts. Text 1: {}. Text 2: {}. Which text is associated with the image? Reply with 'Text 1' or 'Text 2'.",
            "<Img><ImageHere></Img> Review the following two texts. Text 1: {}. Text 2: {}. Which text is related to the given image? Answer 'Text 1' or 'Text 2'.",
            "<Img><ImageHere></Img> With reference to the image, consider these texts. Text 1: {}. Text 2: {}. Which text fits the image? Reply with 'Text 1' or 'Text 2'."
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key = list(self.index.keys())[idx]
        offset = self.index[key]
        with open(self.data_file, 'r') as f:
            f.seek(offset)
            data1 = json.loads(f.readline().strip())

            try:
                data2 = json.loads(f.readline().strip())
            except json.JSONDecodeError:
                f.seek(0)
                data2 = json.loads(f.readline().strip())

        # Randomize the assignment of Text 1 and Text 2
        if random.random() > 0.5:
            text1 = data1['content'].replace('\n', ' ').strip()
            text2 = data2['content'].replace('\n', ' ').strip()
            answer = "Text 1"
        else:
            text1 = data2['content'].replace('\n', ' ').strip()
            text2 = data1['content'].replace('\n', ' ').strip()
            answer = "Text 2"

        question_template = random.choice(self.question_templates)
        question = question_template.format(text1, text2)

        image_path = os.path.join(self.vis_root, data1['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)

        return {
            "image": image,
            "instruction_input": question,
            "answer": answer,
        }

class WITText(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.entries = []
        with open(ann_path, 'r') as file:
            for line in file:
                self.entries.append(json.loads(line.strip()))
        
        self.question_templates = [
            "First half of a wikipedia article in {language} is as follows: '{first_half}'. Generate the second half",
            "Given the first part of the Wikipedia passage in {language}, complete it: '{first_half}'",
            "Here is a truncated Wikipedia article in {language}: '{first_half}'. Please continue the article.",
            "This is a part of a Wikipedia article in {language}: '{first_half}'. Give a possible completion of it.",
            "Finish the following text, retrieved in part from Wikipedia: '{first_half}'. Answer in {language}."
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        language_code = entry['language']
        
        language = Language.get(language_code).display_name()

        question_template = random.choice(self.question_templates)
        question = question_template.format(first_half=entry['first_half'], language=language)

        return {
            "instruction_input": question,
            "answer": entry['second_half']
        }