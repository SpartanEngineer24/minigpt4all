import os
import json
import random
import langcodes
from PIL import Image
from torch.utils.data import Dataset

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
        self.instruction_pool = [
            ' <Img><ImageHere></Img> Given the image, could you please generate in {} a scientific caption for it? ',
            ' <Img><ImageHere></Img> Based on the image, please give a Wikipedia-style description in {}. ',
            ' <Img><ImageHere></Img> Given the visual representation, please use {} to provide a scholastic description. ',
            ' <Img><ImageHere></Img> For the image provided, please describe it in {} using academic language. ',
            ' <Img><ImageHere></Img> Considering the picture, please come up with an official caption for it in {}. ',
            ' <Img><ImageHere></Img> With regards to this image, please use {} to write an informative description. '
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key = list(self.index.keys())[idx]
        offset = self.index[key]
        with open(self.data_file, 'r') as f:
            f.seek(offset)
            data = json.loads(f.readline().strip())
        
        image_path = os.join(self.vis_root, data['image_id'])
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)

        language_code = data['language']
        language = langcodes.Language.get(language_code).display_name()
        
        question = random.choice(self.instruction_pool).format(language)
        answer = data['content'].replace('\n', ' ').strip()

        return {
            "image": image,
            "instruction_input": question,
            "answer": answer,
        }