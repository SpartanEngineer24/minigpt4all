import os
import json
from PIL import Image
from torch.utils.data import Dataset

class Llava_Multi(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        base, ext = os.path.splitext(ann_path)
        index_path = f"{base}_index{ext}"

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.connect_sym = "!@#"
        
        self.index = {}
        with open(index_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.index[entry['index']] = entry['offset']
        self.data_file = ann_path
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        key = list(self.index.keys())[idx]
        offset = self.index[key]
        with open(self.data_file, 'r') as f:
            f.seek(offset)
            data = json.loads(f.readline().strip())

        image_path = os.path.join(self.vis_root, data['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)

        first_instruction = data['conversations'][0]['value'].replace('<image>', '').replace('\n', ' ').strip()
        first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)

        questions = [first_instruction]
        answers = []

        for i, item in enumerate(data["conversations"][1:]):
            if i % 2 == 0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:  # human
                human_instruction = item["value"] + " "
                questions.append(human_instruction)

        questions = self.connect_sym.join(questions)
        answers = self.connect_sym.join(answers)

        return {
            "image": image,
            "conv_q": questions,
            "conv_a": answers,
            "connect_sym": self.connect_sym
        }