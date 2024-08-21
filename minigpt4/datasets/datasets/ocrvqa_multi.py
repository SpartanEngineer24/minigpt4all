import os
import json
import random
import transformers
from PIL import Image
from torch.utils.data import Dataset

def create_image_path(vis_root, image_path):
    vis_root = os.path.normpath(vis_root)
    image_path = os.path.normpath(image_path)
    
    vis_root_parts = vis_root.split(os.sep)
    image_parts = image_path.split(os.sep)
    
    overlap_start = 0
    for part in image_parts:
        if part in vis_root_parts:
            overlap_start = vis_root_parts.index(part)
            break
    
    vis_root_corrected = os.path.join(*vis_root_parts[:overlap_start])
    
    overlap_index = image_parts.index(vis_root_parts[overlap_start]) if overlap_start < len(vis_root_parts) else 0
    non_overlapping_image_path = os.path.join(*image_parts[overlap_index:])
    
    final_path = os.path.join(vis_root_corrected, non_overlapping_image_path)
    return "/" + final_path

class OCRVQAMultilingual(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        base, ext = os.path.splitext(ann_path)
        index_path = f"{base}_index{ext}"

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.connect_sym = "!@#"
        
        # Load the index file into memory
        self.index = {}
        with open(index_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.index[entry['id']] = entry['offset']
        self.data_file = ann_path

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        # Obtain the data using the index
        key = list(self.index.keys())[idx]
        offset = self.index[key]
        with open(self.data_file, 'r') as f:
            f.seek(offset)
            data = json.loads(f.readline().strip())

        image_path = create_image_path(self.vis_root, data['image'])
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
            "image_id": data['id'],
            "connect_sym": self.connect_sym
        }
