import os
import json
from PIL import Image
from torch.utils.data import Dataset

class Cambrian(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)        
        self.connect_sym = "!@#"
    
    def read_and_format_json(self, file_path):
        json_array = []
        with open(file_path, 'r') as file:
            for line in file:
                json_object = json.loads(line.strip())
                json_array.append(json_object)        
        return json_array    
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        info = self.ann[index]
        image_file = info['image']
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)        
        first_instruction = info['conversations'][0]['value'].replace('<image>', '').strip()
        first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)        
        questions = [first_instruction]
        answers = []        
        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 ==0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"]+" "
                questions.append(human_instruction)        
        questions = self.connect_sym.join(questions)
        answers = self.connect_sym.join(answers)

        # print("###################################")
        # print(image_path)
        # print(image)
        # print(questions)
        # print(answers)
        # print("###################################")
        # import sys; sys.exit(-1) 
        return {
            "image": image,
            "conv_q": questions,
            'conv_a': answers,
            "connect_sym": self.connect_sym
        }

class Cambrian_Text(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)        
        self.connect_sym = "!@#"    
    
    def read_and_format_json(self, file_path):
        json_array = []        # Open the file and read line by line
        with open(file_path, 'r') as file:
            for line in file:
                # Parse each line as a JSON object and append it to the list
                json_object = json.loads(line.strip())
                json_array.append(json_object)        
        return json_array
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        info = self.ann[index]
        first_instruction = info['conversations'][0]['value']
        questions = [first_instruction]
        answers = []
        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 ==0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"]+" "
                questions.append(human_instruction)        
                
        questions = self.connect_sym.join(questions)
        answers = self.connect_sym.join(answers)
        # print("###################################")
        # print(questions)
        # print(answers)
        # print("###################################")
        # import sys; sys.exit(-1)
        return {
            "conv_q": questions,
            'conv_a': answers,
            "connect_sym": self.connect_sym
        }