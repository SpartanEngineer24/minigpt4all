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

        return {
            "image": image,
            "conv_q": questions,
            'conv_a': answers,
            "connect_sym": self.connect_sym
        }

class Cambrian_Text(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, char_limit=8000):
        self.vis_root = vis_root
        self.text_processor = text_processor
        self.char_limit = char_limit  # Character limit for truncation
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)        
        self.connect_sym = "!@#"
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        info = self.ann[index]
        conversations = info.get('conversations', [])
        
        questions = []
        answers = []
        total_char = 0
        
        # Iterate over conversations and append question-answer pairs
        # Assuming the first element is the initial question
        for i in range(0, len(conversations), 2):
            question = conversations[i].get('value', "").strip()
            answer = ""
            if i + 1 < len(conversations):
                answer = conversations[i + 1].get('value', "").strip()
            
            pair_q = question
            pair_a = answer
            pair_total = len(pair_q) + len(pair_a)
            
            if total_char + pair_total > self.char_limit:
                if not questions and not answers:
                    # print("EXCEEDING LIMIT: ", total_char + pair_total)
                    return {
                        "conv_q": self.connect_sym,
                        'conv_a': self.connect_sym,
                        "connect_sym": self.connect_sym
                    }
                else:
                    break
            else:
                questions.append(pair_q)
                answers.append(pair_a)
                total_char += pair_total
        
        truncated_questions = self.connect_sym.join(questions)
        truncated_answers = self.connect_sym.join(answers)
        
        return {
            "conv_q": truncated_questions,
            'conv_a': truncated_answers,
            "connect_sym": self.connect_sym
        }