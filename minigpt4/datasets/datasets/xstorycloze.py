import random
import json
from torch.utils.data import Dataset

class XStoryCloze(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.entries = []
        with open(ann_path, 'r') as file:
            for line in file:
                self.entries.append(json.loads(line.strip()))
        
        self.question_templates = [
            "Given the story: '{}'. Choose the correct ending. Option 1: {}. Option 2: {}.",
            "Here is a story: '{}'. What is the most suitable conclusion? Option 1: {}. Option 2: {}.",
            "Consider this narrative: '{}'. Which of the following endings fits best? Option 1: {}. Option 2: {}.",
            "Read the following story: '{}'. Which ending do you think is more appropriate? Option 1: {}. Option 2: {}.",
            "After reading this story segment: '{}', decide which of these endings aligns better with the story. Option 1: {}. Option 2: {}.",
            "For this story context: '{}', which conclusion seems more fitting? Option 1: {}. Option 2: {}.",
            "Given the narrative detailed in '{}', which of these endings do you think best completes the tale? Option 1: {}. Option 2: {}."
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        question_template = random.choice(self.question_templates)
        question = question_template.format(entry['Paragraph'], entry['Option1'], entry['Option2'])

        correct_answer = 'Option 1' if entry['Answer'] == '1' else 'Option 2'

        return {
            "instruction_input": question,
            "answer": correct_answer
        }
