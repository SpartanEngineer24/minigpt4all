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
            "Given the {language} story: '{input_sentences}'. Provide a simple one-sentence ending in {language}.",
            "Here is a story in {language}: '{input_sentences}'. What would be a good, simple, single-sentence ending to this passage? Provide in {language}.",
            "Consider this {language} narrative: '{input_sentences}'. Generate in {language} a one sentence conclusion to the narrative.",
            "Read the following story in {language}: '{input_sentences}'. In {language}, provide a conclusion with a simple sentence.",
            "After reading this {language} story segment: '{input_sentences}', provide a suitable one-sentence ending in {language}."
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        language = entry['language']
        input_sentences = entry['input_sentences']

        question_template = random.choice(self.question_templates)

        question = question_template.format(language=language, input_sentences=input_sentences)

        return {
            "instruction_input": question,
            "answer": entry['fifthsentence']
        }
