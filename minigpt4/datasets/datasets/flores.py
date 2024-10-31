import random
import json
from torch.utils.data import Dataset

class Flores(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.language_names = {
            'ben_Beng': 'Bengali', 'arb_Arab': 'Arabic', 'kor_Hang': 'Korean', 
            'zho_Hans': 'Chinese', 'rus_Cyrl': 'Russian', 'spa_Latn': 'Spanish',
            'fra_Latn': 'French', 'jpn_Jpan': 'Japanese', 'urd_Arab': 'Urdu', 
            'hin_Deva': 'Hindi', 'eng_Latn': 'English'
        }
        
        self.languages = list(self.language_names.keys())
        
        self.language_pairs = [
            (lang1, lang2) for lang1 in self.languages for lang2 in self.languages if lang1 != lang2
        ]
        
        self.data = []
        
        # Read the JSONL file line by line
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_obj = json.loads(line)
                    self.data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_number}: {e}")
        
        self.total_length = len(self.data) * len(self.language_pairs)

        self.translation_templates = [
            "Translate the following {source_lang} passage into {target_lang}: '{text_lang1}'",
            "Convert this text from {source_lang} to {target_lang}: '{text_lang1}'",
            "Here is a sentence in {source_lang}. Please translate it into {target_lang}: '{text_lang1}'",
            "Given the {source_lang} sentence, provide the {target_lang} translation: '{text_lang1}'",
            "Rewrite the following in {target_lang} from {source_lang}: '{text_lang1}'",
            "How would you say this in {target_lang} (originally in {source_lang}): '{text_lang1}'",
            "Translate from {source_lang} to {target_lang} for this passage: '{text_lang1}'",
            "Change this {source_lang} text into {target_lang}: '{text_lang1}'",
            "Please provide a {target_lang} version of this {source_lang} sentence: '{text_lang1}'"
        ]

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        data_idx = index // len(self.language_pairs)
        pair_idx = index % len(self.language_pairs)
        
        data_item = self.data[data_idx]
        lang1, lang2 = self.language_pairs[pair_idx]
        
        text_lang1 = data_item.get('sentence_' + lang1)
        text_lang2 = data_item.get('sentence_' + lang2)
        
        if text_lang1 is None or text_lang2 is None:
            return self.__getitem__((index + 1) % self.total_length)

        lang1_name = self.language_names.get(lang1, lang1)
        lang2_name = self.language_names.get(lang2, lang2)

        template = random.choice(self.translation_templates)
        instruction_input = template.format(source_lang=lang1_name, target_lang=lang2_name, text_lang1=text_lang1)

        return {
            "instruction_input": instruction_input,
            "answer": text_lang2
        }
