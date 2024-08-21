import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

class CCSBUMulti(Dataset):
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

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key = list(self.index.keys())[idx]
        offset = self.index[key]
        with open(self.data_file, 'r') as f:
            f.seek(offset)
            data = json.loads(f.readline().strip())
        
        image_path = os.path.join(self.vis_root, f"{data['image_id']}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)

        language = data['language']
        if language == 'eng_Latn':
            question = '<Img><ImageHere></Img> Please describe this image in detail. {}'
        elif language == 'zho_Hans':
            question = '<Img><ImageHere></Img> 请详细描述这张图片。{}'
        elif language == 'kor_Hang':
            question = '<Img><ImageHere></Img> 이 이미지를 자세히 설명해주세요. {}'
        elif language == 'hin_Deva':
            question = '<Img><ImageHere></Img> कृपया इस छवि का विस्तार से वर्णन करें। {}'
        elif language == 'spa_Latn':
            question = '<Img><ImageHere></Img> Por favor describe esta imagen en detalle. {}'
        elif language == 'fra_Latn':
            question = '<Img><ImageHere></Img> Veuillez décrire cette image en détail. {}'
        elif language == 'arb_Arab':
            question = '<Img><ImageHere></Img> يرجى وصف هذه الصورة بالتفصيل. {}'
        elif language == 'ben_Beng':
            question = '<Img><ImageHere></Img> এই ছবিটি বিস্তারিত বর্ণনা করুন. {}'
        elif language == 'rus_Cyrl':
            question = '<Img><ImageHere></Img> Пожалуйста, опишите этот образ подробно. {}'
        elif language == 'urd_Arab':
            question = '<Img><ImageHere></Img> براہ کرم اس تصویر کو تفصیل سے بیان کریں۔ {}'
        elif language == 'jpn_Jpan':
            question = '<Img><ImageHere></Img> この画像を詳しく説明してください。{}'
        else:
            question = '<Img><ImageHere></Img> Please describe this image. {}'

        answer = data['caption']

        return {
            "image": image,
            "instruction_input": question,
            "answer": answer,
        }