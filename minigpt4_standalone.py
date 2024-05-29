#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import os
import random
import requests
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, StoppingCriteriaList, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


# In[4]:


parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)


# In[5]:


print('Initializing Chat')
cfg = Config(parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_object_detection_448x448.yaml']))

# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2/minigpt4_all_pretrain_dataset_448_llama_7b/20230628101/checkpoint_11.pth"
# cfg.model_cfg.ckpt="/ibex/project/c2133/zhud/minigpt4_v2_checkpoint/minigpt4_second_stage_finetune/20230704000/checkpoint_0.pth"
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_vqa_slide_448_llama_7b_64x64/20230629093/checkpoint_21.pth"
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_plotvqa_llama_7b_448_v1/20230630223/checkpoint_9.pth"

# # 224-64
# cfg = Config(parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_object_detection_224x224.yaml']))
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_instructblip_ratio_224x224/20230701235/checkpoint_32.pth"


## 224-100
# cfg = Config(parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_object_detection_224x224.yaml']))
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_instructblip_ratio_224x224_bbox_100/20230702005/checkpoint_32.pth"



# 336-100
cfg = Config(parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_object_detection_336x336.yaml']))
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_instructblip_ratio_336x336_bbox_100/20230702010/checkpoint_19.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_336x336_best_ratio_bbox64_v3/20230703013/checkpoint_11.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_336x336_best_ratio_bbox64/20230703010/checkpoint_14.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_vqa_slide_336_llama_7b_64x64_1e4/20230705015/checkpoint_32.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_vqa_slide_336_llama_7b_64x64_2e4_v1/20230715010/checkpoint_20.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_vqa_slide_336_llama_7b_64x64_2e4_v1/4gpus/checkpoint_17.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/336_finetune_test/20230716094/checkpoint_3.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/336_v4/20230716095/checkpoint_9.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/llava_finetune_336/20230716215/checkpoint_2.pth"
# cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_vqa_slide_336_llama_7b_64x64_1e4/20230705015/checkpoint_32.pth"
cfg.model_cfg.ckpt = "/ibex/project/c2133/minigpt4_v2_checkpoint/336_v4/20230716095/checkpoint_21.pth"
# 448-64
# cfg = Config(parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_object_detection_448x448.yaml']))
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_vqa_slide_448_llama_7b_64x64/20230629093/checkpoint_21.pth"

# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_instructblip_ratio_448x448/20230701235/checkpoint_11.pth"

# 448-100
# cfg = Config(parser.parse_args(['--cfg-path', 'eval_configs/minigpt4_object_detection_448x448.yaml']))
# cfg.model_cfg.ckpt="/ibex/project/c2133/minigpt4_v2_checkpoint/minigpt4_caption_all_datasets_instructblip_ratio_448x448_bbox100/20230702005/checkpoint_5.pth"


model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)

# print("model config", model_config)
model = model_cls.from_config(model_config).to('cuda:0')

# vis_processor_cfg = cfg.datasets_cfg.cc_combine.vis_processor.train
# vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
# chat = Chat(model, vis_processor)
print('Initialization Finished')


# In[6]:


bounding_box_size=64


# In[7]:


vis_processor_cfg = cfg.datasets_cfg.coco.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


# In[ ]:





# In[8]:


import os
import json
import random
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import re


def random_sample_and_display(folder_path):
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    
    # Randomly select one image file
    image_filename = random.choice(image_files)
    
    # Construct the file paths
    image_path = os.path.join(folder_path, image_filename)
    print("image_path", image_path)
    json_filename = os.path.splitext(image_filename)[0] + '.json'
    json_path = os.path.join(folder_path, json_filename)
    
    # Load the image
    image = Image.open(image_path)
    
    # Load the JSON data
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    bbox = json_data["bbox"]
    objects = json_data["objects"]
        
    image_width, image_height = image.size
    
    
    draw = ImageDraw.Draw(image)
    
    pattern = r'^\d+'


    font = ImageFont.load_default()
    font_size = 30
    
    # Iterate over the bounding boxes in the JSON data
    for bbox,obj in zip(bbox,objects):
        x0,y0,width,height = bbox[0],bbox[1], bbox[2], bbox[3]
        
        # Calculate the coordinates of the bounding box
        left = x0 * image_width
        bottom = y0 * image_height
        right = x0 * image_width+ width * image_width
        top = y0 * image_height + height * image_height

        
        # Draw the bounding box rectangle
#         draw.rectangle([left, bottom, right, top])
        draw.rectangle([0, 0, 5, 5])
        
        
        # Determine the position to draw the object name
        text_width, text_height = draw.textsize(obj, font=font)
        text_x = left
        text_y = top - text_height - 2
        
        # Draw the object name
        draw.text((text_x, text_y), obj, fill='black', font=font)
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    return image, json_data

# image, json_file = random_sample_and_display('/ibex/project/c2133/object_detection_datasets/Objects365/openImages/dataset/0')
image,json_file = random_sample_and_display("/ibex/project/c2133/object_detection_datasets/coco_bbox/dataset/1")


# In[9]:


import os
import json
import random
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import re


def random_sample_and_display(folder_path):
    
    
    json_data = json.load(open(folder_path))
    key_sample = random.sample(json_data["data"],1)[0]
#     print(key_sample)
    
    
    ocr_info = key_sample["ocr_info"]

    sampled_ocr = random.sample(ocr_info,1)[0]

    # print("sampled ocr", sampled_ocr)

    text = sampled_ocr["word"]
    width = sampled_ocr["bounding_box"]["width"]
    height = sampled_ocr["bounding_box"]["height"]
    top_left_x = sampled_ocr["bounding_box"]["top_left_x"]
    top_left_y = sampled_ocr["bounding_box"]["top_left_y"]
    
    
    image_file = '{}.jpg'.format(key_sample['image_id'])

    image_path = os.path.join("/ibex/project/c2133/minigpt4_v2_dataset/TextCaps/train_images", image_file)
    # Load the image
    image = Image.open(image_path)
        
    image_width, image_height = image.size
    
    bbox = [top_left_x*image_width,top_left_y* image_height ,width* image_width, height*image_height]
    
    
    draw = ImageDraw.Draw(image)
    
    pattern = r'^\d+'


    font = ImageFont.load_default()
    font_size = 30
    
    # Iterate over the bounding boxes in the JSON data

    x0,y0,width,height = bbox[0],bbox[1], bbox[2], bbox[3]

    # Calculate the coordinates of the bounding box
    left = x0 
    bottom = y0 
    right = x0 + width 
    top = y0  + height
    
    print(left, bottom, right, height)
    print([left/image_width, bottom/image_height],[right/image_width,top/image_height])


    # Draw the bounding box rectangle
    draw.rectangle([left, bottom, right, top], outline='red')


    # Determine the position to draw the object name
    text_width, text_height = draw.textsize(text, font=font)
    text_x = left
    text_y = top - text_height - 2

    # Draw the object name
#     draw.text((text_x, text_y), text, fill='black', font=font)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    print(left,bottom, right, top, text)
    
    return image, json_data

# image, json_file = random_sample_and_display('/ibex/project/c2133/object_detection_datasets/Objects365/openImages/dataset/0')
# image,json_file = random_sample_and_display("/ibex/ai/project/c2133/minigpt4_v2_dataset/textocr/TextOCR_0.1_train.json")
image,json_file = random_sample_and_display("/ibex/project/c2133/minigpt4_v2_dataset/TextCaps/TextVQA_Rosetta_OCR_v0.2_train.json")


# In[10]:


# from transformers.generation.configuration_utils import GenerationConfig
# new_generation_config = GenerationConfig.from_model_config(chat.model.llama_model.config)
# chat.model.llama_model.generation_config
# new_generation_config


# In[11]:


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        self.conv = CONV_VISION.copy()
        self.img_list = []
        self.raw_answers = []

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def reset(self):
        self.conv.messages = []
        self.img_list = []
        # self.img_list = [img for img in self.conv.system_img]
        self.raw_answers = []

    def ask(self, text):
        if len(self.conv.messages) > 0 and self.conv.messages[-1][0] == self.conv.roles[0] \
                and self.conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            self.conv.messages[-1][1] = ' '.join([self.conv.messages[-1][1], text])
        else:
            self.conv.append_message(self.conv.roles[0], text)

    def answer(self, max_new_tokens=200, num_beams=5, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1):
        self.conv.append_message(self.conv.roles[1], None)
        embs = self.get_context_emb()
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        self.raw_answers.append(output_text)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        self.conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        
        image_emb, _ = self.model.encode_img(image)
        self.img_list.append(image_emb)
        self.conv.append_message(self.conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self):
        prompt = self.conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(self.img_list) + 1, "Unmatched numbers of image placeholders and images."
        
#         print("prompt seg", prompt_segs)
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(self.device).input_ids  # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], self.img_list) for emb in pair] + [seg_embs[-1]]

        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs


# In[12]:


import requests
from io import BytesIO


# In[ ]:





# In[13]:


import os
import json
import random
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import re


def sample_one_bbox(folder_path):
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    
    # Randomly select one image file
    image_filename = random.choice(image_files)
    
    # Construct the file paths
    image_path = os.path.join(folder_path, image_filename)
#     print("image_path", image_path)
    json_filename = os.path.splitext(image_filename)[0] + '.json'
    json_path = os.path.join(folder_path, json_filename)
    
    # Load the image
    image = Image.open(image_path)
    
    # Load the JSON data
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    bbox = json_data["bbox"]
    objects = json_data["objects"]
#     center_point = json_data["center_point"]
    
    image_width, image_height = image.size
    
    draw = ImageDraw.Draw(image)
    

    font = ImageFont.load_default()
    font_size = 30
    
    # Iterate over the bounding boxes in the JSON data
    
    sample_index = random.sample(range(len(bbox)),1)[0]
    sample_bbox = bbox[sample_index]
    sample_object = objects[sample_index]

    x0,y0,x1,y1 = sample_bbox[0],sample_bbox[1], sample_bbox[2], sample_bbox[3]

    left = x0 * image_width
    bottom = y0 * image_height
    right = x0 * image_width + x1 * image_width
    top = y0 * image_height + y1 * image_height

    # Draw the bounding box rectangle
    draw.rectangle([left, bottom, right, top], outline='red')



    # Determine the position to draw the object name
    text_width, text_height = draw.textsize(sample_object, font=font)
    text_x = left
    text_y = top - text_height - 2

    # Draw the object name
    draw.text((text_x, text_y), sample_object, fill='black', font=font)

    return sample_bbox, sample_object,image_path



# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


## refer region


# In[ ]:





# In[15]:


##  visualize all the bounding boxes together


# In[16]:


def extract_substrings(string):
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]
    
    return substrings

def visualize_all_bbox_together(image_path, generation=""):
    
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.load_default()
    font_size = 30


    string_list = extract_substrings(generation)
    
    for string in string_list:
        integers = re.findall(r'-?\d+', string)
        obj = string.split(",")[0]
        if len(integers)==4:
            x0,y0,x1,y1 = int(integers[0]),int(integers[1]), int(integers[2]), int(integers[3])

            left = x0/bounding_box_size * image_width
            bottom = y0/bounding_box_size * image_height
            right = x1/bounding_box_size * image_width
            top = y1/bounding_box_size* image_height

            draw.rectangle([left, bottom, right, top], outline='red')
            
            text_width, text_height = draw.textsize("minigpt-4", font=font)
            text_x = left
            text_y = top - text_height - 2

            draw.text((text_x, text_y), obj, fill='black', font=font)

    
    
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    


# In[17]:


# # sample_bbox, sample_object, image_path = sample_one_bbox('/ibex/project/c2133/object_detection_datasets/coco_bbox/dataset/1')
# sample_bbox, sample_object, image_path = sample_one_bbox('/ibex/project/c2133/object_detection_datasets/Objects365/openImages/dataset/0')

# # print("the sampeld bbox", sample_bbox, "sample object", sample_object)
# chat = Chat(model, vis_processor)
# # chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
# chat.conv.system=""
# image = Image.open(image_path).convert("RGB")
# chat.upload_img(image)
# instruction = r" Given an image, identify the objects and their bounding boxes in the format of {object,x1,y1,x2,y2}. identify the objects as many as possible. "
# chat.ask(instruction)
# a, a_token = chat.answer(max_new_tokens=200)
# print("minigpt-4  stage 1: ",a)


# # chat.ask(r" continue to identify other objects! ")
# # a, a_token = chat.answer(max_new_tokens=200))
# # print("minigpt-4 stage 2: ",a)


# visualize_all_bbox_together(image_path, a)
# chat.reset()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


# chat = Chat(model, vis_processor)
# image_link = "image_path"
# image = Image.open(image_link).convert("RGB")
# chat.upload_img(image)
# chat.ask("please describe this image in details")
# a, a_token = chat.answer(max_new_tokens=1000)
# print(a)
# chat.reset()


# In[19]:


# test_json="/ibex/project/c2133/minigpt4_v2_dataset/plot_vqa/test_qa_pairs_V2.json"
# plot_data = json.load(open(test_json))


# In[20]:


# da = random.sample(plot_data["qa_pairs"],1)[0]
# image_file = str(da["image_index"])+".png"
# question = da["question_string"]
# answer = str(da["answer"])
# image_path = "/ibex/project/c2133/minigpt4_v2_dataset/plot_vqa/test_png/"+image_file
# print(image_path)



# chat = Chat(model, vis_processor)
# # chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
# chat.conv.system=""
# # image_path = "/ibex/project/c2133/minigpt4_v2_dataset/vqav2/val/val2014/COCO_val2014_000000262148.jpg"
# image = Image.open(image_path).convert("RGB")
# print(image.size)
# chat.upload_img(image)
# # instruction = "Identify the object in the spatial location of {<26><37><36><60>} "
# # chat.ask(instruction)
# # chat.ask(r"what are the other people doing in the background")

# chat.ask(r" extract the text that you can see from the image")
# # chat.ask(question)

# a, a_token = chat.answer(max_new_tokens=40)
# print("question: ", question)
# print("gt: ", answer)
# print("minigpt4: ",a)

# # print("chat",chat.conv)
# visualize_all_bbox_together(image_path, a)
# chat.reset()


# In[21]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
# chat.conv.system=""
image_path = "/ibex/project/c2133/minigpt4_v2_dataset/vqav2/val/val2014/COCO_val2014_000000393225.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "Identify the object in the spatial location of {<26><37><36><60>} "
# instruction = "please write a romantic and long poem based on the given image content"
instruction = "where is bounding box location of flowers"
chat.ask(instruction)
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=200)
print(a)
print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()



# In[ ]:





# In[30]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/minigpt4_v2_dataset/vqav2/val/val2014/COCO_val2014_000000393225.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "Identify the object in the spatial location of {<26><37><36><60>} "
# chat.ask(instruction)
# '###Human: <Img><ImageHere></Img> Question: are the lights on in this room? Short answer:###Assistant: '
# chat.ask(r"Question: are there fish in the image? Short answer:")
# chat.ask(r"[bbox] tell me the location of spoon")

# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
chat.ask(r"tell me the bounding box location of the meat")
a, a_token = chat.answer(max_new_tokens=200)
print(a)
print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[23]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
# chat.conv.system=""
image_path = "/ibex/ai/reference/CV/COCO/cocoapi/data/2014/images/jpeg/train/COCO_train2014_000000579057.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "Identify the object in the spatial location of {<26><37><36><60>} "
# chat.ask(instruction)
# '###Human: <Img><ImageHere></Img> Question: are the lights on in this room? Short answer:###Assistant: '
# chat.ask(r"Question: are there fish in the image? Short answer:")
# chat.ask(r"[bbox] tell me the location of orange on the left")

chat.ask(r" tell me the bounding box location of the right-most orange")
# chat.ask(r" can you write a poem based on the image content that you can see in the image? please discuss the poem")
a, a_token = chat.answer(max_new_tokens=20)
print(a)

# chat.ask(r" tell me what fruit you can see from the image")
# chat.ask(r" tell me the bounding box location of them")
# a, a_token = chat.answer(max_new_tokens=20)

# chat.ask(r" what color does it have for the object in {<0><19><8><26>}")
# chat.ask(r"[bbox] tell me where are them?")
# a, a_token = chat.answer(max_new_tokens=20)
# print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[24]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
# chat.conv.system=""
image_path = "/ibex/ai/reference/CV/COCO/cocoapi/data/2014/images/jpeg/train/COCO_train2014_000000579057.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "Identify the object in the spatial location of {<26><37><36><60>} "
# chat.ask(instruction)
# '###Human: <Img><ImageHere></Img> Question: are the lights on in this room? Short answer:###Assistant: '
# chat.ask(r"Question: are there fish in the image? Short answer:")
# chat.ask(r"[bbox] tell me the location of orange on the left")

# chat.ask(r"  can you see the oranges in the image?")
# # chat.ask(r" can you write a poem based on the image content that you can see in the image? please discuss the poem")
# a, a_token = chat.answer(max_new_tokens=20)
# print(a)

# chat.ask(r" tell me what fruit you can see from the image")
chat.ask(r" tell me the bounding box location of oranges")
a, a_token = chat.answer(max_new_tokens=20)

# chat.ask(r" what color does it have for the object in {<0><19><8><26>}")
# chat.ask(r"[bbox] tell me where are them?")
# a, a_token = chat.answer(max_new_tokens=20)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[23]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/pdf_test.png"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "what is in the location of {<10><15><17><29>} "
# chat.ask(instruction)
chat.ask(r" [ocr] what does AI say?")

# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=100)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[33]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/messi_2.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "what is in the location of {<10><15><17><29>} "
# chat.ask(instruction)
chat.ask(r" tell me the bounding box of the right person")

# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=20)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[34]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/messi_3.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "what is in the location of {<10><15><17><29>} "
# chat.ask(instruction)
# chat.ask(r"tell me the bounding box location of messi")
chat.ask(r"how many people are in the image")
# chat.ask(r"who is in the location of {<23><20><49><98>}")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=20)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)


# In[35]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/group.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "what is in the location of {<10><15><17><29>} "
# chat.ask(instruction)
# chat.ask(r"tell me the bounding box location of messi")
chat.ask(r"how many people you can see from the image?")
# chat.ask(r"who is in the location of {<23><20><49><98>}")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=20)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)


# In[36]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/messi_4.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "what is in the location of {<10><15><17><29>} "
# chat.ask(instruction)
# chat.ask(r"please generate the grounded image caption")
# chat.ask(r"provide the description of the image along with the localized region ")
# chat.ask(r"generate an image caption incorporating the grounded phrase within the bounding box ")
# chat.ask(r"generate an image caption including the corresponding grounded bounding box phrase")
chat.ask(r"generate a short image caption")
# chat.ask(r"who is in the location of {<22><12><38><62>}")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=200)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)


# In[37]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/messi_4.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction = "what is in the location of {<10><15><17><29>} "
# chat.ask(instruction)
chat.ask(r"tell me the bounding box location of messi")
# chat.ask(r"how many people are in the image")
# chat.ask(r"who is in the location of {<22><12><38><62>}")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=20)
print(a)
chat.ask(r"tell me where they are in the format of bounding box?")
a, a_token = chat.answer(max_new_tokens=200)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)


# In[38]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/shanghai.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# instruction=r"Extract the text from the given bounding box  {<32><10><48><80>}"
# instruction = r"tell me the bouding box location of adverstiment board writing 'Jaddy' "
instruction = r" Provide a list of texts observed in the provided image"
chat.ask(instruction)
# chat.ask(r" tell me the bouding box location of the person ")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=20)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[ ]:





# In[39]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/office.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)

# instruction = r"Identify the object in the spatial location of {<26><0><32><9>}"
instruction = r"tell me the bounding box location of the lamp"
chat.ask(instruction)
# chat.ask(r" tell me the bouding box location of the table ")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=200)
print(a)
visualize_all_bbox_together(image_path, a)

chat.ask("tell me the bounding box location of the laptop")
a, a_token = chat.answer(max_new_tokens=200)
visualize_all_bbox_together(image_path, a)
print(a)
# print("chat",chat.conv)
chat.reset()


# In[40]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/office.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)

# instruction = r"Identify the object in the spatial location of {<26><0><32><9>}"
instruction = r"tell me the bounding box location of the laptop"
chat.ask(instruction)
# chat.ask(r" tell me the bouding box location of the table ")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=100)
print(a)
visualize_all_bbox_together(image_path, a)

chat.ask("tell me the bounding box location of the lamp")
a, a_token = chat.answer(max_new_tokens=100)
visualize_all_bbox_together(image_path, a)
print(a)
# print("chat",chat.conv)
chat.reset()


# In[58]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/office.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)

# instruction = r"Identify the object in the spatial location of {<15><16><26><28>}"
# instruction = r"is the man on the right side of the laptop?"
# chat.ask(instruction)
chat.ask(r"tell me the bounding box of the man")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=100)
print(a)
visualize_all_bbox_together(image_path, a)
chat.ask(r"tell me the bounding box of the laptop")
a, a_token = chat.answer(max_new_tokens=100)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[42]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/office.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)

# instruction = r"Identify the object in the spatial location of {<15><16><26><28>}"
# instruction = r"is the man on the right side of the laptop?"
# chat.ask(instruction)
chat.ask(r"tell me the bounding box location of the laptop ")
# chat.ask(r" what is the object in the spatial location of {<81>,<3>,<165>,<214>}")
# chat.ask(r" describe this image in very details")
a, a_token = chat.answer(max_new_tokens=100)
print(a)
# print("chat",chat.conv)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[ ]:





# In[59]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. we will ask the  "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/galaxy.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
# chat.ask(r"c")
instruction = r"Identify the object in the spatial location of {<8><0><32><28>}"
chat.ask(instruction)
a, a_token = chat.answer(max_new_tokens=200)
print(a)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[61]:


chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. we will ask the  "
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/people_count.jpg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
chat.ask(r"tell me the bounding box location of the right up stairs")
# chat.ask(r"please generate the grounded image caption as details as possible")
# instruction = r"Identify the object in the spatial location of {<44><0><64><26>}"
# chat.ask(instruction)
a, a_token = chat.answer(max_new_tokens=200)
print(a)
visualize_all_bbox_together(image_path, a)
chat.reset()


# In[ ]:





# In[67]:


image_path = "/ibex/project/c2133/test_images/test_img5_720.jpg"
chat = Chat(model, vis_processor)
# chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
chat.conv.system=""
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)
instruction = r"tell me the bounding box location of the camera"

# instruction = r"Identify the object in the spatial location of {<0><3><14><59>}"
# instruction =r"please generate the grounded image caption"
# instruction = r" what is it in bounding box location {<46><21><58><36>}"
# instruction =r"is the squarrel on the left of the bird?"
chat.ask(instruction)
a, a_token = chat.answer(max_new_tokens=200)
print("minigpt-4: ",a)
visualize_all_bbox_together(image_path, a)

# chat.ask("please describe this image in details")
# a, a_token = chat.answer(max_new_tokens=50)
# print("minigpt-4: ",a)

# chat.ask("[bbox] tell me the location of the camera")
# a, a_token = chat.answer(max_new_tokens=50)
# visualize_all_bbox_together(image_path, a)

# print("minigpt-4: ",a)
# instruction = r" Describe this image"
# chat.ask(instruction)
# a, a_token = chat.answer(max_new_tokens=30)

# print("gt: ", sample_object)
# print("minigpt-4: ",a)
# print(chat.conv)
chat.reset()


# In[ ]:





# In[46]:


instruction


# In[63]:


chat = Chat(model, vis_processor)
chat.conv.system=""
image_path = "/ibex/project/c2133/test_images/xi.jpeg"
image = Image.open(image_path).convert("RGB")
chat.upload_img(image)


# instruction = r"Identify the object in the spatial location of {<6><14><29><63>}"
# instruction = r"the man is on the left side of the flag or right side"
# instruction = r"please generate the grounded image caption"
# chat.ask(r"please generate the grounded image caption")
# chat.ask(r"provide the description of the image along with the localized region ")
# chat.ask(r"generate an image caption incorporating the grounded phrase within the bounding box ")
# chat.ask(r"generate an image caption including the corresponding grounded bounding box phrase")
# chat.ask(r"generate a short image caption with the phrase-grounded bounding boxes")
chat.ask(r" what is it in bounding box location {<7><14><34><63>}")
# chat.ask(instruction)

a, a_token = chat.answer(max_new_tokens=100)
print(a)
visualize_all_bbox_together(image_path, a)
# chat.ask("please describe this image in details ")
# a, a_token = chat.answer(max_new_tokens=40)
# print(a)
# print(chat.conv)

chat.reset()


# In[ ]:





# In[ ]:






# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[194]:





# In[442]:





# In[443]:





# In[ ]:





# In[ ]:





# In[ ]:


# sample_bbox, sample_object, image_path = sample_one_bbox('/ibex/project/c2133/object_detection_datasets/coco_bbox/dataset/1')
# # print("the sampeld bbox", sample_bbox, "sample object", sample_object)
# chat = Chat(model, vis_processor)
# # chat.conv.system="<Img>ImageContent</Img>. Please answer my question. "
# chat.conv.system=""
# image = Image.open(image_path).convert("RGB")
# chat.upload_img(image)
# instruction = r"What is the object in the spatial location of "+str(sample_bbox) + "? "
# chat.ask(instruction)
# a, a_token = chat.answer(max_new_tokens=50)
# print("gt: ", sample_object)
# print("minigpt-4: ",a)
# chat.reset()


# In[244]:





# In[ ]:





# In[ ]:





# In[186]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset():
    chat.reset()
    return None


def gradio_ask(user_message, chatbot):
    chat.ask(user_message)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot


def gradio_answer(chatbot):
    llm_message = chat.answer(1000)
    chatbot[-1][1] = llm_message
    return chatbot


def gradio_upload_img(gr_img, chatbot):
    llm_message = chat.upload_img(gr_img)
    chatbot = chatbot + [[(gr_img,), None]]
    chatbot[-1][1] = llm_message
    return chatbot


with gr.Blocks() as demo:
    gr.Markdown("## GPT-4 Mini")
    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="filepath")
            upload = gr.Button("Upload Image")
            clear = gr.Button("Restart")
        with gr.Column():
            chatbot = gr.Chatbot()
            text_input = gr.Textbox()

    text_input.submit(gradio_ask, [text_input, chatbot], [text_input, chatbot], queue=False).then(
        gradio_answer, chatbot, chatbot
    )

    upload.click(gradio_upload_img, [image, chatbot], chatbot)
    clear.click(gradio_reset, None, chatbot, queue=False)


# In[6]:


demo.launch(share=True)


# In[8]:


demo.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Alignment Dataset Prepare

# In[256]:


import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
openai.api_key = ''


def prepare_chatgpt_message(task_prompt, paragraph):
    messages = [{"role": "system", "content": task_prompt},
                {"role": "user", "content": paragraph}]
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, max_tokens=200, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.7, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens


# In[413]:


import webdataset as wds
from lavis.datasets.datasets.base_dataset import BaseDataset
class PILDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }
    
vis_processor_cfg = cfg.datasets_cfg.cc_combine.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

text_processor_cfg = cfg.datasets_cfg.cc_combine.text_processor.train
text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)

dataset = PILDataset(vis_processor, text_processor, cfg.datasets_cfg.cc_combine.build_info.storage).inner_dataset


# In[332]:


from tqdm import tqdm


# In[291]:


cfg.datasets_cfg.laion.build_info.storage


# In[408]:


builder = registry.get_builder_class('cc_combine')(cfg.datasets_cfg['cc_combine'])
dataset = builder.build_datasets()['train']


# In[416]:


data_iter = iter(dataset)


# In[417]:


data_point = next(data_iter)


# In[418]:


image = data_point['image']


# In[419]:


image


# In[324]:


data_point['text_input']


# In[381]:


save_dir = '/ibex/project/c2133/blip_dataset/image_alignment'


image_list = []
text_list = []
description_list = []
negative_list = []
verify_list = []






for i in tqdm(range(20)):
    data_point = next(data_iter)
    image = data_point['image']
    text = data_point['text_input']
    
    fix_prompt = \
    "Fix the error in given paragraph. " \
    "Remove any repeating sentences, meanless characters, not English sentences, and so on." \
    "Rewrite any incompleted sentences." \
    "Return directly the results WITHOUT explaination." \
    "Return directly the input paragraph if it is already correct WITHOUT explaination."

    answers = []
    answer_tokens = 0
    chat.reset()
    chat.upload_img(image)
    chat.ask("Describe this image in detail. Give as many details as possible. Say everything you see.")
    answer, tokens = chat.answer()
    answers.append(answer)
    answer_tokens += tokens
    if len(answer_tokens) < 80:
        chat.ask("Continue")
        answer, answer_token = chat.answer()
        answers.append(answer)
        answer_tokens += tokens
    answer = ' '.join(answers)

    chatgpt_message = prepare_chatgpt_message(fix_prompt, answer)
    improved_answer, num_token = call_chatgpt(chatgpt_message)
    
    if 'is already correct' in improved_answer:
        improved_answer = answer
    if 'incomplete' in improved_answer or len(improved_answer) < 50:
        negative_list.append(improved_answer)
    else:
        image_list.append(image)
        text_list.append(text)
        description_list.append(improved_answer)


# In[329]:


answer


# In[330]:


improved_answer


# In[389]:


description_list[2]


# In[383]:


negative_list


# In[372]:


len('The input paragraph is incomplete.')


# In[420]:


save_dir = '/ibex/project/c2133/blip_dataset/image_alignment_cc'
texts = {}
for i in tqdm(range(5000)):
    data_point = next(data_iter)
    image = data_point['image']
    texts[i] = data_point['text_input']
    image.save(os.path.join(save_dir, "image/{}.jpg".format(i)))


# In[394]:





# In[397]:


import json


# open a file in write mode
with open(os.path.join(save_dir,"old_cap.json"), "w") as outfile:
    # write the dictionary to the file in JSON format
    json.dump(texts, outfile)


# In[ ]:




