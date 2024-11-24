import json
import argparse
import time
from threading import Thread
from PIL import Image
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from minigpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            ret = self.system + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + self.sep2
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if input_ids.shape[1] >= len(stop):  # Ensure the input is long enough
                if (input_ids[:, -len(stop):] == torch.tensor(stop).to(input_ids.device)).all(dim=1).any():
                    return True
        return False


CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    conv_id = "Vicuna",
)

CONV_VISION_Qwen = Conversation(
    system="<|im_start|>system\nGiven the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.<|im_end|>",
    roles=("\n<|im_start|>user\n", "\n<|im_start|>assistant\n"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.TWO,
    sep="",
    sep2="<|im_end|>",
    conv_id = "Qwen",
)

CONV_VISION_Qwen_2_5 = Conversation(
    system="<|im_start|>system\n"
                "You are a multi-modality upgraded Large Language Model. You are able to understand images given to you. "
                "Please do your best to answer the user's queries in association with the image that the users provide should the image be provided. "
                "The user may speak to you in a different language. You MUST answer in the language or languages spoken by the user. "
                "You have been trained to be able to see images very well. You have also been trained to understand and speak multiple language very well. " 
                "Do not say whether you can or cannot see the image and immediately respond to the user's query.<|im_end|>\n",
    roles=("\n<|im_start|>user\n", "\n<|im_start|>assistant\n"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.TWO,
    sep="",
    sep2="<|im_end|>",
    conv_id = "Qwen2.5",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
    conv_id = "Llama2",
)

CONV_VISION_LLama3 = Conversation(
    system="<|begin_of_text|><|start_header_id|><|start_header_id|>system<|end_header_id|>\n"
                   "You are a multi-modality upgraded Large Language Model. You are now able to understand images given to you. "
                   "Please do your best to answer the user's queries in association with the image that the users provide. "
                   "The user may speak to you in a different language. You MUST answer in the language or languages spoken by the user. "
                   "You have been trained to be able to see images very well. You have also been trained to understand and speak multiple language very well. " 
                   "Do not say whether you can or cannot see the image and immediately respond to the user's query. <|eot_id|>",
    roles=("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="<|eot_id|>",
    conv_id = "Llama3",
)

CONV_VISION_Aya23 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.TWO,
    sep="",
    sep2="<|END_OF_TURN_TOKEN|>",
    conv_id = "Aya23",
)

CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
    conv_id = "Minigptv2",
)

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        hash_token_ids = self.model.llama_tokenizer.encode("###", add_special_tokens=False)

        if stopping_criteria is not None:
            self.stopping_criteria = stopping_criteria
        else:
            # Attempt to get EOS and EOT tokens
            eos_token_id = getattr(model.llama_tokenizer, 'eos_token_id', None)
            eot_token_id = getattr(model.llama_tokenizer, 'eot_token_id', None)

            # Prepare stop tokens list, fallback to "###" if necessary
            stop_words_ids = []
            if eos_token_id is not None:
                stop_words_ids.append([eos_token_id])
            else:
                stop_words_ids.append(hash_token_ids)

            if eot_token_id is not None:
                stop_words_ids.append([eot_token_id])
            else:
                stop_words_ids.append(hash_token_ids)
            
            self.stopping_criteria = StoppingCriteriaList([
                StoppingCriteriaSub(stops=stop_words_ids)
            ])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    def answer(self, conv, img_list, **kargs):
        generation_dict = self.answer_prepare(conv, img_list, **kargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def stream_answer(self, conv, img_list, **kargs):
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output

    def encode_img(self, img_list):
        image = img_list[0]
        img_list.pop(0)
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
        img_list.append(image_emb)

    def upload_img(self, image, conv, img_list):
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        img_list.append(image)
        msg = "Received."

        return msg

    def upload_img_with_tag(self, image, conv, img_list, tag):
        conv.append_message(conv.roles[0], f"<Img>{tag}</Img>")
        img_list.append((tag, image))
        return "Received."

    def ask_middle(self, text, conv):
        if not conv.messages or conv.messages[-1][0] != conv.roles[0]:
            conv.append_message(conv.roles[0], text)
        else:
            conv.messages[-1][1] += " " + text

    def encode_img_list(self, img_list):
        encoded_images = []
        for tag, image in img_list:
            if isinstance(image, str):
                raw_image = Image.open(image).convert('RGB')
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, Image.Image):
                image = self.vis_processor(image).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)

            image_emb, _ = self.model.encode_img(image)
            encoded_images.append((tag, image_emb))
        return encoded_images

    def answer_prepare_multi(self, conv, encoded_img_list, max_new_tokens=300, **kwargs):
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        for tag, img_emb in encoded_img_list:
            prompt = prompt.replace(f"<Img>{tag}</Img>", "<ImageHere>")
        
        embs = self.model.get_context_emb(prompt, [img_emb for _, img_emb in encoded_img_list])

        current_max_len = embs.shape[1] + max_new_tokens
        max_length = kwargs.get('max_length', 2000)
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            **kwargs
        )
        return generation_kwargs

    def answer_multi(self, conv, encoded_img_list, **kwargs):
        generation_dict = self.answer_prepare_multi(conv, encoded_img_list, **kwargs)
        output_token = self.model_generate(**generation_dict)[0]
        output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def stream_answer_multi(self, conv, encoded_img_list, **kwargs):
        generation_kwargs = self.answer_prepare_multi(conv, encoded_img_list, **kwargs)
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer