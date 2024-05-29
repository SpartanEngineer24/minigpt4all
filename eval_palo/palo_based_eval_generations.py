import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Qwen, StoppingCriteriaList, StoppingCriteriaSub


def initialize_model(cfg_path, ckpt_path):
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config.")
    args = parser.parse_args(['--cfg-path', cfg_path])

    cfg = Config(args)
    cfg.model_cfg.ckpt = ckpt_path
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    return model, cfg


def create_chat_instance(model, vis_processor):
    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:0') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    chat = Chat(model, vis_processor, device='cuda:0', stopping_criteria=stopping_criteria)
    return chat


def eval_model(args):
    model, cfg = initialize_model(args.cfg_path, args.ckpt_path)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    if args.debug:
        line = {"question_id": 1, "image": "test_image.jpg", "text": "Describe this image"}
        questions = [line]
        answers_file = os.path.expanduser("debug_answer.jsonl")
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        answers_file = os.path.expanduser(args.answers_file)

    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    ans_file = open(answers_file, "a", encoding="utf-8")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        if args.debug:
            image_path = os.path.expanduser("test_image.jpg")
        else:
            image_path = os.path.join(args.image_folder, image_file)

        image = Image.open(image_path).convert("RGB")

        chat = create_chat_instance(model, vis_processor)
        chat_state = deepcopy(CONV_VISION_Qwen)
        img_list = []
        chat.upload_img(image, chat_state, img_list)
        chat.encode_img(img_list)

        chat.ask(qs, chat_state)
        reply = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=args.num_beams,
                            temperature=args.temperature,
                            max_new_tokens=300,
                            max_length=2000)[0]

        if args.debug:
            print(f"Debug Answer: {reply}")
            break  # only run once in debug mode

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": reply,
                                   "answer_id": idx,
                                   "model_id": "minigpt4",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    print("Initializing Process")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--debug", action='store_true', help="enable debugging mode")
    args = parser.parse_args()

    eval_model(args)
    print("Finished Process")