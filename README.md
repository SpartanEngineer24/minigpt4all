# minigpt4all
Multilingual Finetuning of the MiniGPT4 project

Currently this repository is a work in progress.
The main files changed from the original MiniGPT4 project relevant for current debugging are:

minigpt4>

  conversation>
  
    conversation.py
    
  datasets>
  
    builders>
    
      image_text_pair_builder.py
      
    datasets>
    
      coco_multi.py
      gqa_multi.py
      ocrvqa_multi.py
      vg_multi.py
      textvqa_multi.py
      
  models>
  
    *base_model.py*
    *minigpt_base.py*
    *minigpt4.py*
    
  train_configs>
  
    minigpt4_qwen_14_stage3.yaml
    
