import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')
from tqdm import tqdm
# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/timechat.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cvrr_dataset_path', help='Directory containing folders for each benchmark category.',
                        required=True)
    parser.add_argument('--output_dir', help='path where you want to save all output responses.', required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


print('Initializing Chat')
args = parse_args()
cfg = Config(args)

# Please set up the following path
DIR="/PATH/TO/ckpt/timechat"
MODEL_DIR=f"{DIR}/timechat_7b.pth"

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = MODEL_DIR
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
main_path = args.all_dimension_folder_path
all_folders = os.listdir(main_path)

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
for single_folder in all_folders:
    print(f"Generating predictions for evaluation dimension: {single_folder}")
    json_file_path = args.output_dir + "/" + "/" + single_folder + ".json"
    # Skip this evaluation dimension if results are already present
    if os.path.exists(json_file_path):
        print(f"Skipping prediction generation for evaluation dimension as json file already exists: {single_folder}")
        continue 
    final_path = os.path.join(main_path + single_folder + "/" + "annotations_" + single_folder + ".json")
    qa_pairs = json.load(open(final_path, "r")) # list of dictionaries
    # iterate over each question
    model_response = []
    for single_dict in tqdm(qa_pairs):
        # Get the video path
        video_path = os.path.join(os.path.join(main_path, single_folder), single_dict['VideoID'])
        # CVRR-ES Question
        user_question = single_dict["Q"]
        # TimeChat default code lines
        img_list = []
        chat_state = conv_llava_llama_2.copy()  # Every time, previous history will be erased.
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        msg = chat.upload_video_without_audio(
            video_path=video_path, 
            conv=chat_state,
            img_list=img_list, 
            n_frms=96,
        )
        chat.ask(user_question, chat_state)
        num_beams = args.num_beams
        temperature = args.temperature
        # Generate answer for the question from Video LMM
        try:
            llm_message = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=512,
                                    max_length=2000)[0]

            outputs = llm_message.strip()
            model_response.append({"Q": user_question, "A": outputs})
        except Exception as e:
            print(f"Error processing video file '{video_path}': {e}")
                
        # Save the response dictionary into a JSON file
        with open(json_file_path, "w") as f:
            json.dump(model_response, f)

