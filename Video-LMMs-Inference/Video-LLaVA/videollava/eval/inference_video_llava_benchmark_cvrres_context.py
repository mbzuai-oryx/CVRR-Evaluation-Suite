import torch
import sys
import os
sys.path.append(os.getcwd())
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from tqdm import tqdm
import json
import argparse

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvrr_dataset_path', help='Directory containing folders for each benchmark category.',
                        required=True)
    parser.add_argument('--output_dir', help='path where you want to save all output responses.', required=True)
    return parser.parse_args()

reasoning_prompt = """
As an intelligent video comprehension model, focus on these guidelines:

1. Differentiate recurring objects, count accurately, and identify movements and poses. \n
2. Understand directional movements and temporal order. \n
3. Pay attention to fine-grained actions with precision. \n
4. Assess incomplete actions without assuming completion. \n
5. Detect emotional, social, and visual cues. \n
6. Capture and analyze all relevant actions. \n
7. Identify unusual actions accurately. \n
8. Disagree with incorrect information given in question. \n
9. If you do not find the evidence in the frames, you can give a definite answer by assuming that the asked action/attribute is not present. \n
10. Provide to the point and concise response. \n
Now, proceed with answering the following question faithfully while keeping above guidelines in mind: \n 
Question: What is happening in the video?"""

def main(args):
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    main_path = args.all_dimension_folder_path
    all_folders = os.listdir(main_path)
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
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
            # Load the video
            video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
            if type(video_tensor) is list:
                tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
            else:
                tensor = video_tensor.to(model.device, dtype=torch.float16)
            # Video LLaVA default code lines
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles
            # First iteration of DSCP prompting
            inp = reasoning_prompt
            # Video LLaVA default code lines
            inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # Generate answer for the question from Video LMM
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            my_context = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().strip('\n')
            # Now second iteration of the DSCP prompting technique
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles
            inp = f"Context for the given video is: {my_context}. \n Now answer a question truthfully based on the video and the provided context. Question: " + single_dict["Q"]
            inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            model_response.append({"Q": single_dict["Q"], "A": outputs})
            
        # Save the response dictionary for a single dimension into a JSON file
        with open(json_file_path, "w") as f:
            json.dump(model_response, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)