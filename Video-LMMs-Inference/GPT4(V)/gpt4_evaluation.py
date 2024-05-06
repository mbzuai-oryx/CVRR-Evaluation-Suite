import cv2
import base64
import time
import numpy as np
from openai import OpenAI
import os
import json
import argparse
from tqdm import tqdm

# Set the OpenAI API key.
video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--cvrr_dataset_path", required=True, help="Folder path to CVRR-ES dataset.")
    parser.add_argument("--output_dir", required=True, help="The path to save prediction json files.")
    parser.add_argument("--api_key", required=True, help="openai api key")
    args = parser.parse_args()
    return args

# As GPT4V does not inherently support videos, 
# so it requires an additional prompt to enable video question answering
GPTV_for_video_message = 'These are frames from a video that I want to upload. Use the visual cues to answer the question a question. You need to answer the question in any case and not demand additional context information. Remember the answer to the question is always present in the given frames, so you have to answer in any case without refusing.\n'

def evaluate_videos_single_folder(cvrr_dataset_path, single_folder, output_dir, client):
    annotation_path = os.path.join(cvrr_dataset_path,
                                   single_folder + "/" + "annotations_" + single_folder + ".json")
    qa_pairs = json.load(open(annotation_path, "r"))  # list of dictionaries
    # Skip this if the json file already exists
    json_file_path = os.path.join(output_dir, single_folder + '.json')
    if os.path.exists(json_file_path):
        return
    print(f"Generating GPT4-(V)ision predictions on CVRR-ES benchmark for dimension: {single_folder}")
    # iterate over each question
    model_response = []
    for index, single_dict in enumerate(tqdm(qa_pairs)):
        user_question = single_dict["Q"]
        # Load the video
        video_path = os.path.join(os.path.join(cvrr_dataset_path, single_folder), single_dict['VideoID'])
        video = cv2.VideoCapture(video_path)
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()
        # Sample only 8 frames in uniform order
        num_frames = 8
        duration = len(base64Frames)
        frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
        sampled_base64 = [base64Frames[i] for i in frame_id_list]
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    f"{GPTV_for_video_message}"
                    f"Question:{user_question}",
                    *map(lambda x: {"image": x, "resize": 768}, sampled_base64),
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }
        while True:
            try:
                result = client.chat.completions.create(**params)
                break
            except Exception as e:
                print(f"Error: {e}, sleeping for 60 sec")
                time.sleep(60)
        model_response.append({"Q": user_question, "A": result.choices[0].message.content})
        print(f"{index} completed thankfully")
    # Save the single evaluation dimension dictionary into a JSON file
    with open(json_file_path, "w") as f:
        json.dump(model_response, f)

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    all_folder_names = os.listdir(args.cvrr_dataset_path)
    output_dir = args.output_dir
    # Create output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    api_key = args.api_key
    client = OpenAI(api_key=api_key)
    for single_folder in all_folder_names:
        evaluate_videos_single_folder(args.cvrr_dataset_path, single_folder, args.output_dir, client)
    print("Inference with GPT vision model completed!")


if __name__ == "__main__":
    main()

