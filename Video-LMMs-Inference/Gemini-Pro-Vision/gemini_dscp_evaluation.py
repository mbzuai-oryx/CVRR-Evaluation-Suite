import time
import os
import json
import argparse
from tqdm import tqdm
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

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

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--cvrr_dataset_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--google_cloud_bucket_name", required=True, help="Bucket name. For Gemini, CVRR-ES dataset needs to be also uploaded to google cloud bucket.")
    parser.add_argument("--google_cloud_project_name", required=True, help="Bucket name. For Gemini, please provide the google cloud project name.")
    args = parser.parse_args()
    return args


def evaluate_single_video_dimension(cvrr_dataset_path, single_folder, output_dir, gcp_cloud_bucket_name,
                                    gcp_project_name):
    # Parse arguments.
    vertexai.init(project=gcp_project_name, location="us-central1")
    model = GenerativeModel("gemini-1.0-pro-vision-001")
    # Skip this if the json file already exists
    json_file_path = os.path.join(output_dir, single_folder + '.json')
    if os.path.exists(json_file_path):
        return
    print(f"Generating Gemini-Pro-Vision predictions on CVRR-ES benchmark for dimension: {single_folder}")
    annotation_path = os.path.join(cvrr_dataset_path,
                                   single_folder + "/" + "annotations_" + single_folder + ".json")
    qa_pairs = json.load(open(annotation_path, "r"))  # list of dictionaries
    # iterate over each question
    model_response = []
    for single_dict in tqdm(qa_pairs):
        user_question = single_dict["Q"]
        # Load the video
        # The dataset must be additionally uploaded to google cloud bucket
        my_path = f"gs://{gcp_cloud_bucket_name}/" + "CVRR-ES/" + single_folder + "/" + single_dict['VideoID']
        video_part = Part.from_uri(
            my_path, mime_type="video/mp4"
        )
        message = False
        while True:
            try:
                result = model.generate_content(
                    [video_part, reasoning_prompt,],
                    generation_config={
                        "max_output_tokens": 2048,
                        "temperature": 1,
                        "top_p": 1,
                        "top_k": 32
                    },
                    safety_settings={
                        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    },
                )
                my_context = result.text
                # Now proceed with the second

                user_question = f"Context for the given video is: {my_context}. \n Now answer a question truthfully based on the video and the provided context. Question: " + \
                                single_dict["Q"]

                result = model.generate_content(
                    [video_part, user_question, ],
                    generation_config={
                        "max_output_tokens": 2048,
                        "temperature": 1,
                        "top_p": 1,
                        "top_k": 32
                    },
                    safety_settings={
                        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    },
                )
                answer = result.text
                break
            except Exception as e:
                print(f"Error: {e}, sleeping for 60 sec")
                time.sleep(60)
                try:
                    message = result.to_dict()['prompt_feedback']['block_reason'] >= 0
                    if message:
                        print(result.to_dict()['prompt_feedback'])
                        print(f"gemini has blocked qa for video {my_path}")
                        break
                except Exception as e:
                    print(e)
                    print("Retrying for the same prompt")

            if message:
                model_response.append({"Q": single_dict["Q"], "A": ""})
            else:
                model_response.append({"Q": user_question, "A": answer})

    # Save the response dictionary into a JSON file
    with open(json_file_path, "w") as f:
        json.dump(model_response, f)

def main():
    """
    Main function to control the flow of the program.
    """
    args = parse_args()
    all_folder_names = os.listdir(args.cvrr_dataset_path)
    output_dir = args.output_dir
    # Create output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for single_folder in all_folder_names:
        evaluate_single_video_dimension(args.cvrr_dataset_path, single_folder, args.output_dir,
                                        args.google_cloud_bucket_name, args.google_cloud_project_name)
    print("Inference with Gemini-Pro-Vision model completed!")

if __name__ == "__main__":
    main()