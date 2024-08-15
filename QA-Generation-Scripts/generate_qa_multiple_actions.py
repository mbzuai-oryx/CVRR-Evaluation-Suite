# Required Libraries
import openai
import os
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Generate question-answer pairs using GPT-3.")
    parser.add_argument("--evaluation_dimension_path_name", required=True, help="Path to the ground truth captions file.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    return parser.parse_args()


def annotate(caption_content, output_dir):
    """
    Generates question and answer pairs based on video captions using OpenAI GPT-3.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and intelligent AI assistant which can curate "
                            "high-quality and challenging question and corresponding answers "
                            "used to test the video understanding capabilities of an AI Video system."
            },
            {
                "role": "user",
                "content":
                    "Given a video which contains multiple actions, with the following detailed caption explaining the events: "
                    f"The caption is: {caption_content}."
                    "Formulate 10 diverse questions to test, whether the model can identify the different actions present in the video and also their sequence order."
                    "Also consider asking about the presence of confusing, misleading and similar actions that could most likely occur in the video."
                    "Generate questions that comprise both interrogative and declarative sentences, utilizing different language styles, and provide an explanation for each."
                    "Your response should be presented as a list of dictionary strings with keys 'Q' for questions and 'A' for the answer."
                    "Follow these rules while generating question and answers:"
                    "1. Avoid explicitly mentioning the type of assessment in the reasoning."
                    "2. Do not provide answers in the question itself. For example, the ground-truth attribute or component which makes the answer easy should never be mentioned in the question itself."
                    "3. Ensure the questions are concrete and can be entirely addressed using the provided caption."
                    "4. Do not ask questions whose answer cannot be obtained in the caption."
                    "5. Do not formulate such questions whose answer is not specified in the video & caption."
                    "6. Do not ask vague questions and provide to the point answers with reasoning. "
                    "7. Do not mention that the answer is based on provided captions or descriptions."
                    "For example, format your response as follows: "
                    "[{\"Q\": 'Your first question here...', \"A\": 'Your first answer here...'}, "
                    "{\"Q\": 'Your second question here...', \"A\": 'Your second answer here...'}, "
                    "{\"Q\": 'Your third question here...', \"A\": 'Your third answer here...'}]."
            }
        ]
    )

    # Extract and save the QA pairs
    response_message = response["choices"][0]["message"]["content"]
    response_dict = ast.literal_eval(response_message)

    json_file_path = output_dir + ".json"
    with open(json_file_path, "w") as f:
        json.dump(response_dict, f)

    print(f"Completed, annotations saved in {json_file_path}")


def main():
    """
    Main function to control the flow of the program.
    """
    args = parse_args()
    openai.api_key = args.api_key

    # Process captions
    files = os.listdir(args.single_category_path_name)
    text_files = [f for f in files if f.endswith(".txt")]
    json_files = {f.split('.json')[0] for f in files if f.endswith(".json")}
    incomplete_files = [f for f in text_files if f.split(".")[0] not in json_files]

    print(f"Completed files: {len(json_files)}")
    print(f"Incomplete files: {len(incomplete_files)}")

    for single_video_caption in incomplete_files:
        caption_path = os.path.join(args.single_category_path_name, single_video_caption)
        with open(caption_path) as f:
            caption = f.readline().strip()

        if isinstance(caption, str):
            while True:
                try:
                    output_dir = os.path.join(args.single_category_path_name, single_video_caption.replace(".txt", ""))
                    annotate(caption, output_dir)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print("Sleeping for 10 seconds...")
                    time.sleep(10)


if __name__ == "__main__":
    main()
