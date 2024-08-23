import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import time


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--GT_json_path", required=True, help="The path where json files with GT are located.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args

def annotate(prediction_set, filename, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    result_qa_pair = []
    for single_dict in prediction_set:
        question = single_dict['q']
        answer = single_dict['a']
        while True:
            try:
                # Compute the correctness score
                completion = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an intelligent assistant designed to classify question-answer pairs based on the "
                                "reasoning and robustness definitions provided. "
                                "Your task is to analyze the given question-answer pair and classify it as either a 'Reasoning QA' or 'Robustness QA' pair. "
                                "Use the following definitions to guide your classification:\n"
                                "1) **Reasoning QA**: These are question-answer pairs designed to assess a Video-LMM's ability to understand videos beyond mere content recognition. They require the model to engage in complex reasoning by analyzing the relationships between activities and their context, \n"
                                " including counterfactual and hypothetical scenarios. The goal is to validate the model's capability to infer underlying rationales rather than just describing what is happening. \n"
                                "2) **Robustness QA**: These are question-answer pairs aimed at testing a Video-LMM's ability to maintain reliable performance when faced with misleading, confusing, or irrelevant questions. This also include questions which ask about non-existent objects as well as non-existent scene descriptions mentioned in questions. \n"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "Please classify the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Ground truth correct Answer: {answer}\n"
                                "Provide your classification as either 'Reasoning QA' or 'Robustness QA'. Additionally, provide a brief explanation for your classification. "
                                "Generate the response in the form of a Python dictionary string with keys 'classification' and 'reason'. "
                                "For example, your response should look like this: {'classification': 'Robustness QA', 'reason': 'your reason'}"
                            )
                        }
                    ]
                )

                # Convert response to a Python dictionary.
                response_message = completion["choices"][0]["message"]["content"]
                try:
                    response_dict = ast.literal_eval(response_message)
                except:
                    start_index = response_message.find("'reason': '") + len("'reason': '")
                    end_index = response_message.find("'", start_index)

                    # Extract the reason value
                    reason_value = response_message[start_index:end_index]

                    # Remove single quotes from the reason value
                    reason_value = reason_value.replace("'", "")

                    # Replace the original reason value with the modified one
                    response_message = response_message[:start_index] + reason_value + '\'}'
                    response_dict = ast.literal_eval(response_message)
                result_qa_pair.append([response_dict, single_dict])
                break
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
                time.sleep(10)

    # Save the question-answer pairs to a json file.
    with open(f"{output_dir}/{filename}.json", "w") as f:
        json.dump(result_qa_pair, f)


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    all_folder_names = os.listdir(args.GT_json_path)
    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    # Loop on each folder
    for single_folder in all_folder_names:  # all folders one by one
        gt_file_path = os.path.join(os.path.join(args.cvrr_dataset_path, single_folder),
                                    "annotations_" + single_folder + ".json")
        json_gt = json.load(open(gt_file_path))
        my_list = []
        for dict_gt in json_gt:  # All QA for a single video
            question = dict_gt['Q']
            answer = dict_gt['A']
            qa_set = {"q": question, "a": answer}
            my_list.append(qa_set)
        prediction_set[single_folder] = my_list
    sum = 0
    for single_key in prediction_set.keys():
        sum += len(prediction_set[single_key])
    print(f"Total number of QA pairs are: {sum}")

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in prediction_set.keys() if (f + ".json") not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            task_args = [(prediction_set[filename], filename, args.output_dir) for filename in prediction_set.keys()]
            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All Categorization completed!")


if __name__ == "__main__":
    main()
