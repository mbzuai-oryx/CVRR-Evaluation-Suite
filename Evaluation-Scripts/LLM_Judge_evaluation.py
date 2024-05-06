import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import time


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--cvrr_dataset_path", required=True, help="Folder path to CVRR-ES dataset..")
    parser.add_argument("--output_dir", required=True, help="The path to save prediction json files.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, filename, output_dir):
    """
    Evaluates question and answer pairs using GPT-3.5
    Returns a score, prediction (correct/incorrect) and a reason for reach Question-Answer pair in CVRR.
    """
    # Check if json file already present for the current evaluation dimension
    save_path = f"{output_dir}/{filename}.json"
    if os.path.exists(save_path):
        print(f"Skipping... LLM Judge results already present for dimension: {filename}")
        return
    result_qa_pair = []
    for single_dict in prediction_set:
        question = single_dict['q']
        answer = single_dict['a']
        pred = single_dict['pred']
        while True:
            try:
                # Compute the correctness score
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
                                "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
                                "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n"
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Ground truth correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                                "Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is  a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should providethe reason behind the decision."
                                "Only provide the Python dictionary string."
                                "For example, your response should look like this: {'pred': 'correct', 'score': 4.8, 'reason': reason}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
                response_message = completion["choices"][0]["message"]["content"]
                try:
                    response_dict = ast.literal_eval(response_message)
                except:
                    # Remove the special characters.
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
                time.sleep(60)

    # Save the question-answer pairs to a json file.
    with open(f"{output_dir}/{filename}.json", "w") as f:
        json.dump(result_qa_pair, f)


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    all_folder_names = os.listdir(args.cvrr_dataset_path)
    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    # Loop on each video dimension
    for single_folder in all_folder_names:  # all folders one by one
        gt_file_path = os.path.join(os.path.join(args.cvrr_dataset_path, single_folder),
                                    "annotations_" + single_folder + ".json")
        pred_file_path = args.pred_path + "/" + single_folder + ".json"
        # Loop through each file in the folder
        my_list = []
        json_gt = json.load(open(gt_file_path))
        json_pred = json.load(open(pred_file_path))
        assert len(json_gt) == len(json_pred)
        for dict_pred, dict_gt in zip(json_pred, json_gt):  # All QA for a single video
            question = dict_gt['Q']
            answer = dict_gt['A']
            pred = dict_pred['A']
            if pred == "":
                print(f"Skipping 1 QA pair due to empty prediction in dimension {single_folder}")
                continue
            qa_set = {"q": question, "a": answer, "pred": pred}
            my_list.append(qa_set)
        prediction_set[single_folder] = my_list
    sum = 0
    for single_key in prediction_set.keys():
        sum += len(prediction_set[single_key])
    print(f"Total number of QA pairs are: {sum}")

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    # While loop to ensure that all QA pairs are processed.
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

            task_args = [(prediction_set[filename], filename, args.output_dir) for filename in incomplete_files]
            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
    print("All evaluation completed!")


if __name__ == "__main__":
    main()
