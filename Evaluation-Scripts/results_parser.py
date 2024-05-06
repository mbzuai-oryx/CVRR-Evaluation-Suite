import os
import json
import argparse

proper_names = {
    "non_existent_actions_with_non_existent_scene_depictions": 'Non-existent actions with non-existent scene depictions',
    "interpretation_of_social_context": 'Interpretation of social context',
    "non_existent_actions_with_existent_scene_depictions": 'Non-existent actions with existent scene depictions',
    'multiple_actions_in_a_single_video': 'Multiple actions in a single video',
    'unusual_and_physically_anomalous_activities': 'Unusual and Physically Anomalous activities',
    'partial_actions': 'Partial actions',
    'understanding_emotional_context': 'Understanding of emotional context',
    'fine_grained_action_understanding': 'Fine-grained action understanding',
    "interpretation_of_visual_context": 'Interpretation of visual context',
    'continuity_and_object_instance_count': "Continuity and Object Instance Count",
    "time_order_understanding": 'Time order understanding'
}


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--judge_results_path", required=True,
                        help="The path to the folder containing evaluation results from Judge LLM.")
    args = parser.parse_args()
    return args


def parse_results(judge_results_path):
    all_json_files = os.listdir(judge_results_path)
    for single_file_name in all_json_files:
        with open(os.path.join(judge_results_path, single_file_name), "r") as json_file:
            contents = json.load(json_file)
        # Calculate average score and accuracy
        score_sum = 0
        count = 0
        yes_count = 0
        no_count = 0
        for single_content in contents:
            # Computing score
            count += 1
            try:
                score_match = single_content[0]['score']
            except:
                score_match = single_content[0][0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            try:
                pred = single_content[0]['pred']
            except:
                try:
                    pred = single_content[0][0]['pred']
                except:
                    pred = single_content[0]['type']

            if pred.lower() == "correct":
                yes_count += 1
            elif pred.lower() == "incorrect":
                no_count += 1
            else:
                print("Error occurred during counting predictions")

        average_score = score_sum / count
        accuracy = yes_count / (yes_count + no_count)
        print(f"Accuracy for evaluation dimension [{proper_names[single_file_name.split('.json')[0]]}]: "
              f"{accuracy * 100:.2f}")
        print(f"Average score for evaluation dimension [{proper_names[single_file_name.split('.json')[0]]}]: "
              f"{average_score:.2f}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    parse_results(args.judge_results_path)


if __name__ == "__main__":
    main()
