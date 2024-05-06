## Comparing the Predicted Answers with Ground-Truth Answers using LLM-Assisted evaluation

This readme provides detailed instructions on producing final evaluation scores of Video-LMMs for the CVRR-ES benchmark. Make sure that the initial step of generating predictions has been completed as described in [PREDICTIONS.md](PREDICTIONS.md). 

**How the accuracy scores are obtained?** We provide Video-LMM predictions and the corresponding ground-truth answers to the LLM Judge alongside the evaluation prompt. The Judge determines whether the prediction is correct or incorrect through a binary judgment, assigns a score from 1 to 5 representing the quality of the prediction, and provides a reasoning to explain its decision. Finally, we sum all correct binary judgements as the final accuracy for each evaluation dimension.

### Evaluation of Video-LMMs using LLM as the Judge
Follow the steps below to produce the evaluation scores.

1) Run the following commands to complete the LLM (GPT-3.5 in our experiments) based evaluation.
```shell
# Install the openai package
pip install openai==0.28.0

cd Evaluation-Scripts

# LLM based evaluation
python LLM_Judge_evaluation.py --pred_path <predictions-folder-path>\
--cvrr_dataset_path <path-to-CVRR_ES-folder> \
--output_dir <folder-path-to-save-judge-results> \
--api_key <your-open-ai-key>
```

The above command will generate LLM-based results for each evaluation dimension of CVRR-ES benchmark.

2) Now run the below command to aggregate indiviudal QA scores and print final results. 
```shell
# LLM based evaluation
python results_parser.py --judge_results_path <path-to-folder-obtained-from-step-1>
```

This should print the final results. Sample output is given as:
```shell
# python results_parser.py --judge_results_path  ./Judge-results/standard_prompting/Gemini_Vision_Pro/
Accuracy for evaluation dimension [Interpretation of visual context]: 63.00
Average score for evaluation dimension [Interpretation of visual context]: 3.54
Accuracy for evaluation dimension [Non-existent actions with non-existent scene depictions]: 49.64
Average score for evaluation dimension [Non-existent actions with non-existent scene depictions]: 3.11
Accuracy for evaluation dimension [Interpretation of social context]: 64.29
Average score for evaluation dimension [Interpretation of social context]: 3.56
Accuracy for evaluation dimension [Partial actions]: 67.48
Average score for evaluation dimension [Partial actions]: 3.68
Accuracy for evaluation dimension [Non-existent actions with existent scene depictions]: 57.25
Average score for evaluation dimension [Non-existent actions with existent scene depictions]: 3.37
Accuracy for evaluation dimension [Fine-grained action understanding]: 51.61
Average score for evaluation dimension [Fine-grained action understanding]: 3.21
Accuracy for evaluation dimension [Continuity and Object Instance Count]: 36.16
Average score for evaluation dimension [Continuity and Object Instance Count]: 2.66
Accuracy for evaluation dimension [Time order understanding]: 45.39
Average score for evaluation dimension [Time order understanding]: 3.20
Accuracy for evaluation dimension [Understanding of emotional context]: 47.26
Average score for evaluation dimension [Understanding of emotional context]: 2.95
Accuracy for evaluation dimension [Unusual and Physically Anomalous activities]: 60.00
Average score for evaluation dimension [Unusual and Physically Anomalous activities]: 3.38
Accuracy for evaluation dimension [Multiple actions in a single video]: 43.08
Average score for evaluation dimension [Multiple actions in a single video]: 2.99
```
