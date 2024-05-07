## Generating Predictions for CVRR-ES dataset from Video-LMMs
This document provides detailed instructions on generating answers (predictions) from Video-LMMs for video questions in the CVRR-ES benchmark. 

**About the Benchmark:** CVRR-ES benchmark consists of 2400 open-ended question-answer (QA) pairs spanning over 214 unique videos (224 videos in total as some videos are used for multiple evaluation dimensions) for evaluating Video-LMMs. The benchmark aims to assess their robustness to user textual queries (e.g., confusing, misleading questions etc.) and reasoning capabilities in a variety of complex and contextual videos covering 11 diverse evaluation dimensions.

**How the predictions are generated?** For each QA pair of CVRR-ES benchmark, we provide Video-LMM with the question alongside with the corresponding video, which generates prediction answer in an auto-regressive manner. Each QA pair is processed without maintaining the chat history.

Below we provide separate instructions for each model for generating predictions.

### Generating Predictions using TimeChat
Follow the steps below to generate answers for TimeChat Video-LMM

1) First install the required packages and create the environment for TimeChat by following instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for TimeChat using standard prompting (i.e. by asking the question only)
```shell
cd Video-LMMs-Inference/TimeChat
```

```shell
# --cvrr_dataset_path is the path that points to the downloaded CVRR-ES folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python inference_benchmark_cvrres.py \
--cvrr_dataset_path <path-to-CVRR-ES-folder> \
--output_dir <folder-path-to-save-predictions>
```

The above command will generate predictions using the standard prompting method. 
In order to generate responses using Dual-Step Contextual Prompting (DSCP) method, run the following command:

```shell
# --cvrr_dataset_path is the path that points to the downloaded CVRR-ES folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python inference_benchmark_context_cvrres.py \
--cvrr_dataset_path <path-to-CVRR_ES-folder> \
--output_dir <folder-path-to-save-predictions>
```

The generated predictions will be saved into the folder specified by --output_dir argument.

### Generating Predictions using Video-LLaVA
Follow the steps below to generate answers for Video-LLaVA Video-LMM

1) First install the required packages and create the environment for Video-LLaVA by following instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for Video-LLaVA using standard prompting (i.e. by asking the question only)
```shell
cd Video-LMMs-Inference/Video-LLaVA
conda activate videollava
```

```shell
# --cvrr_dataset_path is the path that points to the downloaded CVRR-ES folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python videollava/eval/inference_video_llava_benchmark_cvrres.py \
--cvrr_dataset_path <path-to-CVRR_ES-folder> \
--output_dir <folder-path-to-save-predictions>
```

The above command will generate predictions using the standard prompting method. 
In order to generate responses using Dual-Step Contextual Prompting (DSCP) method, run the following command:

```shell
# --cvrr_dataset_path is the path that points to the downloaded CVRR-ES folder
# --output_dir can be any path to save the predictions
CUDA_VISIBLE_DEVICES=0 python videollava/eval/inference_video_llava_benchmark_cvrres_context.py \
--cvrr_dataset_path <path-to-CVRR_ES-folder> \
--output_dir <folder-path-to-save-predictions>
```

The generated predictions will be saved into the folder specified by --output_dir argument.

### Generating Predictions using Gemini-Pro-Vision
Follow the steps below to generate answers using Gemini-Pro-Vision.

1) First install the required packages and create the environment for Gemini by following instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for Gemini using standard prompting (i.e. by asking the question only)
```shell
cd Video-LMMs-Inference/Gemini-Pro-Vision
conda activate gemini
```

```shell
# --cvrr_dataset_path should point to CVRR-ES folder inside your google cloud bucket
# --output_dir can be any path to save the predictions
python gemini_base_evaluation.py \
--cvrr_dataset_path <path-to-CVRR_ES-folder-relative-to-google-cloud-bucket> \
--output_dir <local-folder-path-to-save-predictions> \
--google_cloud_bucket_name <name-of-your-google-cloud-bucket> \
--google_cloud_project_name <name-of-your-google-cloud-project>
```

The above command will generate predictions using the standard prompting method. 
In order to generate responses using Dual-Step Contextual Prompting (DSCP) method, run the following command:

```shell
# --cvrr_dataset_path should point to CVRR-ES folder inside your google cloud bucket
# --output_dir can be any path to save the predictions
python gemini_dscp_evaluation.py \
--cvrr_dataset_path <path-to-CVRR_ES-folder-relative-to-google-cloud-bucket> \
--output_dir <local-folder-path-to-save-predictions> \
--google_cloud_bucket_name <name-of-your-google-cloud-bucket> \
--google_cloud_project_name <name-of-your-google-cloud-project>
```

The generated predictions will be saved into the folder specified by --output_dir argument.



### Generating Predictions using GPT4-(V)ison
Follow the steps below to generate answers using GPT-4V.

1) First install the required packages and create the environment for GPT-4V by following instructions in [INSTALL.md](INSTALL.md).

2) Run the following commands to generate predictions for GPT-4V using standard prompting (i.e. by asking the question only)
```shell
    cd Video-LMMs-Inference/GPT4(V)
    conda activate gpt4v
```

```shell
# --cvrr_dataset_path is the path that points to the downloaded CVRR-ES folder
# --output_dir can be any path to save the predictions
python gpt4_evaluation.py \
--cvrr_dataset_path <path-to-CVRR_ES-folder> \
--output_dir <folder-path-to-save-predictions> \
--api_key <enter-openai-api-key>
```

The above command will generate predictions using the standard prompting method.
The generated predictions will be saved into the folder specified by --output_dir argument.

