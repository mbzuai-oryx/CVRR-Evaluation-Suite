## Generating Question-Answer Pairs using LLMs for CVRR-ES Benchmark

This document provides instructions for executing the sample code to generate open-ended question-answer pairs for the CVRR-ES benchmark.

**For users aiming to utilize the official CVRR-ES dataset:** Please note that the first version of the CVRR-ES dataset is finalized and available for use (refer to [README.md](../README.md) for more details). If you would like to generate the question-answer pairs from scratch, follow the instructions provided here. These instructions are intended for reproduction purposes only.

#### Generating QA Pairs using LLMs

1. Set up the CVRR-ES dataset folder (for videos and captions) by following steps provided in [README.md](../README.md). 
2. Install the openai library using pip.
```shell
# Install the openai package
pip install openai==0.28.0

cd QA-Generation-Scripts/

```
3. To generate question-answer pairs for the evaluation dimension of _Interpretation of social context_, run the below command.

```shell

python generate_qa_social_context.py --evaluation_dimension_path_name <folder-path-to-interpretation_of_social_context> --api_key <your-open-ai-key>

```

This will generate and save the question answer pairs in separate json files for each video in the provided folder directory.
The above steps can be repeated to generate QA pairs other evaluation dimension datasets by running their respective code file provided in [QA-Generation-Scripts](../QA-Generation-Scripts).

Feel free to open an issue if you encounter any problems.



