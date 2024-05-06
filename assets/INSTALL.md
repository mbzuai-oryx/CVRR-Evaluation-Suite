## Installation 
We provide installation instructions for:
- Setting up environments for inference with Video-LMMs
- Downloading and setting-up model weights (if required) for Video-LMMs

## Setting environment and weights for TimeChat 
Note: instructions are borrowed from the [TimeChat Github repository](https://github.com/RenShuhuai-Andy/TimeChat)
1) Run the following commands to install environment for TimeChat 
```shell
cd Video-LMMs-Inference/TimeChat
# First, install ffmpeg.
apt update
apt install ffmpeg
# Then, create a conda environment:
conda env create -f environment.yml
conda activate timechat
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

2) Follow the below instructions to set-up the model weights for TimeChat

#### Pre-trained Image Encoder (EVA ViT-g)
```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```

#### Pre-trained Image Q-Former (InstructBLIP Q-Former)
```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
```

#### Pre-trained Language Decoder (LLaMA-2-7B) and Video Encoder (Video Q-Former of Video-LLaMA)

Use `git-lfs` to download weights of [Video-LLaMA (7B)](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main):
```bash
git lfs install
git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned
```

#### Instruct-tuned [TimeChat-7B](https://huggingface.co/ShuhuaiRen/TimeChat-7b)
```bash
git lfs install
git clone https://huggingface.co/ShuhuaiRen/TimeChat-7b
```

The file structure looks like:
```
TimeChat/ckpt/
        |–– Video-LLaMA-2-7B-Finetuned/
            |-- llama-2-7b-chat-hf/
            |-- VL_LLaMA_2_7B_Finetuned.pth
        |–– instruct-blip/
            |-- instruct_blip_vicuna7b_trimmed.pth
        |–– eva-vit-g/
            |-- eva_vit_g.pth
        |-- timechat/
            |-- timechat_7b.pth
```


## Setting environment for Video-LLaVA
Note: instructions are borrowed from the [Video-LLaVA Github repository](https://github.com/PKU-YuanGroup/Video-LLaVA)
1) Run the following commands to install environment for Video-LLaVA 
```shell
## Following requirements must be met for successful installation
# Python >= 3.10
# Pytorch == 2.0.1
# CUDA Version >= 11.7
# Install required packages:

cd Video-LMMs-Inference/Video-LLaVA
# install anaconda environment and packages
conda create -n videollava python=3.10 -y
conda activate videollava

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

**Model Weights:** Note that Video-LLaVA will automatically download the weights after running for first time. No need to manually download the model weights.

## Setting environment for Gemini-Pro-Vision
**Note:** We use google-cloud platform for performing inference using Gemini model. Specifically, you would need to set-up the following:
1) Configure a project (or use an existing one, if any) on google cloud [more info here](https://developers.google.com/workspace/guides/create-project)
2) Create a google-cloud bucket, and upload the [CVRR-ES dataset](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/uzair_khattak_mbzuai_ac_ae/EktDA83_8UxJrc23DQfrfv8Bvw41YxWVBgD3Fapxs69rRg?e=Xxanhp) in that bucket. 
3) Run the following commands to install the packages
```shell
conda create -n gemini python=3.10 -y
pip install --upgrade google-cloud-aiplatform
gcloud auth application-default login
```

## Setting environment for GPT4-(V)ision
3) Run the following commands to install the packages
```shell
conda create -n gpt4v python=3.10 -y
# install open-ai
pip install openai==1.13.3
```
