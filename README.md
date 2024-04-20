# FinetuneSpeechT5-Spanish
[NYU Steinhardt, Deep Learning for Media Final Project]


Miya Ding, Heqi Qiao, Trevor Freed


This repository hosts the code and resources for fine-tuning a SpeechT5 model for text-to-speech (TTS) tasks using the VoxPopuli dataset. The project leverages the Hugging Face Transformers and Datasets libraries to prepare, process, and train a model capable of generating human-like speech.

## Installation

To run the scripts, you need to install several dependencies. Execute the following commands in your terminal to set up the environment:

```bash
pip3 install transformers datasets soundfile speechbrain==0.5.16 accelerate librosa
pip3 install git+https://github.com/huggingface/transformers.git
```


Or just run the environment file:

```bash
pip install -r envs.txt
```
## Dataset

The project uses the [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) dataset, specifically the Spanish portion. The dataset is preprocessed to normalize audio and text features suitable for training.

## Model

The base model is [microsoft/speecht5_tts](https://huggingface.co/docs/transformers/en/model_doc/speecht5), fine-tuned on the processed dataset. A custom data collator handles the batching and padding of training data.

## Training

Training configurations are set up using the Seq2SeqTrainingArguments class from the Transformers library. The model is trained with a focus on using a GPU for acceleration, and the best model is saved based on validation loss.

## Inference

Post-training, the model can generate spectrograms from text, which are then converted into audible speech using the HiFi-GAN vocoder model. Example scripts demonstrate how to perform this conversion and how to save the output as a WAV file.

## Usage

Example usage of the model to generate speech:  


```

# Load pre-trained model and processor
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
processor = SpeechT5Processor.from_pretrained('path_to_processor')
model = SpeechT5ForTextToSpeech.from_pretrained('path_to_model')

# Prepare input text and generate spectrogram
input_text = "Hello, world!"
inputs = processor(text=input_text, return_tensors="pt")
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

# Convert spectrogram to audio and play it
from IPython.display import Audio
Audio(speech.numpy(), rate=16000)


```
- Remember to use `git lfs pull` to pull the safetensors file for loading the model
- Run the ***Play_and_Evaluate.ipynb*** file to try the model and see its performance!
- There are 3 man speakers and 3 woman speakers for you to choose.


## Reference

[Text to Speech](https://huggingface.co/docs/transformers/en/tasks/text-to-speech)
