#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pip install bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
# pip install bigdl-core-xe
# pip install librosa soundfile datasets
# pip install accelerate
# pip install SpeechRecognition sentencepiece colorama



# export USE_XETLA=OFF
# export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 ENABLE_SDP_FUSION=1

# python generate.py --llama2-repo-id-or-model-path /home/wid/.ipc/changmin/Llama-2-7b-chat-hf --whisper-repo-id-or-model-path /home/wid/.ipc/yina/models/whisper-small --audio-dir /home/wid/.ipc/yina/IPEX_DPCPP --n-predict 128 

# 进terminal后需要source /opt/intel/oneapi/setvars.sh  #注意： 需要oneAPI 2023.2，要不然会出错

 
# audio-dir 是放音频文件的dir
# 音频文件需要时16k采样，转换为16k方法： ffmpeg -i input.wav -ar 16000 output.wav



import os
import torch
import time
import argparse
import numpy as np

from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import LlamaTokenizer
import intel_extension_for_pytorch as ipex
from transformers import WhisperProcessor
from transformers import TextStreamer
from colorama import Fore
import speech_recognition as sr
from datasets import load_dataset
from bigdl.llm.ggml.model.chatglm.chatglm import ChatGLM
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer


# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
DEFAULT_SYSTEM_PROMPT = """\
"""
CHATGLM_V2_PROMPT_FORMAT = "问：{prompt}\n\n答："

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def get_input_features(r, audio_file):
    print("audio_file: ", audio_file)
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # read the entire audio file
    frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
    print("audio: ", audio, "frame_data: ", frame_data)
    input_features = processor(frame_data, sampling_rate=audio.sample_rate, return_tensors="pt").input_features
    input_features = input_features.half().contiguous().to('xpu')
    return input_features

def get_input_features_micro(r):
    with sr.Microphone(device_index=1, sample_rate=16000) as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)

        print(Fore.YELLOW + "Listening now..." + Fore.RESET)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
            # refer to https://github.com/openai/whisper/blob/main/whisper/audio.py#L63
            frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
            input_features = processor(frame_data, sampling_rate=audio.sample_rate, return_tensors="pt").input_features
            input_features = input_features.half().contiguous().to('xpu')
            print("Recognizing...")
        except Exception as e:
            unrecognized_speech_text = (
                f"Sorry, I didn't catch that. Exception was: \n {e}"
            )
            print(unrecognized_speech_text)
    
    return input_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')

    parser.add_argument('--chatglm2-repo-id-or-model-path', type=str, default="/home/adc2/crystal/llm/chatglm2-6b-int4",
                        help=' to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--whisper-repo-id-or-model-path', type=str, default="./whisper-medium-int4",                    
                        help='The huggingface repo id for the Whisper (e.g. `openai/whisper-small` and `openai/whisper-medium`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--audio-dir', type=str, default="~/whisper/", help="The path to the audio directory.")

    parser.add_argument('--audio', type=str, default="./input.wav", help="The absolute path to the audio directory.")#绝对路径
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')

    args = parser.parse_args()
    whisper_model_path = args.whisper_repo_id_or_model_path
    #llama_model_path = args.llama2_repo_id_or_model_path
    chatglm2_model_path = args.chatglm2_repo_id_or_model_path

    print("Converting and loading models...")
    print("loading whisper----1")
    processor = WhisperProcessor.from_pretrained(whisper_model_path)

    # generate token ids
    print("loading whisper----2")
    #whisper =  AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, load_in_4bit=True, optimize_model=False)
    whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True, optimize_model=False)
    whisper.config.forced_decoder_ids = None
    whisper = whisper.half().to('xpu')
    #whisper.save_low_bit("./whisper-medium-int4/")
    print("loading whisper----Done")
    
    #llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, trust_remote_code=True, optimize_model=False)
    #llama_model = llama_model.half().to('xpu')
    #tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)

    print("loading chatglm2---------")
    #chatglm2_model = AutoModel.from_pretrained(chatglm2_model_path, load_in_4bit=True, trust_remote_code=True, optimize_model=False)
    chatglm2_model =  AutoModel.load_low_bit(chatglm2_model_path, trust_remote_code=True, optimize_model=False)
    #chatglm2_model = ChatGLM(chatglm2_model_path + "ggml-chatglm2-6b-q4_0.bin", n_threads=20,n_ctx=4096)
    chatglm2_model = chatglm2_model.half().to('xpu')
    tokenizer = AutoTokenizer.from_pretrained(chatglm2_model_path, trust_remote_code=True)
    print("loading chatglm2---------Done")

    audio_dir = args.audio_dir
    audio = args.audio
    r = sr.Recognizer()

    with torch.inference_mode():
        # warm up
        #input_features = get_input_features(r, os.path.join(audio_dir, f"audio_0.flac"))
        input_features = get_input_features(r, audio)
        torch.xpu.synchronize()
        predicted_ids = whisper.generate(input_features)
        torch.xpu.synchronize()
        output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        output_str = output_str[0]
        input_ids = tokenizer.encode(output_str, return_tensors="pt").to('xpu')
        #output = llama_model.generate(input_ids, do_sample=False, max_new_tokens=32)
        print(chatglm2_model.device, input_ids.device)
        output = chatglm2_model.generate(input_ids, do_sample=False, max_new_tokens=32)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        torch.xpu.synchronize()
        
        for i in range(3):
            #input_features = get_input_features(r, os.path.join(audio_dir, f"audio_{i}.flac"))
            input_features = get_input_features(r, audio)
            predicted_ids = whisper.generate(input_features)
            output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            output_str = output_str[0]
            print("\n" + Fore.GREEN + "Whisper : " + Fore.RESET + "\n" + output_str)
            print("\n" + Fore.BLUE + "BigDL-LLM: " + Fore.RESET)
            prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=output_str)
            #prompt = get_prompt(output_str, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
            #prompt = output_str

            if i==0:
                prompt= CHATGLM_V2_PROMPT_FORMAT.format(prompt="你好")
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
            streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
            #_ = llama_model.generate(input_ids, streamer=streamer, do_sample=False, max_new_tokens=args.n_predict)
            _ = chatglm2_model.generate(input_ids, streamer=streamer, do_sample=False, max_new_tokens=args.n_predict)
            ##out = chatglm2_model.generate(input_ids, do_sample=False, max_new_tokens=args.n_predict)
            #out_str = tokenizer.decode(out[0], skip_special_tokens=True)
            #print("out_str: ", out_str)
