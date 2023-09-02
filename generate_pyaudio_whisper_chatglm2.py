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

import os
import torch
import time
import intel_extension_for_pytorch as ipex
import argparse
import numpy as np
import inquirer
import sounddevice
import wave

from bigdl.llm.transformers import AutoModelForCausalLM,AutoModel
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import LlamaTokenizer,AutoTokenizer
from transformers import WhisperProcessor
from transformers import TextStreamer
from colorama import Fore
import speech_recognition as sr
from datasets import load_dataset
import pyaudio

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--llama2-repo-id-or-model-path', type=str, default="./chatglm2-6b-int4/",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--whisper-repo-id-or-model-path', type=str, default="./whisper-medium-int4/",
                        help='The huggingface repo id for the Whisper (e.g. `openai/whisper-small` and `openai/whisper-medium`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')

    args = parser.parse_args()

    # Select device


    whisper_model_path = args.whisper_repo_id_or_model_path
    llama_model_path = args.llama2_repo_id_or_model_path


    print("Converting and loading models...")
    processor = WhisperProcessor.from_pretrained(whisper_model_path)

    # generate token ids
   # whisper =  AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, load_in_4bit=True, optimize_model=False)
    whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True, optimize_model=False)
    whisper.config.forced_decoder_ids = None
    whisper = whisper.to('xpu')
    
  #  llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, load_in_4bit=True, trust_remote_code=True, optimize_model=False)
    llama_model = AutoModel.load_low_bit(llama_model_path, trust_remote_code=True, optimize_model=False)
    llama_model = llama_model.to('xpu')
   # tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path, trust_remote_code=True)

    r = sr.Recognizer()

    with torch.inference_mode():
        torch.xpu.synchronize()

        while 1:
                  
            aa = str(input("是否开始录音？   （Y/N）"))
            if  aa == str("Y") or aa == str("y") :
                CHUNK = 1024
                FORMAT = pyaudio.paInt16
                CHANNELS = 1                # 声道数
                RATE = 16000               # 采样率
                RECORD_SECONDS = 10
                WAVE_OUTPUT_FILENAME = "/home/adc2/crystal/llm/pyaudio_out.wav"
                p = pyaudio.PyAudio()
                
                stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)

                print("*"*10, "开始录音：请输入语音")
                frames = []
                stat = True
                tempnum = 0
                data =0
                start_time = time.time()
            #    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                while stat:
                    data = stream.read(CHUNK)  ## <class 'bytes'> ,exception_on_overflow = False
                    frames.append(data)   ## <class 'list'>
            
                    if (time.time() - start_time) > RECORD_SECONDS:
                        stat = False
                print("*"*10, "录音结束\n")

                stream.stop_stream()
                stream.close()
                p.terminate()
                time_record = time.time()
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
            elif aa == str("N") or aa == str("n") :
                break
          
            r = sr.Recognizer()
            with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source1:
                audio = r.record(source1)  # read the entire audio file
            print("save time",time.time() - time_record)   
            frame_data = np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
           # print(frame_data)
            
           # print(np.max(np.frombuffer(data, np.int16)))
           # frame_data_ = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
           # print(frame_data_)
            print("Recognizing...")
            input_features = processor(frame_data, sampling_rate=RATE, return_tensors="pt").input_features
            input_features = input_features.contiguous().to('xpu')

            predicted_ids = whisper.generate(input_features)
            output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            output_str = output_str[0]
            print("\n" + Fore.GREEN + "Whisper : " + Fore.RESET + "\n" + output_str)
            print("\n" + Fore.BLUE + "BigDL-LLM: " + Fore.RESET)
            prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=output_str)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
            streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
            _ = llama_model.generate(input_ids, streamer=streamer, do_sample=False, max_new_tokens=args.n_predict)
