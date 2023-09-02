# microphone_chatglm2
microphone in (pyaudio) > ASR (whisper-medium) > chat (chatglm2-6b)

OS: Ubuntu22.04 kernel 5.19

Arc A770 with driver 23.17

oneAPI 2023.2 base toolkit

```
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install librosa soundfile datasets
pip install accelerate
pip install SpeechRecognition sentencepiece colorama
# If you failed to install PyAudio, try to run sudo apt install portaudio19-dev on ubuntu
pip install PyAudio inquirer sounddevice

source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
python generate_pyaudio_whisper_chatglm2.py
```

code refer to https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/gpu/voiceassistant
