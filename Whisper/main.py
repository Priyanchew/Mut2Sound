import pyaudio
import wave
import requests
import playsound
import sys
import keyboard
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

# Configuration for the audio stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-base.en"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

# Chat history
chat_history = []


def setup_readline_interface():
    print("Press Enter when you're ready to start speaking. Press Ctrl+C to exit.")


def start_recording():
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording... Press Enter to stop")

    frames = []

    try:
        while True:
            if keyboard.is_pressed('enter'):
                break
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("Recording stopped")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    transcribe_and_chat()


def transcribe_and_chat():
    result = pipe(WAVE_OUTPUT_FILENAME)

    print(f">> You said: {result['text']}")
    #
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant providing concise responses in at most two sentences."},
    #     *chat_history,
    #     {"role": "user", "content": transcribed_text}
    # ]
    #
    # chat_response = requests.post(
    #     "https://api.openai.com/v1/chat/completions",
    #     headers=headers,
    #     json={
    #         "model": "gpt-3.5-turbo",
    #         "messages": messages
    #     }
    # )
    # chat_response_text = chat_response.json()['choices'][0]['message']['content']
    #
    # chat_history.append({"role": "user", "content": transcribed_text})
    # chat_history.append({"role": "assistant", "content": chat_response_text})
    #
    # print(f">> Assistant said: {chat_response_text}")

    print("Press Enter to speak again, or Ctrl+C to quit.")


def streamed_audio(input_text):
    url = "https://api.openai.com/v1/audio/speech"


if __name__ == "__main__":
    setup_readline_interface()

    while True:
        if keyboard.is_pressed('enter'):
            start_recording()
        elif keyboard.is_pressed('ctrl+c'):
            print("Exiting application...")
            sys.exit(0)
