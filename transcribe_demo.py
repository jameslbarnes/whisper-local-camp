#! python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from flask import Flask, request, jsonify
import logging
import threading

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Global variables
phrase_time = None
data_queue = Queue()
transcription_words = []
transcription_lock = threading.Lock()
MAX_WORD_COUNT = 200  # Define the maximum number of words in the transcription

# Initialize recognizer and model
recorder = sr.Recognizer()
audio_model = None

def init_model(args):
    global audio_model
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            logging.info("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                logging.info(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    with source:
        recorder.adjust_for_ambient_noise(source)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)
    logging.info("Started listening in background")

    # Start the transcription thread
    threading.Thread(target=transcription_thread, daemon=True).start()

def record_callback(_, audio: sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)
    logging.debug("Audio data added to queue")

def transcription_thread():
    global phrase_time, transcription_words
    while True:
        if not data_queue.empty():
            now = datetime.utcnow()
            phrase_complete = False
            if phrase_time and now - phrase_time > timedelta(seconds=3):
                phrase_complete = True
            phrase_time = now

            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()

            with transcription_lock:
                new_words = text.split()
                if phrase_complete:
                    transcription_words.extend(new_words)
                else:
                    transcription_words[-len(new_words):] = new_words

                # Trim the transcription words list if it exceeds the maximum word count
                if len(transcription_words) > MAX_WORD_COUNT:
                    transcription_words = transcription_words[-MAX_WORD_COUNT:]

            logging.debug(f"Transcription updated: {' '.join(transcription_words)}")
        sleep(0.1)

@app.route('/transcription', methods=['GET'])
def get_transcription():
    with transcription_lock:
        concatenated_transcription = ' '.join(transcription_words)
    return jsonify({'transcription': concatenated_transcription})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    init_model(args)
    app.run(host='0.0.0.0', port=5001)