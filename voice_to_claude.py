#!/usr/bin/env python3
"""
Voice-to-Claude: Push-to-talk voice input for Claude Code CLI.

Usage:
    Hold Ctrl+Alt+V (or configured hotkey) to record, release to transcribe and send.

Modes:
    --mode clipboard   : Copy transcription to clipboard (default)
    --mode type        : Type transcription into active window
    --mode claude      : Send directly to claude CLI in current terminal
    --mode stdout      : Print to stdout only
"""

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
MODEL_SIZE = "base.en"  # Options: tiny.en, base.en, small.en, medium.en, large-v3
HOTKEY = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode.from_char('v')}  # Ctrl+Alt+V

class VoiceRecorder:
    def __init__(self, model_size: str = MODEL_SIZE, device: int | None = None):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.audio_data = []
        self.current_keys = set()
        self.hotkey_pressed = False

    def load_model(self):
        """Load Whisper model (lazy loading for faster startup)."""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_size}'...", file=sys.stderr)
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )
            print("Model loaded.", file=sys.stderr)
        return self.model

    def audio_callback(self, indata, frames, time_info, status):
        """Called for each audio block during recording."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            return
        self.recording = True
        self.audio_data = []
        print("\r[Recording...] ", end="", file=sys.stderr, flush=True)

    def stop_recording(self) -> np.ndarray | None:
        """Stop recording and return audio data."""
        if not self.recording:
            return None
        self.recording = False

        # Collect all queued audio
        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())

        if not self.audio_data:
            print("\r[No audio recorded]", file=sys.stderr)
            return None

        audio = np.concatenate(self.audio_data, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE
        print(f"\r[Recorded {duration:.1f}s] ", end="", file=sys.stderr, flush=True)
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        model = self.load_model()
        print("[Transcribing...] ", end="", file=sys.stderr, flush=True)

        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text = " ".join(segment.text.strip() for segment in segments)
        print(f"[Done]", file=sys.stderr)
        return text.strip()

    def on_press(self, key):
        """Handle key press."""
        self.current_keys.add(key)
        if self.current_keys == HOTKEY and not self.hotkey_pressed:
            self.hotkey_pressed = True
            self.start_recording()

    def on_release(self, key):
        """Handle key release."""
        if key in HOTKEY and self.hotkey_pressed:
            self.hotkey_pressed = False
            self.current_keys.discard(key)
            return True  # Signal to process recording
        self.current_keys.discard(key)
        return False

def copy_to_clipboard(text: str):
    """Copy text to clipboard using xclip."""
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(),
            check=True
        )
        print(f"[Copied to clipboard]", file=sys.stderr)
    except FileNotFoundError:
        print("[Error: xclip not found. Install with: sudo apt install xclip]", file=sys.stderr)

def type_text(text: str):
    """Type text into active window using xdotool."""
    try:
        subprocess.run(["xdotool", "type", "--clearmodifiers", text], check=True)
    except FileNotFoundError:
        print("[Error: xdotool not found. Install with: sudo apt install xdotool]", file=sys.stderr)

def send_to_claude(text: str):
    """Send text to claude CLI."""
    # Type the text followed by enter
    type_text(text)
    time.sleep(0.1)
    subprocess.run(["xdotool", "key", "Return"], check=True)

def list_devices():
    """List available audio input devices."""
    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  {i}: {dev['name']}{default}")

def main():
    parser = argparse.ArgumentParser(description="Voice-to-Claude: Push-to-talk voice input")
    parser.add_argument("--mode", choices=["clipboard", "type", "claude", "stdout"],
                        default="clipboard", help="Output mode (default: clipboard)")
    parser.add_argument("--model", default=MODEL_SIZE,
                        help=f"Whisper model size (default: {MODEL_SIZE})")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--hotkey", default="ctrl+alt+v",
                        help="Hotkey combination (default: ctrl+alt+v)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Parse hotkey
    global HOTKEY
    hotkey_parts = args.hotkey.lower().split("+")
    hotkey_set = set()
    for part in hotkey_parts:
        part = part.strip()
        if part in ("super", "cmd", "win"):
            hotkey_set.add(keyboard.Key.cmd)
        elif part in ("ctrl", "control"):
            hotkey_set.add(keyboard.Key.ctrl)
        elif part in ("alt",):
            hotkey_set.add(keyboard.Key.alt)
        elif part in ("shift",):
            hotkey_set.add(keyboard.Key.shift)
        elif len(part) == 1:
            hotkey_set.add(keyboard.KeyCode.from_char(part))
        else:
            print(f"Unknown hotkey part: {part}", file=sys.stderr)
            return
    HOTKEY = hotkey_set

    recorder = VoiceRecorder(model_size=args.model, device=args.device)

    print(f"Voice-to-Claude ready!", file=sys.stderr)
    print(f"  Hotkey: {args.hotkey}", file=sys.stderr)
    print(f"  Mode: {args.mode}", file=sys.stderr)
    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"Press Ctrl+C to exit.", file=sys.stderr)
    print(file=sys.stderr)

    # Pre-load model
    recorder.load_model()

    process_event = threading.Event()

    def on_release_wrapper(key):
        if recorder.on_release(key):
            process_event.set()

    # Start audio stream
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=args.device,
        callback=recorder.audio_callback
    ):
        # Start keyboard listener
        listener = keyboard.Listener(
            on_press=recorder.on_press,
            on_release=on_release_wrapper
        )
        listener.start()

        try:
            while True:
                # Wait for hotkey release
                if process_event.wait(timeout=0.1):
                    process_event.clear()

                    audio = recorder.stop_recording()
                    if audio is not None and len(audio) > SAMPLE_RATE * 0.3:  # Min 0.3s
                        text = recorder.transcribe(audio)
                        if text:
                            print(f"\n>>> {text}\n", file=sys.stderr)

                            if args.mode == "clipboard":
                                copy_to_clipboard(text)
                            elif args.mode == "type":
                                type_text(text)
                            elif args.mode == "claude":
                                send_to_claude(text)
                            elif args.mode == "stdout":
                                print(text)
                                sys.stdout.flush()
                    else:
                        print("\r[Recording too short, ignored]", file=sys.stderr)

        except KeyboardInterrupt:
            print("\nExiting...", file=sys.stderr)
        finally:
            listener.stop()

if __name__ == "__main__":
    main()
