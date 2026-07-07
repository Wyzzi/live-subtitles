"""
Live local subtitles for videos / Discord calls.

Captures audio either from a microphone, or (with --loopback) directly from
whatever is playing out of a speaker/headphone device via WASAPI loopback --
so audio keeps playing normally for the listener and nothing needs to be
rerouted through a virtual cable. Detects speech segments with simple
energy-based VAD, transcribes them locally with faster-whisper on the GPU,
and shows the result in an always-on-top overlay window. The subtitle
language matches whatever was spoken (Russian speech -> Russian text,
English speech -> English text) via Whisper's language auto-detection --
nothing is translated.

Everything runs locally; no audio or text leaves the machine.
"""
import argparse
import glob
import os
import queue
import sys
import threading
import tkinter as tk
import time


def _register_nvidia_dll_dirs():
    """Make the CUDA/cuDNN DLLs from the nvidia-* pip packages loadable.

    pip installs them under site-packages/nvidia/*/bin, but that is not on
    the default DLL search path, so ctranslate2 fails with e.g.
    "Library cublas64_12.dll is not found" even though it's installed.
    """
    if sys.platform != "win32":
        return
    bin_dirs = glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin"))
    print(f"[cuda-dll-setup] sys.prefix={sys.prefix}")
    print(f"[cuda-dll-setup] found {len(bin_dirs)} nvidia bin dir(s): {bin_dirs}")
    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(bin_dir)
            print(f"[cuda-dll-setup] registered: {bin_dir}")
        except (OSError, AttributeError) as e:
            print(f"[cuda-dll-setup] FAILED to register {bin_dir}: {e}")

    import ctypes
    for dll_name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(dll_name)
            print(f"[cuda-dll-setup] loaded OK: {dll_name}")
        except OSError as e:
            print(f"[cuda-dll-setup] FAILED to load {dll_name}: {e}")


_register_nvidia_dll_dirs()

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100 ms blocks at 16kHz

MODEL_CHOICES = [
    ("Fast", "small", "Quicker captions, slightly less accurate. Good for live conversation."),
    ("Accurate", "large-v3", "Better quality, more delay per caption. Good for movies/lectures."),
]


def choose_model_interactively(default_model: str) -> str:
    """Show a small picker window so she can pick the model before it loads."""
    picked = {"model": default_model}

    root = tk.Tk()
    root.title("Live Subtitles")
    root.attributes("-topmost", True)
    root.configure(bg="#1e1e1e")

    tk.Label(
        root, text="Choose a speech recognition model", font=("Segoe UI", 14, "bold"),
        fg="white", bg="#1e1e1e",
    ).pack(padx=24, pady=(20, 12))

    def pick(model_name):
        picked["model"] = model_name
        root.destroy()

    for label, model_name, description in MODEL_CHOICES:
        frame = tk.Frame(root, bg="#1e1e1e")
        frame.pack(fill="x", padx=24, pady=6)
        tk.Button(
            frame, text=f"{label}  ({model_name})", font=("Segoe UI", 12), width=26,
            command=lambda m=model_name: pick(m),
        ).pack()
        tk.Label(
            frame, text=description, font=("Segoe UI", 9), fg="#aaaaaa", bg="#1e1e1e",
            wraplength=320, justify="center",
        ).pack(pady=(4, 0))

    tk.Label(
        root, text="(Closing this window without choosing keeps the default.)",
        font=("Segoe UI", 8), fg="#777777", bg="#1e1e1e",
    ).pack(pady=(4, 16))

    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()
    return picked["model"]


class Overlay:
    """Borderless, always-on-top subtitle bar."""

    def __init__(self, font_size: int, opacity: float, min_hold_s: float = 1.2,
                 chars_per_sec: float = 18.0):
        self.min_hold_s = min_hold_s
        self.chars_per_sec = chars_per_sec
        self._next_allowed_change = 0.0
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", opacity)
        self.root.configure(bg="black")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = int(screen_w * 0.8)
        win_h = int(screen_h * 0.15)
        x = (screen_w - win_w) // 2
        y = int(screen_h * 0.80)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        self.label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", font_size, "bold"),
            fg="white",
            bg="black",
            wraplength=win_w - 40,
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=10, pady=10)

        # Drag to reposition.
        self.label.bind("<Button-1>", self._start_move)
        self.label.bind("<B1-Motion>", self._do_move)
        self._drag_offset = (0, 0)

        # Esc closes the overlay.
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _start_move(self, event):
        self._drag_offset = (event.x, event.y)

    def _do_move(self, event):
        x = self.root.winfo_x() + event.x - self._drag_offset[0]
        y = self.root.winfo_y() + event.y - self._drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def set_text(self, text: str):
        self.label.config(text=text)

    def poll_queue(self, q: "queue.Queue[str]"):
        now = time.monotonic()
        if now >= self._next_allowed_change:
            try:
                text = q.get_nowait()
                self.set_text(text)
                # Keep each caption on screen long enough to read, so a
                # quick follow-up line doesn't instantly wipe it out.
                hold = max(self.min_hold_s, len(text) / self.chars_per_sec)
                self._next_allowed_change = now + hold
            except queue.Empty:
                pass
        self.root.after(50, self.poll_queue, q)

    def run(self, q: "queue.Queue[str]"):
        self.poll_queue(q)
        self.root.mainloop()


class SpeechSegmenter:
    """Simple energy-based voice activity detector that yields speech segments."""

    def __init__(self, silence_threshold: float, min_silence_ms: int, max_segment_s: float):
        # silence_threshold is expressed on a 0-1 float scale (as a fraction
        # of full scale); push() receives int16 PCM, so convert here.
        self.silence_threshold = silence_threshold * 32768.0
        self.min_silence_blocks = max(1, int(min_silence_ms / (BLOCK_SIZE / SAMPLE_RATE * 1000)))
        self.max_segment_blocks = int(max_segment_s / (BLOCK_SIZE / SAMPLE_RATE))
        self.buffer = []
        self.silence_run = 0
        self.speaking = False

    def push(self, block: np.ndarray):
        """Feed one audio block. Returns a finished segment (np.ndarray) or None."""
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        is_speech = rms > self.silence_threshold

        if is_speech:
            self.buffer.append(block)
            self.speaking = True
            self.silence_run = 0
            if len(self.buffer) >= self.max_segment_blocks:
                return self._flush()
            return None

        if self.speaking:
            self.silence_run += 1
            self.buffer.append(block)
            if self.silence_run >= self.min_silence_blocks:
                return self._flush()
        return None

    def _flush(self):
        segment = np.concatenate(self.buffer) if self.buffer else None
        self.buffer = []
        self.speaking = False
        self.silence_run = 0
        return segment


def transcriber_worker(model: WhisperModel, audio_q: "queue.Queue[np.ndarray]",
                        text_q: "queue.Queue[str]", beam_size: int):
    while True:
        segment = audio_q.get()
        if segment is None:
            break
        try:
            t0 = time.monotonic()
            audio_float = segment.astype(np.float32) / 32768.0
            segments, info = model.transcribe(
                audio_float,
                beam_size=beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            elapsed = time.monotonic() - t0
            if text:
                print(f"[{info.language}] ({elapsed:.1f}s) {text}")
                text_q.put(text)
            else:
                print("(segment had no recognizable speech)")
        except Exception as e:
            print(f"Transcription error: {e}")


def _resample_to_16k(mono: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE:
        return mono
    duration = len(mono) / src_rate
    n_out = int(round(duration * SAMPLE_RATE))
    if n_out <= 0:
        return np.empty(0, dtype=np.float32)
    src_idx = np.linspace(0, len(mono) - 1, num=len(mono))
    dst_idx = np.linspace(0, len(mono) - 1, num=n_out)
    return np.interp(dst_idx, src_idx, mono).astype(np.float32)


def _resolve_recorder(device_name: str, loopback: bool):
    if loopback:
        speaker = sc.get_speaker(device_name)
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        native_rate = 48000
    else:
        mic = sc.get_microphone(device_name)
        native_rate = 48000
    return mic, native_rate


def audio_capture_loop(device_name: str, segmenter: SpeechSegmenter,
                        audio_q: "queue.Queue[np.ndarray]", stop_event: threading.Event,
                        loopback: bool):
    mic, native_rate = _resolve_recorder(device_name, loopback)
    native_block = int(BLOCK_SIZE * native_rate / SAMPLE_RATE)
    threshold16 = segmenter.silence_threshold

    print(f"Capturing from '{device_name}' (loopback={loopback}) at {native_rate} Hz. "
          f"Silence threshold: {threshold16:.0f}/32768.")

    last_meter = 0.0
    try:
        with mic.recorder(samplerate=native_rate) as rec:
            while not stop_event.is_set():
                frames = rec.record(numframes=native_block)
                mono = frames.mean(axis=1) if frames.ndim > 1 and frames.shape[1] > 1 else frames[:, 0]
                resampled = _resample_to_16k(mono, native_rate)
                pcm16 = (resampled * 32768.0).astype(np.int16)

                now = time.monotonic()
                if now - last_meter > 1.0:
                    rms = float(np.sqrt(np.mean(pcm16.astype(np.float64) ** 2)))
                    bar_len = min(40, int(rms / 32768.0 * 400))
                    marker = "SPEECH" if rms > threshold16 else "silence"
                    print(f"level: {'#' * bar_len:<40} {rms:7.0f}/32768  [{marker}]")
                    last_meter = now

                segment = segmenter.push(pcm16)
                if segment is not None and len(segment) > SAMPLE_RATE * 0.3:
                    print(f"-> speech segment captured ({len(segment) / SAMPLE_RATE:.1f}s), transcribing...")
                    audio_q.put(segment)
    except Exception as e:
        print(f"Audio capture error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Live local subtitles via microphone or speaker loopback.")
    parser.add_argument("--device", type=str, required=True,
                         help="Exact device name (run list_devices.py to list them).")
    parser.add_argument("--loopback", action="store_true",
                         help="Capture what plays out of an OUTPUT device (e.g. headphones/"
                              "speakers) via WASAPI loopback, instead of recording an input "
                              "device. Use this to caption audio she's hearing without any "
                              "virtual cable, so her own playback isn't rerouted or muted.")
    parser.add_argument("--model", default="small",
                         help="faster-whisper model size: tiny/base/small/medium/large-v3 "
                              "(default: small -- fast and accurate enough for live captions on "
                              "an RTX 4070; use medium/large-v3 if you want more accuracy and can "
                              "tolerate more lag). Ignored if --no-model-picker is not set, since "
                              "the picker window lets you choose this interactively instead.")
    parser.add_argument("--no-model-picker", action="store_true",
                         help="Skip the startup model-choice window and use --model directly.")
    parser.add_argument("--compute-type", default="float16",
                         help="float16 (recommended on RTX 4070), int8_float16, or int8.")
    parser.add_argument("--beam-size", type=int, default=1,
                         help="Whisper beam search width. 1 (default) = greedy decoding, "
                              "fastest. Higher (e.g. 5) is a bit more accurate but slower.")
    parser.add_argument("--silence-threshold", type=float, default=0.01,
                         help="RMS amplitude below which audio is considered silence (0-1 float scale).")
    parser.add_argument("--min-silence-ms", type=int, default=350,
                         help="Silence duration (ms) that ends a speech segment. Lower = captions "
                              "appear sooner after each pause, at some risk of cutting words short.")
    parser.add_argument("--max-segment-s", type=float, default=6.0,
                         help="Hard cap on segment length in seconds when speech doesn't pause. "
                              "Lower = more frequent, shorter updates instead of one long delayed "
                              "chunk of text.")
    parser.add_argument("--min-hold-s", type=float, default=1.2,
                         help="Minimum time (seconds) each caption stays on screen before the "
                              "next one can replace it, so quick lines don't flash by unread.")
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--opacity", type=float, default=0.85)
    args = parser.parse_args()

    chosen_model = args.model
    if not args.no_model_picker:
        chosen_model = choose_model_interactively(args.model)
        print(f"Model chosen: {chosen_model}")

    print(f"Loading faster-whisper model '{chosen_model}' on GPU ...")
    model = WhisperModel(chosen_model, device="cuda", compute_type=args.compute_type)
    print("Model loaded. Listening ...")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    text_q: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    segmenter = SpeechSegmenter(args.silence_threshold, args.min_silence_ms, args.max_segment_s)

    t_worker = threading.Thread(
        target=transcriber_worker, args=(model, audio_q, text_q, args.beam_size), daemon=True
    )
    t_worker.start()

    t_capture = threading.Thread(
        target=audio_capture_loop,
        args=(args.device, segmenter, audio_q, stop_event, args.loopback),
        daemon=True,
    )
    t_capture.start()

    overlay = Overlay(font_size=args.font_size, opacity=args.opacity, min_hold_s=args.min_hold_s)
    try:
        overlay.run(text_q)
    finally:
        stop_event.set()
        audio_q.put(None)


if __name__ == "__main__":
    main()
