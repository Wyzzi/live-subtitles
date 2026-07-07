"""
Live local subtitles for videos / Discord calls.

Captures audio from one of three sources:
  - a speaker/headphone device via WASAPI loopback (--loopback): everything
    the listener hears, with audio still playing normally;
  - a single application, e.g. Discord only (--app, or picked in the startup
    window): uses WASAPI process loopback, so music/game sounds from other
    apps don't pollute the captions;
  - a microphone (no flags).

Detects speech segments with simple energy-based VAD, transcribes them
locally with faster-whisper on the GPU, and shows the result in an
always-on-top overlay window. The subtitle language matches whatever was
spoken (Russian speech -> Russian text, English speech -> English text) via
Whisper's language auto-detection -- nothing is translated.

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

# soundcard initializes COM in multithreaded (MTA) mode; comtypes (used by
# pycaw for listing apps with audio sessions) defaults to single-threaded
# (STA) and would fail with RPC_E_CHANGED_MODE on the same thread. Telling
# comtypes to use MTA as well keeps both libraries happy. Must be set before
# comtypes is first imported.
sys.coinit_flags = 0  # COINIT_MULTITHREADED


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

# Per-application capture (WASAPI process loopback) is optional: the native
# extension ships as prebuilt wheels for Python 3.10-3.13 only.
try:
    from proctap import ProcessAudioCapture
    from proctap._native import ProcessLoopback as _proctap_native_check  # noqa: F401
    HAS_APP_CAPTURE = True
except Exception:
    ProcessAudioCapture = None
    HAS_APP_CAPTURE = False

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100 ms blocks at 16kHz

MODEL_CHOICES = [
    ("Быстрая / Fast", "small", "Субтитры появляются быстрее, точность чуть ниже. Для живого общения (Discord)."),
    ("Точная / Accurate", "large-v3", "Лучшее качество, но больше задержка. Для фильмов и лекций."),
]


def list_audio_apps():
    """Return [(pid, name, is_active)] for processes that have an audio session.

    Uses the Windows Audio Session API via pycaw, so it finds exactly the
    apps that are (or recently were) producing sound -- Discord, browsers,
    players -- rather than guessing by process name.
    """
    try:
        from pycaw.pycaw import AudioUtilities
    except Exception:
        return []
    apps = []
    seen = set()
    try:
        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if proc is None:
                continue
            try:
                key = (proc.pid, proc.name())
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            apps.append((proc.pid, key[1], int(session.State) == 1))
    except Exception as e:
        print(f"Could not enumerate audio sessions: {e}")
    # Active (currently playing) sessions first.
    apps.sort(key=lambda a: (not a[2], a[1].lower()))
    return apps


def find_app_by_name(name: str):
    """Resolve an --app name (case-insensitive substring) to (pid, full name)."""
    matches = [a for a in list_audio_apps() if name.lower() in a[1].lower()]
    if not matches:
        raise SystemExit(
            f"No app with an audio session matches '{name}'. Make sure the app is "
            f"running and has played some sound, then try again "
            f"(run with no --app to see the list in the picker window)."
        )
    return matches[0][0], matches[0][1]


def choose_options_interactively(default_model: str):
    """Startup window: pick a model and an audio source (whole system / one app).

    Returns (model_name, source) where source is None for "whole system"
    (device loopback) or (pid, app_name) for single-app capture.
    """
    picked = {"model": default_model, "source": None}

    root = tk.Tk()
    root.title("Live Subtitles")
    root.attributes("-topmost", True)
    root.configure(bg="#1e1e1e")

    # --- Audio source section ---
    tk.Label(
        root, text="Откуда захватывать звук?", font=("Segoe UI", 13, "bold"),
        fg="white", bg="#1e1e1e",
    ).pack(padx=24, pady=(18, 6))

    listbox = tk.Listbox(
        root, font=("Segoe UI", 11), width=44, height=6,
        bg="#2b2b2b", fg="white", selectbackground="#4a6da7",
        highlightthickness=0, activestyle="none", exportselection=False,
    )
    listbox.pack(padx=24, pady=(0, 4))
    app_entries = []  # parallel list: None for "whole system", (pid, name) for apps

    def refresh_apps():
        listbox.delete(0, tk.END)
        app_entries.clear()
        listbox.insert(tk.END, "🔊  Весь звук компьютера")
        app_entries.append(None)
        if HAS_APP_CAPTURE:
            for pid, name, active in list_audio_apps():
                mark = "▶" if active else "·"
                listbox.insert(tk.END, f"{mark}  Только {name}  (PID {pid})")
                app_entries.append((pid, name))
        listbox.selection_set(0)

    refresh_apps()

    if HAS_APP_CAPTURE:
        tk.Button(
            root, text="Обновить список приложений", font=("Segoe UI", 9),
            command=refresh_apps,
        ).pack(pady=(0, 6))
    else:
        tk.Label(
            root, text="(Захват отдельного приложения требует Python 3.10–3.13 —\n"
                       "сейчас доступен только захват всего звука.)",
            font=("Segoe UI", 8), fg="#aa7777", bg="#1e1e1e", justify="center",
        ).pack(pady=(0, 6))

    # --- Model section ---
    tk.Label(
        root, text="Какую модель распознавания использовать?", font=("Segoe UI", 13, "bold"),
        fg="white", bg="#1e1e1e",
    ).pack(padx=24, pady=(10, 6))

    def pick(model_name):
        picked["model"] = model_name
        sel = listbox.curselection()
        if sel:
            picked["source"] = app_entries[sel[0]]
        root.destroy()

    for label, model_name, description in MODEL_CHOICES:
        frame = tk.Frame(root, bg="#1e1e1e")
        frame.pack(fill="x", padx=24, pady=4)
        tk.Button(
            frame, text=f"{label}  ({model_name})", font=("Segoe UI", 12), width=28,
            command=lambda m=model_name: pick(m),
        ).pack()
        tk.Label(
            frame, text=description, font=("Segoe UI", 9), fg="#aaaaaa", bg="#1e1e1e",
            wraplength=360, justify="center",
        ).pack(pady=(2, 0))

    tk.Label(
        root, text="(Если закрыть окно без выбора — весь звук и модель по умолчанию.)",
        font=("Segoe UI", 8), fg="#777777", bg="#1e1e1e",
    ).pack(pady=(6, 14))

    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()
    return picked["model"], picked["source"]


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
                        text_q: "queue.Queue[str]", beam_size: int, language: str):
    lang = None if language == "auto" else language
    while True:
        segment = audio_q.get()
        if segment is None:
            break
        # If transcription falls behind live audio, drop the oldest segments
        # instead of letting captions lag further and further behind.
        dropped = 0
        while audio_q.qsize() > 2:
            newer = audio_q.get()
            if newer is None:
                return
            segment = newer
            dropped += 1
        if dropped:
            print(f"(transcription running behind -- skipped {dropped} older segment(s))")
        try:
            t0 = time.monotonic()
            audio_float = segment.astype(np.float32) / 32768.0
            segments, info = model.transcribe(
                audio_float,
                language=lang,
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


class BlockFeeder:
    """Takes mono float32 audio at a native rate in arbitrary-sized chunks,
    resamples to 16 kHz, slices into fixed BLOCK_SIZE blocks, feeds the
    segmenter, prints a once-a-second level meter, and enqueues finished
    speech segments."""

    def __init__(self, segmenter: SpeechSegmenter, audio_q: "queue.Queue[np.ndarray]",
                 native_rate: int):
        self.segmenter = segmenter
        self.audio_q = audio_q
        self.native_rate = native_rate
        self.carry = np.empty(0, dtype=np.float32)
        self.last_meter = 0.0

    def feed(self, mono: np.ndarray):
        resampled = _resample_to_16k(mono, self.native_rate)
        buf = np.concatenate([self.carry, resampled])
        n_blocks = len(buf) // BLOCK_SIZE
        for i in range(n_blocks):
            block = buf[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            pcm16 = (block * 32768.0).astype(np.int16)

            now = time.monotonic()
            if now - self.last_meter > 1.0:
                rms = float(np.sqrt(np.mean(pcm16.astype(np.float64) ** 2)))
                bar_len = min(40, int(rms / 32768.0 * 400))
                marker = "SPEECH" if rms > self.segmenter.silence_threshold else "silence"
                print(f"level: {'#' * bar_len:<40} {rms:7.0f}/32768  [{marker}]")
                self.last_meter = now

            segment = self.segmenter.push(pcm16)
            if segment is not None and len(segment) > SAMPLE_RATE * 0.3:
                print(f"-> speech segment captured ({len(segment) / SAMPLE_RATE:.1f}s), transcribing...")
                self.audio_q.put(segment)
        self.carry = buf[n_blocks * BLOCK_SIZE:]


def audio_capture_loop(device_name: str, segmenter: SpeechSegmenter,
                        audio_q: "queue.Queue[np.ndarray]", stop_event: threading.Event,
                        loopback: bool):
    mic, native_rate = _resolve_recorder(device_name, loopback)
    native_block = int(BLOCK_SIZE * native_rate / SAMPLE_RATE)

    print(f"Capturing from '{device_name}' (loopback={loopback}) at {native_rate} Hz. "
          f"Silence threshold: {segmenter.silence_threshold:.0f}/32768.")

    feeder = BlockFeeder(segmenter, audio_q, native_rate)
    try:
        with mic.recorder(samplerate=native_rate) as rec:
            while not stop_event.is_set():
                frames = rec.record(numframes=native_block)
                mono = frames.mean(axis=1) if frames.ndim > 1 and frames.shape[1] > 1 else frames[:, 0]
                feeder.feed(mono.astype(np.float32))
    except Exception as e:
        print(f"Audio capture error: {e}")
        raise


def app_capture_loop(pid: int, app_name: str, segmenter: SpeechSegmenter,
                      audio_q: "queue.Queue[np.ndarray]", stop_event: threading.Event):
    """Capture the audio of one process only, via WASAPI process loopback."""
    native_rate = 48000  # proctap always delivers 48 kHz / 2ch / float32
    read_timeout = 0.2

    print(f"Capturing app '{app_name}' (PID {pid}) via process loopback. "
          f"Silence threshold: {segmenter.silence_threshold:.0f}/32768.")

    feeder = BlockFeeder(segmenter, audio_q, native_rate)
    cap = ProcessAudioCapture(pid=pid)
    cap.start()
    try:
        while not stop_event.is_set():
            chunk = cap.read(timeout=read_timeout)
            if chunk is None:
                # The app is silent (or paused): synthesize silence so open
                # speech segments still get flushed after min-silence-ms.
                mono = np.zeros(int(read_timeout * native_rate), dtype=np.float32)
            else:
                arr = np.frombuffer(chunk, dtype=np.float32)
                mono = arr.reshape(-1, 2).mean(axis=1).astype(np.float32)
            feeder.feed(mono)
    except Exception as e:
        print(f"App capture error: {e}")
        raise
    finally:
        cap.close()


def main():
    parser = argparse.ArgumentParser(description="Live local subtitles via microphone, speaker loopback, or a single app.")
    parser.add_argument("--device", type=str, default=None,
                         help="Exact device name (run list_devices.py to list them). Not needed "
                              "when capturing a single app via --app or the picker window.")
    parser.add_argument("--loopback", action="store_true",
                         help="Capture what plays out of an OUTPUT device (e.g. headphones/"
                              "speakers) via WASAPI loopback, instead of recording an input "
                              "device. Use this to caption audio she's hearing without any "
                              "virtual cable, so her own playback isn't rerouted or muted.")
    parser.add_argument("--app", type=str, default=None,
                         help="Capture the audio of one application only (WASAPI process "
                              "loopback), e.g. --app Discord. Case-insensitive substring of the "
                              "process name. The app must be running with an audio session. "
                              "Overrides --device/--loopback.")
    parser.add_argument("--model", default="small",
                         help="faster-whisper model size: tiny/base/small/medium/large-v3 "
                              "(default: small -- fast and accurate enough for live captions on "
                              "most CUDA GPUs; use medium/large-v3 if you want more accuracy and can "
                              "tolerate more lag). Ignored if --no-model-picker is not set, since "
                              "the picker window lets you choose this interactively instead.")
    parser.add_argument("--no-model-picker", action="store_true",
                         help="Skip the startup model-choice window and use --model directly.")
    parser.add_argument("--compute-type", default="float16",
                         help="float16 (recommended on most CUDA GPUs), int8_float16, or int8.")
    parser.add_argument("--beam-size", type=int, default=1,
                         help="Whisper beam search width. 1 (default) = greedy decoding, "
                              "fastest. Higher (e.g. 5) is a bit more accurate but slower.")
    parser.add_argument("--language", default="auto",
                         help="Lock the caption language (e.g. ru or en) to skip per-segment "
                              "language detection -- noticeably faster, but mixed-language "
                              "speech will be forced into this one language. Default: auto.")
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

    # Decide model + audio source (CLI flags, or the startup picker window).
    chosen_model = args.model
    app_target = None  # (pid, name) for single-app capture, None for device capture
    if args.app:
        if not HAS_APP_CAPTURE:
            raise SystemExit(
                "Per-app capture needs the proc-tap native extension, which has "
                "prebuilt wheels for Python 3.10-3.13 only. Recreate the venv with "
                "one of those versions, or capture the whole system instead."
            )
        app_target = find_app_by_name(args.app)
    if not args.no_model_picker:
        chosen_model, picked_source = choose_options_interactively(args.model)
        print(f"Model chosen: {chosen_model}")
        if args.app is None:
            app_target = picked_source
        if app_target:
            print(f"Source chosen: app '{app_target[1]}' (PID {app_target[0]})")
        else:
            print("Source chosen: whole system (device loopback)")

    if app_target is None and args.device is None:
        raise SystemExit(
            "No audio source: pass --device \"<name>\" (see list_devices.py) for "
            "whole-system capture, or --app <name> / pick an app in the window."
        )

    print(f"Loading faster-whisper model '{chosen_model}' on GPU ...")
    model = WhisperModel(chosen_model, device="cuda", compute_type=args.compute_type)
    # Warm-up: the very first inference triggers CUDA kernel/cuDNN
    # initialization and would otherwise delay the first real caption.
    warmup_t0 = time.monotonic()
    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1)[0])
    print(f"Model loaded and warmed up ({time.monotonic() - warmup_t0:.1f}s). Listening ...")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    text_q: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    segmenter = SpeechSegmenter(args.silence_threshold, args.min_silence_ms, args.max_segment_s)

    t_worker = threading.Thread(
        target=transcriber_worker,
        args=(model, audio_q, text_q, args.beam_size, args.language),
        daemon=True,
    )
    t_worker.start()

    if app_target is not None:
        t_capture = threading.Thread(
            target=app_capture_loop,
            args=(app_target[0], app_target[1], segmenter, audio_q, stop_event),
            daemon=True,
        )
    else:
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
