# Live Local Subtitles

Real-time, fully local speech-to-text subtitles for videos, games, or Discord
calls — built so a deaf person can follow along by reading captions in the
language actually being spoken (Russian audio -> Russian captions, English
audio -> English captions, auto-detected per segment). Nothing is sent to the
cloud; the model runs locally on any NVIDIA GPU with CUDA cores.

## Описание

Приложение для отображения субтитров в реальном времени для видео, игр или
звонков в Discord — сделано, чтобы человек с нарушением слуха мог читать
распознанную речь на том языке, на котором она произносится (русская речь ->
русские субтитры, английская речь -> английские субтитры, язык определяется
автоматически для каждого фрагмента). Всё работает полностью локально, без
отправки чего-либо в интернет; модель распознавания речи выполняется на
видеокарте NVIDIA с поддержкой CUDA (подойдёт практически любая современная
видеокарта NVIDIA — GTX 10-й серии и новее, вся линейка RTX).

Как это работает:

1. Приложение захватывает звук прямо с **колонок/наушников** через WASAPI
   loopback — оно просто "слушает" то, что уже воспроизводится, точно так же,
   как это делают программы записи экрана со звуком. Виртуальный кабель не
   нужен, звук никуда не перенаправляется: человек продолжает слышать
   Discord/видео совершенно нормально через свои обычные наушники.
2. Каждый фрагмент речи отправляется в локальную модель
   [faster-whisper](https://github.com/SYSTRAN/faster-whisper), которая
   работает на видеокарте (CUDA).
3. Распознанный текст показывается в небольшой полупрозрачной панели поверх
   всех окон, которую можно перетаскивать в удобное место.

**Запуск через bat-файлы (проще всего, без командной строки):**

- `setup.bat` — разовая установка: создаёт виртуальное окружение и ставит все
  зависимости. Запускать один раз (и повторно — после обновления кода).
- `list_devices.bat` — показывает список звуковых устройств (колонок,
  наушников, микрофонов), чтобы найти точное имя нужного устройства.
- `run_subtitles.bat` — запускает субтитры. При первом запуске попросит имя
  устройства (можно один раз ввести вручную, либо прописать в файле в
  строке `DEVICE=`, чтобы больше не вводить).

Просто дважды щёлкните по каждому файлу по очереди — `setup.bat`, затем
`list_devices.bat`, затем `run_subtitles.bat`.

## How it works

1. This app captures audio directly from the **speakers/headphones** using
   WASAPI loopback — it just listens to whatever is already playing, the
   same way a screen recorder captures "what you hear." No virtual cable, no
   rerouting: the listener keeps hearing Discord/video audio completely
   normally through their real headphones.
2. Each speech chunk is sent to a local
   [faster-whisper](https://github.com/SYSTRAN/faster-whisper) model running
   on the GPU (CUDA).
3. The recognized text is shown in a small always-on-top, draggable subtitle
   bar that can be positioned over the video or call window.

(A virtual cable is only needed for the opposite direction — capturing what
a *microphone* says. See "Captioning a microphone instead" below.)

## 1. Install

Requires an NVIDIA GPU with CUDA cores and a working CUDA driver (already
present if you game/use CUDA apps — practically any GeForce GTX 10-series or
newer, including the whole RTX lineup, works) and Python 3.10+.

**Easiest way:** double-click `setup.bat`. It creates a virtual environment
and installs everything needed. Re-run it any time after updating the code
to pick up new dependencies.

Or manually:

```powershell
cd live-subtitles
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`faster-whisper` (via `ctranslate2`) needs the CUDA and cuDNN runtime DLLs.
These are installed automatically as regular pip packages
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) by `requirements.txt` — no manual
CUDA toolkit install required. If you still get a "cannot load cudnn/cublas"
error the first time you run it, try:

```powershell
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```

## 2. Find the device name

Double-click `list_devices.bat` (or run `python list_devices.py`).

This lists the speakers/headphones (for `--loopback`) and microphones (for
normal mic capture). Copy the **exact name** shown.

## 3. Run it

**Easiest way:** double-click `run_subtitles.bat`. The first time, it will
ask you to type the device name found in step 2; you can also edit
`DEVICE=` inside the file so it's remembered for next time. It already has
`LOOPBACK=1` set by default.

Or manually:

```powershell
python subtitles.py --device "Headset Earphone (Your Headset Name)" --loopback
```

(replace with the exact name from step 2 — quote it, names often contain
spaces)

As soon as it launches, a small window pops up asking you to choose a model:

- **Fast (`small`)** — quicker captions, slightly less accurate. Best for
  live conversation (Discord calls).
- **Accurate (`large-v3`)** — better quality, more delay per caption. Best
  for movies/lectures where a second or two of extra lag doesn't matter.

Whichever one is picked gets downloaded on first use (`small` ~500MB,
`large-v3` ~3GB) and cached — after that it works fully offline. To skip this
window and always use a fixed model, pass `--no-model-picker --model <name>`.

A subtitle bar appears near the bottom of the screen. Drag it with the mouse
to reposition; press `Esc` to close it.

### Useful options

| Flag | Meaning |
|---|---|
| `--loopback` | Capture what plays out of the named device (speakers/headphones) instead of recording it as a microphone. This is what you want for captioning Discord/video audio without affecting playback. |
| `--model` | `tiny`, `base`, `small` (default), `medium`, `large-v3`. Bigger = more accurate but slower. `small` gives the best speed/accuracy balance for live captions on most GPUs; go up to `medium`/`large-v3` only if accuracy matters more than latency to you (also selectable at launch via the model-picker window). |
| `--compute-type` | `float16` (default, works well on any CUDA GPU), or `int8_float16` for less VRAM/more speed at a small accuracy cost. |
| `--beam-size` | `1` (default) = greedy decoding, fastest. Raise to e.g. `5` for slightly better accuracy at the cost of speed. |
| `--silence-threshold` | Fraction of full scale (0-1) below which audio counts as silence. Raise if background noise/music triggers false captions; lower if quiet speech is missed. |
| `--min-silence-ms` | How much silence ends a caption segment (default `350`ms). Lower = captions appear sooner after each pause. |
| `--max-segment-s` | Hard cap on how long continuous speech is buffered before it's transcribed (default `6`s). Lower this further (e.g. `4`) for even more frequent updates during long uninterrupted speech, at the cost of each chunk having less context for Whisper to work with. |
| `--min-hold-s` | Minimum seconds each caption stays on screen before the next one can replace it (default `1.2`), so a quick follow-up line doesn't instantly wipe out the previous one before it can be read. Raise if captions still feel like they flash by too fast. |
| `--font-size`, `--opacity` | Overlay appearance. |

### If it still feels slow

- Try `--model tiny` or `--model base` for the fastest possible turnaround
  (less accurate, especially on quieter/accented speech).
- The console prints how long each transcription took, e.g.
  `[ru] (0.4s) привет как дела` — if that number is consistently high, the
  model itself is the bottleneck and a smaller `--model` will help most.
  If it's low but captions still feel delayed, try lowering
  `--max-segment-s` and `--min-silence-ms` instead.
- Weaker GPUs will need a smaller model/compute-type than a high-end one for
  the same latency — there's no one-size-fits-all setting.

## Captioning a microphone instead

To caption someone talking into a real microphone near the PC (e.g. a
hearing friend speaking face-to-face, or a mic for your own reference)
rather than app/Discord audio, pick a microphone name from `list_devices.py`
and run **without** `--loopback`:

```powershell
python subtitles.py --device "Microphone (Your Mic Name)"
```

A virtual audio cable (e.g. VB-Audio Virtual Cable) is only needed if you
specifically want to route one app's *output* into another app's *microphone
input* — not for the captioning use case above, since loopback already
captures speaker output directly.

## Notes

- Language is auto-detected per speech segment by Whisper itself — no
  translation happens, so mixed-language conversations (e.g. Discord friends
  switching between Russian and English) are captioned correctly in
  whichever language was spoken.
- Everything (audio + model + text) stays on the local machine.
