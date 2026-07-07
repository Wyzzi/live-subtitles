"""List audio devices for live-subtitles.

Shows two things:
  - Speakers/headphones: use one of these NAMES with --loopback to caption
    whatever she's actually hearing, without touching audio routing (so she
    keeps hearing everything normally, no virtual cable needed).
  - Microphones: use one of these NAMES without --loopback, e.g. to caption
    her own mic, or a virtual cable if you've set one up for some other
    reason.
"""
import soundcard as sc

print("=== Speakers / headphones (use with --loopback to caption what she hears) ===\n")
default_speaker = sc.default_speaker()
for spk in sc.all_speakers():
    default = "  (default)" if spk.name == default_speaker.name else ""
    print(f"- {spk.name}{default}")

print("\n=== Microphones (use WITHOUT --loopback) ===\n")
default_mic = sc.default_microphone()
for mic in sc.all_microphones():
    default = "  (default)" if mic.name == default_mic.name else ""
    print(f"- {mic.name}{default}")

print("\nTip: for captioning Discord / videos while she keeps hearing audio normally,")
print("copy the exact name of her headphones/speakers from the list above and run:")
print('    subtitles.py --device "<exact name>" --loopback')
