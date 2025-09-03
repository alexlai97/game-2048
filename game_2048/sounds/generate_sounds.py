#!/usr/bin/env python3
"""
Simple sound generator for 2048 game.
Creates basic sine wave sounds for different game events.
"""

import os
import wave

import numpy as np


def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.3):
    """Generate a sine wave tone."""
    frames = int(duration * sample_rate)
    arr = amplitude * np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))

    # Apply envelope to avoid clicks
    fade_frames = int(0.01 * sample_rate)  # 10ms fade
    if fade_frames < frames // 2:
        # Fade in
        arr[:fade_frames] *= np.linspace(0, 1, fade_frames)
        # Fade out
        arr[-fade_frames:] *= np.linspace(1, 0, fade_frames)

    return arr


def generate_chord(frequencies, duration, sample_rate=44100, amplitude=0.2):
    """Generate a chord from multiple frequencies."""
    frames = int(duration * sample_rate)
    arr = np.zeros(frames)

    for freq in frequencies:
        tone = amplitude * np.sin(2 * np.pi * freq * np.linspace(0, duration, frames))
        arr += tone

    # Normalize
    arr = arr / len(frequencies)

    # Apply envelope
    fade_frames = int(0.01 * sample_rate)  # 10ms fade
    if fade_frames < frames // 2:
        arr[:fade_frames] *= np.linspace(0, 1, fade_frames)
        arr[-fade_frames:] *= np.linspace(1, 0, fade_frames)

    return arr


def save_wav(filename, audio_data, sample_rate=44100):
    """Save audio data as a WAV file."""
    # Convert to 16-bit integers
    audio_data = np.int16(audio_data * 32767)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def main():
    """Generate all sound files for the 2048 game."""
    sounds_dir = os.path.dirname(__file__)

    print("Generating sound files...")

    # Move sound - gentle whoosh (quick pitch slide)
    print("  move.wav")
    move_sound = np.concatenate(
        [generate_tone(400, 0.05), generate_tone(350, 0.05), generate_tone(300, 0.05)]
    )
    save_wav(os.path.join(sounds_dir, "move.wav"), move_sound)

    # Merge sound - satisfying chord
    print("  merge.wav")
    merge_sound = generate_chord([440, 554, 659], 0.2)  # A major chord
    save_wav(os.path.join(sounds_dir, "merge.wav"), merge_sound)

    # Win sound - triumphant ascending notes
    print("  win.wav")
    win_notes = [
        generate_tone(523, 0.15),  # C5
        generate_tone(659, 0.15),  # E5
        generate_tone(784, 0.15),  # G5
        generate_tone(1047, 0.3),  # C6
    ]
    win_sound = np.concatenate(win_notes)
    save_wav(os.path.join(sounds_dir, "win.wav"), win_sound)

    # Game over sound - descending minor chord
    print("  lose.wav")
    lose_notes = [
        generate_chord([440, 523, 622], 0.2),  # A minor chord
        generate_chord([415, 494, 587], 0.3),  # G# minor chord (lower)
    ]
    lose_sound = np.concatenate(lose_notes)
    save_wav(os.path.join(sounds_dir, "lose.wav"), lose_sound)

    # Spawn sound - quick high ping (optional)
    print("  spawn.wav")
    spawn_sound = generate_tone(800, 0.1, amplitude=0.15)
    save_wav(os.path.join(sounds_dir, "spawn.wav"), spawn_sound)

    print("Sound generation complete!")


if __name__ == "__main__":
    main()
