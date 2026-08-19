"""Continuous shuffled playback with equal-power crossfades and a gapless, reshuffling loop."""

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
import miniaudio

SAMPLE_RATE = 44100
CHANNELS = 2
DEFAULT_CROSSFADE_SECONDS = 5.0
MIN_CROSSFADE_SECONDS = 0.0
MAX_CROSSFADE_SECONDS = 10.0
BLOCK_FRAMES = 8192
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}


def _equal_power_curves(n):
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    fade_out = np.cos(t * np.pi / 2).astype(np.float32)
    fade_in = np.sin(t * np.pi / 2).astype(np.float32)
    return fade_out, fade_in


def decode_track(path):
    """Decode an audio file to a (frames, CHANNELS) float32 numpy array at SAMPLE_RATE."""
    sound = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=CHANNELS,
        sample_rate=SAMPLE_RATE,
    )
    arr = np.frombuffer(sound.samples, dtype=np.float32)
    return arr.reshape(-1, CHANNELS)


def find_audio_files(folder):
    folder = Path(folder)
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def _natural_sort_key(path):
    """Alphabetical, but numeric-aware: 'track2' sorts before 'track10'."""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", path.name)
    ]


class ShuffleCrossfadePlayer:
    """Plays a folder of audio files in reshuffled random order, forever, with a
    fixed-length equal-power crossfade at every track transition (including the
    wrap from the last track of one pass back into the first track of the next)."""

    def __init__(self):
        self.files = []
        self.volume = 1.0
        self.crossfade_seconds = DEFAULT_CROSSFADE_SECONDS
        self.shuffle_enabled = True

        self._thread = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._skip_event = threading.Event()
        self._seek_target_frames = None
        self._lock = threading.Lock()

        self.now_playing = None
        self.up_next = None
        self.status = "Idle"
        self.error = None

        # Position/duration only ever reflect the *currently playing* track, and
        # both reset to 0 the moment a crossfade finishes and the next track
        # becomes current - deliberately not accounting for the portion of that
        # track already blended in during the crossfade, so the displayed
        # progress bar always matches what's audibly "the new track" rather than
        # needing to track time consumed by an earlier transition.
        self.position_seconds = 0.0
        self.duration_seconds = 0.0

        # Incremented each time a new track-to-track crossfade begins, alongside
        # the wall-clock start time and actual duration of that crossfade, so a
        # UI can sync its own animations (e.g. a background image crossfade) to
        # the audio crossfade without needing sample-accurate coupling.
        self.crossfade_event_id = 0
        self.crossfade_event_time = 0.0
        self.crossfade_event_duration = 0.0

    # -- public control API -------------------------------------------------

    def set_folder(self, folder):
        files = find_audio_files(folder)
        if not files:
            raise ValueError(f"No .mp3 or .wav files found in {folder}")
        self.files = files

    def start(self):
        if self.is_running():
            return
        if not self.files:
            raise ValueError("No folder loaded")
        self._stop_event.clear()
        self._paused.clear()
        self._skip_event.clear()
        self._seek_target_frames = None
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._paused.clear()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self.now_playing = None
        self.up_next = None
        self.position_seconds = 0.0
        self.duration_seconds = 0.0
        self.status = "Stopped"

    def pause(self):
        self._paused.set()
        self.status = "Paused"

    def resume(self):
        self._paused.clear()
        self.status = "Playing"

    def toggle_pause(self):
        if self._paused.is_set():
            self.resume()
        else:
            self.pause()

    def skip(self):
        if self.is_running():
            self._skip_event.set()

    def seek(self, seconds):
        """Jump to a position (in seconds) within the currently playing track.
        Seeking into the last `crossfade_seconds` of the track behaves like
        Skip: it immediately begins crossfading into the next track from there."""
        if self.is_running():
            self._seek_target_frames = max(0, int(seconds * SAMPLE_RATE))

    def set_crossfade_seconds(self, seconds):
        self.crossfade_seconds = min(MAX_CROSSFADE_SECONDS, max(MIN_CROSSFADE_SECONDS, seconds))

    def set_shuffle(self, enabled):
        self.shuffle_enabled = enabled

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def is_paused(self):
        return self._paused.is_set()

    # -- internal -------------------------------------------------------------

    def _infinite_shuffled_paths(self):
        while True:
            order = self.files.copy()
            random.shuffle(order)
            for f in order:
                yield f

    def _infinite_sequential_paths(self):
        order = sorted(self.files, key=_natural_sort_key)
        while True:
            for f in order:
                yield f

    def _wait_if_paused(self):
        while self._paused.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)

    def _write(self, stream, block):
        if block.shape[0] == 0:
            return
        if self.volume != 1.0:
            block = block * self.volume
        stream.write(np.ascontiguousarray(block, dtype=np.float32))

    def _decode_with_fallback(self, path_iter, executor=None):
        """Try decoding paths from path_iter until one succeeds; returns (path, audio, future_or_None)."""
        for _ in range(len(self.files) + 1):
            path = next(path_iter)
            try:
                if executor is not None:
                    future = executor.submit(decode_track, path)
                    return path, future
                audio = decode_track(path)
                return path, audio
            except Exception as exc:
                self.error = f"Skipping unreadable file {path.name}: {exc}"
                continue
        raise RuntimeError("No decodable audio files found")

    def _run(self):
        path_iter = self._infinite_shuffled_paths() if self.shuffle_enabled else self._infinite_sequential_paths()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.status = "Loading..."
            current_path, current_audio = self._decode_with_fallback(path_iter)
            next_path, next_future = self._decode_with_fallback(path_iter, executor)
            # How many leading frames of `current_audio` were already played as
            # part of the previous transition's crossfade blend (0 for the very
            # first track, which had no incoming crossfade).
            carried_offset = 0

            stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
            stream.start()
            self.status = "Playing"

            try:
                while not self._stop_event.is_set():
                    self.now_playing = current_path.name
                    self.up_next = next_path.name

                    crossfade_frames = int(self.crossfade_seconds * SAMPLE_RATE)

                    pos = carried_offset
                    n = current_audio.shape[0]
                    body_end = max(pos, n - crossfade_frames)
                    self.duration_seconds = n / SAMPLE_RATE
                    self.position_seconds = pos / SAMPLE_RATE

                    while pos < body_end and not self._stop_event.is_set():
                        if self._skip_event.is_set():
                            self._skip_event.clear()
                            body_end = pos
                            break
                        self._wait_if_paused()
                        if self._stop_event.is_set():
                            break
                        if self._seek_target_frames is not None:
                            target = self._seek_target_frames
                            self._seek_target_frames = None
                            pos = min(max(0, target), n)
                            self.position_seconds = pos / SAMPLE_RATE
                            if pos >= body_end:
                                body_end = pos
                                break
                            continue
                        end = min(pos + BLOCK_FRAMES, body_end)
                        self._write(stream, current_audio[pos:end])
                        pos = end
                        self.position_seconds = pos / SAMPLE_RATE

                    if self._stop_event.is_set():
                        break

                    fade_out_src = current_audio[pos:pos + crossfade_frames]

                    try:
                        next_audio = next_future.result()
                    except Exception as exc:
                        self.error = f"Skipping unreadable file {next_path.name}: {exc}"
                        next_path, next_future = self._decode_with_fallback(path_iter, executor)
                        next_audio = next_future.result()

                    final_len = min(fade_out_src.shape[0], next_audio.shape[0])
                    fade_out_curve, fade_in_curve = _equal_power_curves(final_len)
                    mixed = (
                        fade_out_src[:final_len] * fade_out_curve[:, None]
                        + next_audio[:final_len] * fade_in_curve[:, None]
                    )
                    self.crossfade_event_time = time.time()
                    self.crossfade_event_duration = final_len / SAMPLE_RATE
                    self.crossfade_event_id += 1
                    self._write(stream, mixed)

                    carried_offset = final_len
                    current_audio = next_audio
                    current_path = next_path
                    next_path, next_future = self._decode_with_fallback(path_iter, executor)
            finally:
                stream.stop()
                stream.close()
        except Exception as exc:
            self.error = str(exc)
            self.status = "Error"
        finally:
            executor.shutdown(wait=False)
            if not self._stop_event.is_set():
                self.status = "Stopped"
