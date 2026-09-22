"""Audio decoding and normalisation.

Everything downstream (Whisper) expects mono ``float32`` PCM at 16 kHz.

Supported inputs:

* WAV / FLAC / OGG / MP3 — decoded with ``soundfile`` (libsndfile ≥ 1.1 has MP3).
* M4A / AAC / anything else — decoded through ``ffmpeg`` if it is on ``PATH``.
  Without ffmpeg a clear :class:`AudioDecodeError` is raised instead of a
  silent failure.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_SAMPLE_RATE = 16_000
SOUNDFILE_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"})
FFMPEG_EXTENSIONS: frozenset[str] = frozenset({".m4a", ".aac", ".mp4", ".webm", ".wma", ".opus"})


class AudioDecodeError(RuntimeError):
    """Raised when an audio file cannot be decoded."""


@dataclass(slots=True)
class AudioClip:
    """Decoded mono audio."""

    samples: np.ndarray  # float32, shape (n,)
    sample_rate: int
    source: str

    @property
    def duration_s(self) -> float:
        return float(len(self.samples)) / self.sample_rate if self.sample_rate else 0.0


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels of a ``(n, channels)`` array; pass 1-D through."""
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1).astype(np.float32)


def _lowpass_fir(cutoff_ratio: float, taps: int = 63) -> np.ndarray:
    """Windowed-sinc low-pass filter with ``cutoff_ratio`` in cycles/sample (0-0.5)."""
    n = np.arange(taps) - (taps - 1) / 2
    kernel = 2 * cutoff_ratio * np.sinc(2 * cutoff_ratio * n)
    kernel *= np.hamming(taps)
    return (kernel / kernel.sum()).astype(np.float32)


def resample(
    samples: np.ndarray, orig_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Resample mono audio using an anti-aliased linear interpolation.

    This is a dependency-free resampler adequate for speech recognition. When
    downsampling, a windowed-sinc low-pass filter removes content above the
    new Nyquist frequency before interpolation to avoid aliasing.
    """
    if orig_rate == target_rate or len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    samples = samples.astype(np.float32, copy=False)
    if target_rate < orig_rate:
        cutoff = 0.5 * target_rate / orig_rate * 0.9  # slight margin below Nyquist
        samples = np.convolve(samples, _lowpass_fir(cutoff), mode="same")
    duration = len(samples) / orig_rate
    n_out = int(round(duration * target_rate))
    src_positions = np.arange(len(samples)) / orig_rate
    dst_positions = np.arange(n_out) / target_rate
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


class AudioLoader:
    """Decode audio files (or in-memory bytes) into 16 kHz mono clips."""

    def __init__(self, target_sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        self.target_sample_rate = target_sample_rate

    @staticmethod
    def supports(path: Path) -> bool:
        return path.suffix.lower() in SOUNDFILE_EXTENSIONS | FFMPEG_EXTENSIONS

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def load(self, path: Path) -> AudioClip:
        """Decode ``path`` into a normalised :class:`AudioClip`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        try:
            if suffix in SOUNDFILE_EXTENSIONS:
                samples, rate = self._decode_soundfile(path.read_bytes())
            elif suffix in FFMPEG_EXTENSIONS:
                samples, rate = self._decode_ffmpeg(path)
            else:
                raise AudioDecodeError(f"Unsupported audio format '{suffix}' for {path.name}")
        except AudioDecodeError:
            raise
        except Exception as exc:
            # soundfile may not decode e.g. an MP3 on older libsndfile: try ffmpeg.
            if self.ffmpeg_available():
                logger.warning(
                    "soundfile failed on %s (%s); falling back to ffmpeg", path.name, exc
                )
                samples, rate = self._decode_ffmpeg(path)
            else:
                raise AudioDecodeError(f"Cannot decode {path.name}: {exc}") from exc
        return self._finalise(samples, rate, path.name)

    def load_bytes(self, data: bytes, filename: str = "upload.wav") -> AudioClip:
        """Decode audio from raw bytes (UI uploads / API requests)."""
        suffix = Path(filename).suffix.lower()
        if suffix in SOUNDFILE_EXTENSIONS or not suffix:
            try:
                samples, rate = self._decode_soundfile(data)
                return self._finalise(samples, rate, filename)
            except Exception as exc:
                if not self.ffmpeg_available():
                    raise AudioDecodeError(f"Cannot decode {filename}: {exc}") from exc
        if not self.ffmpeg_available():
            raise AudioDecodeError(
                f"Decoding '{suffix}' requires ffmpeg on PATH (not found). Upload WAV/FLAC/MP3 instead."
            )
        samples, rate = self._decode_ffmpeg_bytes(data)
        return self._finalise(samples, rate, filename)

    # --- Internals --------------------------------------------------------

    def _finalise(self, samples: np.ndarray, rate: int, source: str) -> AudioClip:
        mono = to_mono(samples)
        resampled = resample(mono, rate, self.target_sample_rate)
        clip = AudioClip(samples=resampled, sample_rate=self.target_sample_rate, source=source)
        logger.info(
            "Loaded %s: %.2fs @ %d Hz (original %d Hz)",
            source,
            clip.duration_s,
            clip.sample_rate,
            rate,
        )
        return clip

    @staticmethod
    def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int]:
        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(samples), int(rate)

    def _decode_ffmpeg(self, path: Path) -> tuple[np.ndarray, int]:
        return self._run_ffmpeg(["-i", str(path)], stdin=None)

    def _decode_ffmpeg_bytes(self, data: bytes) -> tuple[np.ndarray, int]:
        return self._run_ffmpeg(["-i", "pipe:0"], stdin=data)

    def _run_ffmpeg(self, input_args: list[str], stdin: bytes | None) -> tuple[np.ndarray, int]:
        if not self.ffmpeg_available():
            raise AudioDecodeError("ffmpeg is required for this audio format but is not installed")
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            *input_args,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(self.target_sample_rate),
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True, check=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            raise AudioDecodeError(f"ffmpeg failed: {exc.stderr.decode(errors='replace')}") from exc
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        return samples, self.target_sample_rate
