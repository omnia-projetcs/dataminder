"""
Audio and video transcription module using faster-whisper.

Provides speech-to-text transcription for audio files (mp3, wav, ogg, flac,
m4a, wma, aac) and video files (mp4, mkv, avi, mov, webm, flv, wmv).
Video files have their audio track extracted via ffmpeg before transcription.
"""

import os
import subprocess
import tempfile
from logger import log_error as _shared_log_error

# Lazy-loaded faster-whisper model (singleton to avoid reloading)
_whisper_model_instance = None
_whisper_model_size = None
_whisper_device = None
_whisper_compute_type = None

# Supported extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.wma', '.aac'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}


def _log_error(filepath, error_msg):
    """Log transcription errors to data/error.log."""
    _shared_log_error(filepath, error_msg, category="TRANSCRIPTION")


def _get_model(model_size="base", device="cpu"):
    """
    Get or create the faster-whisper model (singleton pattern).

    The model is loaded once and reused across all transcription calls
    to avoid repeated downloads and loading overhead.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        device: "cpu" or "cuda".
    """
    global _whisper_model_instance, _whisper_model_size
    global _whisper_device, _whisper_compute_type

    compute_type = "int8" if device == "cpu" else "float16"

    # Return cached model if config hasn't changed
    if (_whisper_model_instance is not None
            and _whisper_model_size == model_size
            and _whisper_device == device
            and _whisper_compute_type == compute_type):
        return _whisper_model_instance

    from faster_whisper import WhisperModel

    print(f"[Whisper] Loading model '{model_size}' on {device} ({compute_type})...")
    _whisper_model_instance = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    _whisper_model_size = model_size
    _whisper_device = device
    _whisper_compute_type = compute_type
    print(f"[Whisper] Model '{model_size}' ready.")
    return _whisper_model_instance


def _check_ffmpeg():
    """Check that ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _extract_audio_from_video(video_path, output_path):
    """
    Extract the audio track from a video file using ffmpeg.

    Converts to WAV (16kHz mono) which is optimal for Whisper.

    Args:
        video_path: Path to the input video file.
        output_path: Path for the output WAV file.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn",                  # No video
                "-acodec", "pcm_s16le", # 16-bit PCM WAV
                "-ar", "16000",         # 16kHz (Whisper native rate)
                "-ac", "1",             # Mono
                "-y",                   # Overwrite output
                output_path
            ],
            capture_output=True, check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        _log_error(video_path, f"ffmpeg audio extraction failed: {stderr[:500]}")
        return False


def transcribe_audio(path, model_size="base", device="cpu", language=None):
    """
    Transcribe an audio file to text using faster-whisper.

    Args:
        path: Path to the audio file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        device: "cpu" or "cuda".
        language: Language code (e.g., "en", "fr"). None for auto-detection.

    Returns:
        Transcribed text as a string.
    """
    try:
        model = _get_model(model_size=model_size, device=device)

        # Transcription parameters
        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,          # Filter out silence
            "vad_parameters": {
                "min_silence_duration_ms": 500,
            },
        }
        if language:
            transcribe_kwargs["language"] = language

        segments, info = model.transcribe(path, **transcribe_kwargs)

        # Collect all segments
        text_parts = []
        detected_lang = info.language
        lang_prob = info.language_probability

        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)

        if not full_text.strip():
            _log_error(path, "Transcription produced empty output.")
            return ""

        # Add metadata header
        duration_str = _format_duration(info.duration)
        header = (
            f"<!-- Transcription: {os.path.basename(path)} | "
            f"Language: {detected_lang} ({lang_prob:.0%}) | "
            f"Duration: {duration_str} | "
            f"Model: whisper-{model_size} -->\n\n"
        )

        return header + full_text

    except Exception as e:
        _log_error(path, f"Transcription failed: {e}")
        return ""


def transcribe_video(path, model_size="base", device="cpu", language=None):
    """
    Transcribe a video file by extracting its audio track first.

    Requires ffmpeg to be installed on the system.

    Args:
        path: Path to the video file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        device: "cpu" or "cuda".
        language: Language code (e.g., "en", "fr"). None for auto-detection.

    Returns:
        Transcribed text as a string.
    """
    if not _check_ffmpeg():
        _log_error(path, "ffmpeg is not installed. Required for video transcription.")
        print(f"[{path}] ERROR: ffmpeg is required for video transcription. "
              "Install it with: sudo apt-get install ffmpeg")
        return ""

    try:
        # Extract audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio_path = tmp.name

        print(f"[{os.path.basename(path)}] Extracting audio track with ffmpeg...")
        if not _extract_audio_from_video(path, tmp_audio_path):
            return ""

        # Check if audio was actually extracted (file size > 0)
        if os.path.getsize(tmp_audio_path) < 1024:
            _log_error(path, "Extracted audio file is empty. Video may have no audio track.")
            print(f"[{os.path.basename(path)}] No audio track found in video.")
            return ""

        # Transcribe the extracted audio
        result = transcribe_audio(
            tmp_audio_path,
            model_size=model_size,
            device=device,
            language=language
        )
        return result

    except Exception as e:
        _log_error(path, f"Video transcription failed: {e}")
        return ""

    finally:
        # Clean up temporary file
        if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def _format_duration(seconds):
    """Format a duration in seconds to HH:MM:SS or MM:SS."""
    if seconds is None or seconds <= 0:
        return "unknown"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
