"""ffmpeg-based video processing utilities."""
from __future__ import annotations

import os
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec_video: str
    codec_audio: str


class VideoProcessor:
    """ffmpeg wrapper for video/audio extraction and assembly."""

    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> str:
        """Extract audio from video as WAV (16kHz mono by default for ASR)."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ar", str(sample_rate),
                "-ac", str(channels),
                "-vn",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def extract_reference_audio(
        self,
        video_path: str,
        output_path: str,
        start_sec: float = 0.0,
        duration_sec: float = 10.0,
        sample_rate: int = 22050,
    ) -> str:
        """Extract a clean speech segment for XTTS-v2 voice cloning reference."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(start_sec),
                "-t", str(duration_sec),
                "-ar", str(sample_rate),
                "-ac", "1",
                "-vn",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        reencode_video: bool = False,
    ) -> str:
        """Replace video audio track with new audio."""
        video_codec = "libx264" if reencode_video else "copy"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", video_codec,
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        fps: Optional[float] = None,
    ) -> list[str]:
        """Extract video frames as PNG images."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(out_dir / "frame_%06d.png")
        cmd = ["ffmpeg", "-y", "-i", video_path]
        if fps:
            cmd += ["-vf", f"fps={fps}"]
        cmd.append(pattern)
        subprocess.run(cmd, check=True, capture_output=True)
        return sorted(str(p) for p in out_dir.glob("frame_*.png"))

    def frames_to_video(
        self,
        frames_dir: str,
        fps: float,
        output_path: str,
    ) -> str:
        """Assemble PNG frames into a silent video."""
        pattern = str(Path(frames_dir) / "frame_%06d.png")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Get video metadata via ffprobe."""
        import json
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
        )
        audio_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {}
        )
        fmt = data.get("format", {})

        fps = 25.0
        fps_str = video_stream.get("r_frame_rate", "25/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 25.0

        return VideoInfo(
            duration_seconds=float(fmt.get("duration", 0)),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            has_audio=bool(audio_stream),
            codec_video=video_stream.get("codec_name", "unknown"),
            codec_audio=audio_stream.get("codec_name", "unknown"),
        )

    def trim_video(
        self,
        video_path: str,
        output_path: str,
        start_sec: float,
        end_sec: float,
    ) -> str:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-c", "copy",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def add_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
    ) -> str:
        """Burn SRT subtitles into video."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles={srt_path}",
                "-c:a", "copy",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path

    def check_ffmpeg(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
