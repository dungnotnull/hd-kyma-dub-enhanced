"""Lip-sync generation using Wav2Lip with ffmpeg video assembly."""
from __future__ import annotations

import os
import sys
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WAV2LIP_FACE_DET_CONFIDENCE = 0.85


@dataclass
class LipSyncResult:
    output_video_path: str
    sync_confidence: float
    frames_processed: int
    model_used: str
    skipped: bool = False
    skip_reason: str = ""


class LipSyncEngine:
    """Wav2Lip-based lip-sync generation with graceful degradation."""

    def __init__(self, wav2lip_checkpoint: str = "Wav2Lip/checkpoints/wav2lip_gan.pth"):
        self.checkpoint_path = wav2lip_checkpoint
        self._wav2lip_available = None

    def _check_wav2lip(self) -> bool:
        if self._wav2lip_available is not None:
            return self._wav2lip_available
        available = (
            Path(self.checkpoint_path).exists()
            or _find_wav2lip_checkpoint() is not None
        )
        self._wav2lip_available = available
        if not available:
            logger.warning(
                "Wav2Lip checkpoint not found at %s; lip-sync will be skipped",
                self.checkpoint_path,
            )
        return available

    def generate(
        self,
        video_path: str,
        dubbed_audio_path: str,
        output_path: str,
        face_crop: bool = False,
        resize_factor: int = 1,
    ) -> LipSyncResult:
        """Generate lip-synced video by combining dubbed audio with source video."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Source video not found: {video_path}")
        if not Path(dubbed_audio_path).exists():
            raise FileNotFoundError(f"Dubbed audio not found: {dubbed_audio_path}")

        if not self._check_wav2lip():
            return self._skip_lipsync(video_path, dubbed_audio_path, output_path, "Wav2Lip checkpoint not found")

        face_detected, frame_count = self._detect_faces(video_path)
        if not face_detected:
            return self._skip_lipsync(
                video_path, dubbed_audio_path, output_path,
                "No face detected in source video"
            )

        try:
            confidence = self._run_wav2lip(
                video_path=video_path,
                audio_path=dubbed_audio_path,
                output_path=output_path,
                face_crop=face_crop,
                resize_factor=resize_factor,
            )

            if confidence < WAV2LIP_FACE_DET_CONFIDENCE:
                logger.warning(
                    "Wav2Lip sync_confidence %.2f < threshold %.2f; skipping lip-sync",
                    confidence, WAV2LIP_FACE_DET_CONFIDENCE,
                )
                return self._skip_lipsync(
                    video_path, dubbed_audio_path, output_path,
                    f"Low sync confidence: {confidence:.2f}",
                )

            return LipSyncResult(
                output_video_path=output_path,
                sync_confidence=confidence,
                frames_processed=frame_count,
                model_used="Wav2Lip",
            )

        except Exception as e:
            logger.error("Wav2Lip inference failed (%s); falling back to audio-only dubbing", e)
            return self._skip_lipsync(video_path, dubbed_audio_path, output_path, str(e))

    def _run_wav2lip(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        face_crop: bool,
        resize_factor: int,
    ) -> float:
        checkpoint = _find_wav2lip_checkpoint() or self.checkpoint_path
        wav2lip_dir = str(Path(checkpoint).parent.parent)

        cmd = [
            sys.executable,
            os.path.join(wav2lip_dir, "inference.py"),
            "--checkpoint_path", checkpoint,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", str(resize_factor),
        ]
        if face_crop:
            cmd += ["--crop", "0 -1 0 -1"]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=wav2lip_dir)
        if result.returncode != 0:
            raise RuntimeError(f"Wav2Lip failed: {result.stderr[-500:]}")

        confidence = _parse_wav2lip_confidence(result.stdout + result.stderr)
        return confidence

    def _detect_faces(self, video_path: str) -> tuple[bool, int]:
        """Detect if video contains a human face using OpenCV."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = min(10, total_frames)
            face_detected = False

            for i in range(sample_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total_frames / sample_frames))
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    face_detected = True
                    break
            cap.release()
            return face_detected, total_frames
        except Exception as e:
            logger.warning("Face detection failed (%s); assuming face present", e)
            return True, 0

    def _skip_lipsync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        reason: str,
    ) -> LipSyncResult:
        """Merge dubbed audio with original video frames (no lip-sync modification)."""
        logger.info("Skipping lip-sync: %s. Merging audio with original video.", reason)
        try:
            self._ffmpeg_merge_audio(video_path, audio_path, output_path)
        except Exception as e:
            logger.error("ffmpeg merge also failed (%s); copying original video", e)
            import shutil
            shutil.copy2(video_path, output_path)

        return LipSyncResult(
            output_video_path=output_path,
            sync_confidence=0.0,
            frames_processed=0,
            model_used="none",
            skipped=True,
            skip_reason=reason,
        )

    def _ffmpeg_merge_audio(self, video_path: str, audio_path: str, output_path: str):
        """Replace video audio track with dubbed audio using ffmpeg."""
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

    def crop_face_region(self, video_path: str, output_path: str) -> Optional[str]:
        """Crop video to face region for better Wav2Lip accuracy."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            pad = int(h * 0.3)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = x + w + pad
            y2 = y + h + pad

            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1}",
                    "-c:a", "copy",
                    output_path,
                ],
                check=True, capture_output=True,
            )
            return output_path
        except Exception as e:
            logger.warning("Face crop failed (%s)", e)
            return None


def _find_wav2lip_checkpoint() -> Optional[str]:
    candidates = [
        "Wav2Lip/checkpoints/wav2lip_gan.pth",
        "wav2lip/checkpoints/wav2lip_gan.pth",
        "upstream/Wav2Lip/checkpoints/wav2lip_gan.pth",
        os.path.expanduser("~/Wav2Lip/checkpoints/wav2lip_gan.pth"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _parse_wav2lip_confidence(output: str) -> float:
    """Parse Wav2Lip inference output for average face detection confidence."""
    import re
    matches = re.findall(r"confidence[:\s]+([\d.]+)", output, re.IGNORECASE)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return WAV2LIP_FACE_DET_CONFIDENCE
