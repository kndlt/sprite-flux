#!/usr/bin/env python3
"""
split_shots.py ──────────────────────────────────────────────────────────────
Split an input clip into one file per shot using PySceneDetect + FFmpeg.

Features
--------
• Handles **MP4** *or* animated **WebP** as input.
• Exports each shot as **MP4** (stream‑copy or loss‑less re‑encode) or
  **loss‑less animated WebP**.
• CLI knobs for detector type, threshold, minimum scene length,
  fixed‑duration slicing, and exact‑frame re‑encode.

Dependencies
------------
    pip install scenedetect[opencv]
    # plus a working ffmpeg on your PATH.
    # For animated‑WebP input you also need either:
    #   1) ffmpeg built with --enable-libwebp, OR
    #   2) libwebp tools (webpmux) for the fallback path.

Examples
--------
    # Loss‑less MP4 stream‑copy slices (fast, but key‑frame aligned)
    python split_shots.py clip.mp4 --out shots_mp4

    # Frame‑perfect cuts (re‑encode) & minimum scene length 40 frames
    python split_shots.py clip.mp4 --reencode --min-scene-len 40

    # Animated WebP → three WebP shots
    python split_shots.py anim.webp --format webp --detector adaptive \
                         --threshold 30
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

# ── PySceneDetect -----------------------------------------------------------
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.frame_timecode import FrameTimecode

# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def detect_scenes(video: Path, *, detector_cls, threshold: float, min_len: int) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Return list of (start, end) FrameTimecodes for each detected scene."""
    video_manager = VideoManager([str(video)])
    scene_manager = SceneManager()

    detector = detector_cls(threshold=threshold, min_scene_len=min_len)
    scene_manager.add_detector(detector)

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()

# ---------------------------------------------------------------------------
# WebP helper – tries ffmpeg first, falls back to webpmux → PNG seq → ffmpeg
# ---------------------------------------------------------------------------

def webp_to_tmp_mp4(src: Path) -> Path:
    """Convert animated WebP → temporary MP4 that PySceneDetect can read."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="splitshots_"))
    dst_mp4 = tmp_dir / f"{src.stem}.mp4"

    def _try_ffmpeg() -> bool:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
            "-movflags", "+faststart", str(dst_mp4),
        ]
        return subprocess.call(cmd) == 0

    if _try_ffmpeg():
        return dst_mp4  # success via ffmpeg

    # ── ffmpeg failed – use webpmux fallback --------------------------------
    if not shutil.which("webpmux"):
        raise RuntimeError(
            "FFmpeg cannot decode animated WebP and 'webpmux' is not installed."
        )

    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    # 1) Dump each frame as PNG
    subprocess.check_call(["webpmux", "-dump", str(src), "-o", str(frames_dir / "frame.png")])

    # 2) Encode PNG sequence → MP4 (30 fps default)
    subprocess.check_call([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", "30",
        "-pattern_type", "glob",
        "-i", str(frames_dir / "frame*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
        "-movflags", "+faststart", str(dst_mp4),
    ])

    # Clean up PNGs to save space
    for png in glob.glob(str(frames_dir / "frame*.png")):
        os.remove(png)

    return dst_mp4

# ---------------------------------------------------------------------------
# FFmpeg split helper
# ---------------------------------------------------------------------------

def encode_cmd(
    fmt: str,
    src: Path,
    start_ts: str,
    end_ts: str,
    dst: Path,
    *,
    reencode: bool,
):
    if fmt == "mp4":
        if reencode:
            # accurate frame-level cut: seek *after* input  use duration (-t)
            return [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-ss", start_ts,
                "-t",  end_ts,           # here end_ts == duration
                "-c:v", "libx264", "-crf", "0", "-preset", "veryslow",
                "-pix_fmt", "yuv420p",
                str(dst),
            ]
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", start_ts, "-to", end_ts, "-i", str(src),
            "-c", "copy", str(dst),
        ]

    if fmt == "webp":  # always re‑encode
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", start_ts, "-to", end_ts, "-i", str(src),
            "-an",
            "-vcodec", "libwebp", "-lossless", "1", "-preset", "default",
            "-loop", "0", "-vsync", "0", str(dst),
        ]

    raise ValueError(f"Unsupported format: {fmt}")


def split_scenes(
    src: Path,
    scenes: List[Tuple[FrameTimecode, FrameTimecode]],
    out_dir: Path,
    *,
    fmt: str,
    reencode: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Detected {len(scenes)} shots.")
    for idx, (start, end) in enumerate(scenes):
        # ── 1) make end exclusive when re-encoding ────────────────────────
        adj_end = end - 1 if reencode else end          # drop 1 frame
        duration = (adj_end - start)                    # FrameTimecode → duration

        dst = out_dir / f"shot-{idx:03}.{fmt}"
        cmd = encode_cmd(
            fmt,
            src,
            start.get_timecode(),
            duration.get_timecode() if reencode else adj_end.get_timecode(),
            dst,
            reencode=reencode,
        )
        subprocess.check_call(cmd)
        print(f"    ↳ {dst.name}  ({start.get_timecode()} → {adj_end.get_timecode()})")

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Split a clip into per‑shot files.")

    p.add_argument("video", type=Path, help="Source clip (.mp4 or animated .webp)")
    p.add_argument("--out", type=Path, default=Path("./shots"), help="Output directory")

    # Detection tuning
    p.add_argument("--detector", choices=["content", "adaptive"], default="content",
                   help="Scene detector type (default: content)")
    p.add_argument("--threshold", type=float, default=8.0,
                   help="Detector sensitivity (lower → more cuts)")
    p.add_argument("--min-scene-len", type=int, default=15,
                   help="Ignore cuts that happen before N frames have elapsed")
    p.add_argument("--seconds-per-shot", type=float, default=None,
                   help="Skip detection; hard‑cut every N seconds instead")

    # Output
    p.add_argument("--format", choices=["mp4", "webp"], default="mp4", help="Output container")
    p.add_argument("--reencode", action="store_true", default=True,
                   help="Re‑encode each shot for frame‑perfect cuts (MP4 only)")

    args = p.parse_args()

    work_file = args.video
    cleanup = False

    # ── Handle animated WebP input -----------------------------------------
    if work_file.suffix.lower() == ".webp":
        print("[i] Input is WebP – converting to temporary MP4 for detection…")
        work_file = webp_to_tmp_mp4(work_file)
        cleanup = True

    # ── Build scene list ----------------------------------------------------
    if args.seconds_per_shot:
        # Fake scenes every N seconds
        import cv2
        cap = cv2.VideoCapture(str(work_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_chunk = int(args.seconds_per_shot * fps)
        scenes = []
        for f0 in range(0, total_frames, frames_per_chunk):
            start = FrameTimecode(f0, fps)
            end = FrameTimecode(min(f0 + frames_per_chunk, total_frames), fps)
            scenes.append((start, end))
    else:
        Detector = ContentDetector if args.detector == "content" else AdaptiveDetector
        scenes = detect_scenes(
            work_file,
            detector_cls=Detector,
            threshold=args.threshold,
            min_len=args.min_scene_len,
        )

    if not scenes:
        print("[!] No cuts detected – nothing to split.")
        return

    split_scenes(
        work_file,
        scenes,
        args.out,
        fmt=args.format,
        reencode=args.reencode,
    )

    if cleanup:
        try:
            work_file.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
