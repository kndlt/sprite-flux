import imageio.v3 as iio

def mp4_to_gif(input_mp4: str, output_gif: str, fps: int = None):
    """
    Convert an MP4 video to a looping GIF.

    Args:
        input_mp4:  Path to input .mp4
        output_gif: Path to output .gif
        fps:        Frames per second for the GIF.
                    If None, will use the MP4’s native fps or default to 10.
    """
    # Try to grab the source FPS from FFmpeg metadata
    try:
        meta = iio.immeta(input_mp4, plugin="FFMPEG")
        source_fps = meta.get("fps", 10)
    except Exception:
        source_fps = 10

    fps = fps or source_fps

    # Read all frames from the MP4
    frames = iio.imread(input_mp4, plugin="FFMPEG")
    # Write out as GIF
    iio.imwrite(
        output_gif,
        frames,
        format="GIF",  # explicit GIF format
        fps=fps,
        loop=0         # infinite loop
    )
    print(f"Converted {input_mp4} → {output_gif} at {fps} fps")

if __name__ == "__main__":
    import fire
    fire.Fire(mp4_to_gif)