import imageio.v3 as iio
import numpy as np
from PIL import Image
import imagehash

def phash_vec(img):
    """Return perceptual hash as uint8 vector."""
    h = imagehash.phash(img)
    return h.hash.astype(np.uint8).flatten()
def frame_distance(a, b):
    """Hamming distance between hashes (0 = identical)."""
    return np.count_nonzero(a ^ b)
def detect_best_pair(hashes, min_gap=4, max_gap=60):
    n = len(hashes)
    best = (999999, 0, 0)
    for i in range(n - min_gap):
        j_end = min(i + max_gap, n - 1)
        for j in range(i + min_gap, j_end + 1):
            d = frame_distance(hashes[i], hashes[j])
            if d < best[0]:
                best = (d, i, j)
    return best
def save_trimmed(frames, start, end, out_path, fps):
    sliced = frames[start:end+1]
    if out_path.lower().endswith(".gif"):
        sliced[0].save(
            out_path,
            save_all=True,
            append_images=sliced[1:],
            loop=0,
            duration=int(1000/fps),
            disposal=2
        )
    else:
        arr = [np.array(f) for f in sliced]
        iio.imwrite(out_path, arr, fps=fps, codec="libx264", quality=8)

def cut_loop(
    input_path: str,
    out: str = "loop.mp4",    # <- add out param
    min_gap: int = 4,
    max_gap: int = 60,
    threshold: int = 2,
    limit: int = 0
):
    meta = iio.immeta(input_path, plugin="FFMPEG")
    fps = meta.get("fps", 12)
    frames_np = iio.imread(input_path, plugin="FFMPEG")
    if frames_np.ndim == 3:
        frames_np = frames_np[None, ...]
    if limit > 0:
        frames_np = frames_np[:limit]
    frames = [Image.fromarray(f) for f in frames_np]
    hashes = [phash_vec(f) for f in frames]
    dist, start, end = detect_best_pair(hashes, min_gap, max_gap)
    print(f"Best pair: {start} -> {end}, hash distance={dist}")
    if dist <= threshold:
        print("Loop accepted. Saving...")
        save_trimmed(frames, start, end, out, fps)
        print(f"Saved {out}")
    else:
        print("No clean loop under threshold. Raise --threshold or preprocess frames.")

if __name__ == "__main__":
    import fire
    fire.Fire(cut_loop)