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

def frame_vec(img, size=16):
    """Simple grayscale downsample for autocorrelation."""
    return np.array(
        img.convert("L").resize((size, size), Image.NEAREST),
        dtype=np.float32
    ).flatten()

def period_autocorr(frames, min_len=30, max_len=120):
    """Estimate loop period via cosine-sim autocorrelation."""
    X = np.stack([frame_vec(f) for f in frames], axis=0)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    best_k, best_sim = 0, -1
    # Only search up to half the sequence length to avoid trivial wrap
    for k in range(min_len, min(max_len, len(X)//2) + 1):
        sim = (X[:-k] * X[k:]).sum(axis=1).mean()
        if sim > best_sim:
            best_sim, best_k = sim, k
    return best_k, best_sim

def motion_profile(frames):
    """Per-frame MSE to measure motion energy between consecutive frames."""
    mp = []
    for i in range(len(frames) - 1):
        a = np.asarray(frames[i], dtype=np.float32)
        b = np.asarray(frames[i+1], dtype=np.float32)
        mp.append(((a - b) ** 2).mean())
    return np.array(mp)

def score_segment(hashes, motion, start, length, lam=0.5):
    """Score a candidate loop segment."""
    end = start + length
    seam = frame_distance(hashes[start], hashes[end])
    avg_mot = motion[start:end].mean()
    # Penalize low motion; tweak as needed
    repetition_penalty = 1.0 / (avg_mot + 1e-6)
    total = seam + lam * repetition_penalty
    return total, seam, avg_mot
def detect_best_loop(hashes, frames, min_len=20, max_len=120, lam=0.5):
    """
    Find loop using:
      1) autocorr to guess period
      2) score segments for low seam + high internal motion
    """
    period, _ = period_autocorr(frames, min_len, max_len)
    if period < min_len:
        period = min_len
    motion = motion_profile(frames)
    n = len(frames)
    best = (999999, 0, 0, 0, 0)  # total_score, start, end, seam, avg_mot
    # Ensure we don't overflow end index
    for start in range(0, n - period - 1):
        total, seam, avg_mot = score_segment(hashes, motion, start, period, lam)
        if total < best[0]:
            best = (total, start, start + period, seam, avg_mot)
    return best

def cut_loop(
    input_path: str,
    out: str = "loop.mp4",
    min_gap: int = 12,
    max_gap: int = 120,
    threshold: int = 2,
    limit: int = 0,
    lam: float = 0.5
):
    # Try FFMPEG first; fallback to generic readers for GIFs/APNGs
    fps = 12
    try:
        meta = iio.immeta(input_path, plugin="FFMPEG")
        fps = meta.get("fps", fps)
        frames_np = iio.imread(input_path, plugin="FFMPEG")
    except Exception:
        frames_np = iio.imread(input_path)
    if frames_np.ndim == 3:
        frames_np = frames_np[None, ...]
    if limit > 0:
        frames_np = frames_np[:limit]
    frames = [Image.fromarray(f) for f in frames_np]
    hashes = [phash_vec(f) for f in frames]
    total, start, end, seam, avg_mot = detect_best_loop(
        hashes, frames, min_len=min_gap, max_len=max_gap, lam=lam
    )
    print(f"Loop: {start}->{end} len={end-start} seam={seam}, avg_mot={avg_mot:.3f}, score={total:.3f}")
    if seam <= threshold:
        save_trimmed(frames, start, end, out, fps)
        print(f"Saved {out}")
    else:
        print("Seam too big. Raise threshold or preprocess.")
    return {
        "start": start,
        "end": end,
        "length": end - start,
        "seam": seam,
        "avg_motion": avg_mot,
        "score": total,
        "fps": fps
    }

if __name__ == "__main__":
    import fire
    fire.Fire(cut_loop)