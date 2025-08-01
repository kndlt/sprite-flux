from PIL import Image
import numpy as np

black_bg_path = "inputs/bg_removal/sprite_black.png"
white_bg_path = "inputs/bg_removal/sprite_white.png"
output_path = "outputs/sprite_with_alpha.png"
alpha_debug_path = "outputs/alpha_debug.png"

# Load and normalize
img_black = Image.open(black_bg_path).convert("RGB")
img_white = Image.open(white_bg_path).convert("RGB")

if img_black.size != img_white.size:
    raise ValueError("Images must have the same dimensions")

arr_black = np.array(img_black).astype(np.float32) / 255.0
arr_white = np.array(img_white).astype(np.float32) / 255.0

# Compute absolute difference per pixel
diff = np.abs(arr_white - arr_black)
diff_magnitude = np.mean(diff, axis=-1)

# === Soft alpha mask with inversion ===
# Transparent if diff is low (background), opaque if high (sprite)
alpha = np.clip((1.0 - diff_magnitude) ** 2 * 255, 0, 255).astype(np.uint8)

# Clamp RGB and combine
rgb = (arr_black * 255).astype(np.uint8)
rgba = np.dstack((rgb, alpha))

# Save output image with alpha
Image.fromarray(rgba, mode="RGBA").save(output_path)
print(f"✅ Saved output to {output_path}")

# Optional: save alpha debug image
Image.fromarray(alpha).save(alpha_debug_path)
print(f"🩻 Saved alpha debug to {alpha_debug_path}")
