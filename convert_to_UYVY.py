import os
import cv2
import numpy as np

def convert_images_to_uyvy(input_folder, output_folder):
    """
    Converts all RGB(D) images in the input folder to UYVY format and saves them as .bin files.
    Also saves the depth channel (if available) as grayscale PNG images.

    Args:
        input_folder (str): Path to the folder containing input RGBD images.
        output_folder (str): Path to save UYVY .bin files and depth maps.
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None or img.shape[2] < 3:
            print(f"Skipping {fname}: not a valid RGB image.")
            continue

        base_name = os.path.splitext(fname)[0]
        rgb = img[:, :, :3]
        h, w = rgb.shape[:2]

        # Ensure even width for UYVY packing
        if w % 2 != 0:
            rgb = rgb[:, :-1, :]
            w -= 1

        # Convert RGB (OpenCV BGR) to YUV
        yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0]
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]

        # Prepare UYVY packed array (U Y0 V Y1 per 2 pixels)
        uyvy = np.zeros((h, w * 2), dtype=np.uint8)

        for row in range(h):
            for col in range(0, w, 2):
                idx = col * 2
                uyvy[row, idx + 0] = u[row, col]        # U
                uyvy[row, idx + 1] = y[row, col]        # Y0
                uyvy[row, idx + 2] = v[row, col]        # V
                uyvy[row, idx + 3] = y[row, col + 1]    # Y1

        # Save UYVY as .bin
        uyvy_path = os.path.join(output_folder, f"{base_name}_UYVY.bin")
        uyvy.tofile(uyvy_path)

        # Save depth if available
        if img.shape[2] == 4:
            d = img[:, :, 3]
            d_path = os.path.join(output_folder, f"{base_name}_D.png")
            cv2.imwrite(d_path, d)
            print(f"Saved {base_name}_UYVY.bin and depth map")
        else:
            print(f"Saved {base_name}_UYVY.bin")

    print("All images converted to UYVY format.")

