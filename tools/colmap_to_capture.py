"""Convert a COLMAP dataset (Mip-NeRF 360 format) to our capture format.

Usage:
  python tools/colmap_to_capture.py --input data/garden --output data/garden_capture --factor 4
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import imageio.v2 as imageio
import numpy as np


def read_cameras_binary(path: str) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
            n = num_params.get(model_id, 4)
            params = struct.unpack(f"<{n}d", f.read(8 * n))
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path: str) -> dict:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images[img_id] = {
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": cam_id,
                "name": name.decode(),
            }
    return images


def qvec_to_rotmat(qvec: tuple) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to COLMAP dataset")
    parser.add_argument("--output", required=True, help="Output capture directory")
    parser.add_argument("--factor", type=int, default=1, help="Downsample factor (1, 2, 4, 8)")
    args = parser.parse_args()

    # Read COLMAP data
    sparse_dir = os.path.join(args.input, "sparse", "0")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))

    # Determine image directory
    if args.factor > 1:
        img_dir = os.path.join(args.input, f"images_{args.factor}")
    else:
        img_dir = os.path.join(args.input, "images")

    # Sort images by name for consistent ordering
    sorted_images = sorted(images.values(), key=lambda x: x["name"])

    # Read actual image size from first image
    first_img_path = os.path.join(img_dir, sorted_images[0]["name"])
    first_img = imageio.imread(first_img_path)
    actual_height, actual_width = first_img.shape[:2]

    # Build camera-to-world matrices and intrinsics
    c2w_list = []
    K_list = []
    image_size_list = []

    for img in sorted_images:
        cam = cameras[img["camera_id"]]

        # COLMAP stores world-to-camera: R, t such that x_cam = R @ x_world + t
        R = qvec_to_rotmat(img["qvec"])
        t = np.array(img["tvec"])

        # Convert to camera-to-world
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        c2w_list.append(c2w.tolist())

        # Build intrinsics matrix, scaled to actual image size
        # PINHOLE: fx, fy, cx, cy
        params = cam["params"]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        sx = actual_width / cam["width"]
        sy = actual_height / cam["height"]
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        K_list.append(K)
        image_size_list.append([float(actual_width), float(actual_height)])

    # Create output directory
    inputs_dir = os.path.join(args.output, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    # Copy and rename images
    rgb_indices = []
    for i, img in enumerate(sorted_images):
        src = os.path.join(img_dir, img["name"])
        dst = os.path.join(inputs_dir, f"rgb_{i}.png")
        print(f"  [{i+1}/{len(sorted_images)}] {img['name']}")
        im = imageio.imread(src)
        imageio.imwrite(dst, im)
        rgb_indices.append(i)

    # Write metadata
    metadata = {
        "camera": {
            "projection_type": "PINHOLE",
            "camera_to_world": c2w_list,
            "camera_to_pixel": K_list,
            "image_size_xy": image_size_list,
        },
        "pixel_data": {
            "rgb": rgb_indices,
        },
    }

    meta_path = os.path.join(inputs_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConverted {len(sorted_images)} images to {args.output}")
    print(f"  Metadata: {meta_path}")
    print(f"  Image size: {int(image_size_list[0][0])}x{int(image_size_list[0][1])}")


if __name__ == "__main__":
    main()
