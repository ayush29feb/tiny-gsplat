from __future__ import annotations

import json
import os
import random

import imageio.v2 as imageio
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class CaptureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        test_every: int = 8,
        val_ids: list[int] | None = None,
    ) -> None:
        meta_path = os.path.join(data_dir, "inputs", "metadata.json")
        with open(meta_path) as f:
            meta: dict = json.load(f)

        cam: dict = meta["camera"]
        all_c2w: Tensor = torch.tensor(cam["camera_to_world"], dtype=torch.float32)
        all_Ks: Tensor = torch.tensor(cam["camera_to_pixel"], dtype=torch.float32)
        image_size: list[float] = cam["image_size_xy"][0]
        self._width: int = int(image_size[0])
        self._height: int = int(image_size[1])

        # Distortion coefficients (OpenCV convention)
        all_radial: Tensor | None = None
        all_tangential: Tensor | None = None
        if "radial_distortion" in cam:
            rd = torch.tensor(cam["radial_distortion"], dtype=torch.float32)
            if rd.shape[-1] == 4:
                rd = torch.nn.functional.pad(rd, (0, 2))  # pad to 6 for pinhole
            all_radial = rd
        if "tangential_distortion" in cam:
            all_tangential = torch.tensor(cam["tangential_distortion"], dtype=torch.float32)

        rgb_indices: list[int] = meta["pixel_data"]["rgb"]
        input_dir: str = os.path.join(data_dir, "inputs")
        all_images: list[Tensor] = []
        for idx in rgb_indices:
            img = imageio.imread(os.path.join(input_dir, f"rgb_{idx}.png"))[..., :3]
            all_images.append(torch.from_numpy(img).float() / 255.0)
        all_images_t: Tensor = torch.stack(all_images)

        centers: Tensor = all_c2w[:, :3, 3]
        mean_center: Tensor = centers.mean(dim=0)
        self._scene_scale: float = (centers - mean_center).norm(dim=-1).max().item()

        n: int = len(all_images_t)
        if val_ids:
            val_set = set(val_ids)
            train_indices = [i for i in range(n) if i not in val_set]
            val_indices = list(val_ids)
        elif test_every > 0:
            val_indices = list(range(0, n, test_every))
            train_indices = [i for i in range(n) if i not in val_indices]
        else:
            train_indices = list(range(n))
            val_indices = []

        self.indices: list[int] = train_indices if split == "train" else val_indices
        self.camtoworlds: Tensor = all_c2w[self.indices]
        self.Ks: Tensor = all_Ks[self.indices]
        self.images: Tensor = all_images_t[self.indices]
        self.radial_coeffs: Tensor | None = all_radial[self.indices] if all_radial is not None else None
        self.tangential_coeffs: Tensor | None = all_tangential[self.indices] if all_tangential is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Tensor | int]:
        return {
            "camtoworld": self.camtoworlds[i],
            "K": self.Ks[i],
            "image": self.images[i],
            "image_id": self.indices[i],
        }

    @property
    def scene_scale(self) -> float:
        return self._scene_scale

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


def load_cameras(path: str) -> tuple[Tensor, Tensor, int, int]:
    with open(path) as f:
        cam: dict = json.load(f)
    camtoworlds: Tensor = torch.tensor(cam["camera_to_world"], dtype=torch.float32)
    Ks: Tensor = torch.tensor(cam["camera_to_pixel"], dtype=torch.float32)
    image_size: list[float] = cam["image_size_xy"][0]
    width, height = int(image_size[0]), int(image_size[1])
    return camtoworlds, Ks, width, height
