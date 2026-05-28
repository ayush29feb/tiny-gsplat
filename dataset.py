from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class CameraData:
    camtoworld: Tensor
    K: Tensor
    width: int
    height: int
    radial_coeffs: Tensor | None = None
    tangential_coeffs: Tensor | None = None


@dataclass
class CameraBatch:
    camtoworlds: Tensor
    Ks: Tensor
    width: int
    height: int
    radial_coeffs: Tensor | None = None
    tangential_coeffs: Tensor | None = None

    def __len__(self) -> int:
        return len(self.camtoworlds)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return CameraData(
                camtoworld=self.camtoworlds[idx],
                K=self.Ks[idx],
                width=self.width,
                height=self.height,
                radial_coeffs=self.radial_coeffs[idx] if self.radial_coeffs is not None else None,
                tangential_coeffs=self.tangential_coeffs[idx] if self.tangential_coeffs is not None else None,
            )
        return CameraBatch(
            camtoworlds=self.camtoworlds[idx],
            Ks=self.Ks[idx],
            width=self.width,
            height=self.height,
            radial_coeffs=self.radial_coeffs[idx] if self.radial_coeffs is not None else None,
            tangential_coeffs=self.tangential_coeffs[idx] if self.tangential_coeffs is not None else None,
        )

    def to(self, device: str) -> CameraBatch:
        return CameraBatch(
            camtoworlds=self.camtoworlds.to(device),
            Ks=self.Ks.to(device),
            width=self.width,
            height=self.height,
            radial_coeffs=self.radial_coeffs.to(device) if self.radial_coeffs is not None else None,
            tangential_coeffs=self.tangential_coeffs.to(device) if self.tangential_coeffs is not None else None,
        )


@dataclass
class Sample:
    camera: CameraData
    image: Tensor
    image_id: int


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

        all_cams = parse_camera_dict(meta["camera"])
        all_images = self._load_images(meta, os.path.join(data_dir, "inputs"))
        self._scene_scale = self._compute_scene_scale(all_cams.camtoworlds)

        indices = self._split_indices(len(all_images), split, test_every, val_ids)
        self.indices: list[int] = indices
        self.cameras: CameraBatch = all_cams[indices]
        self.images: Tensor = all_images[indices]

    def _load_images(self, meta: dict, input_dir: str) -> Tensor:
        rgb_indices: list[int] = meta["pixel_data"]["rgb"]
        images: list[Tensor] = []
        for idx in rgb_indices:
            img = imageio.imread(os.path.join(input_dir, f"rgb_{idx}.png"))[..., :3]
            images.append(torch.from_numpy(img).float() / 255.0)
        return torch.stack(images)

    @staticmethod
    def _compute_scene_scale(camtoworlds: Tensor) -> float:
        centers = camtoworlds[:, :3, 3]
        mean_center = centers.mean(dim=0)
        return (centers - mean_center).norm(dim=-1).max().item()

    @staticmethod
    def _split_indices(
        n: int, split: str, test_every: int, val_ids: list[int] | None,
    ) -> list[int]:
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
        return train_indices if split == "train" else val_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Sample:
        return Sample(
            camera=self.cameras[i],
            image=self.images[i],
            image_id=self.indices[i],
        )

    @property
    def scene_scale(self) -> float:
        return self._scene_scale

    @property
    def width(self) -> int:
        return self.cameras.width

    @property
    def height(self) -> int:
        return self.cameras.height


def parse_camera_dict(cam: dict) -> CameraBatch:
    c2w = torch.tensor(cam["camera_to_world"], dtype=torch.float32)
    Ks = torch.tensor(cam["camera_to_pixel"], dtype=torch.float32)
    image_size: list[float] = cam["image_size_xy"][0]
    width, height = int(image_size[0]), int(image_size[1])

    radial: Tensor | None = None
    tangential: Tensor | None = None
    if "radial_distortion" in cam:
        rd = torch.tensor(cam["radial_distortion"], dtype=torch.float32)
        if rd.shape[-1] == 4:
            rd = torch.nn.functional.pad(rd, (0, 2))
        radial = rd
    if "tangential_distortion" in cam:
        tangential = torch.tensor(cam["tangential_distortion"], dtype=torch.float32)

    return CameraBatch(
        camtoworlds=c2w, Ks=Ks, width=width, height=height,
        radial_coeffs=radial, tangential_coeffs=tangential,
    )


def load_cameras(path: str) -> CameraBatch:
    with open(path) as f:
        cam: dict = json.load(f)
    return parse_camera_dict(cam)
