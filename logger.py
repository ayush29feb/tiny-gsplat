from __future__ import annotations

import json
import os
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CaptureDataset, load_cameras
from model import GaussianSplatModel


class TrainMetricsLogger:
    def __init__(self, flush_every: int = 1000) -> None:
        self.flush_every = flush_every
        self.loss_history: list[tuple[int, float]] = []
        self.psnr_history: list[tuple[int, float]] = []
        self.gs_history: list[tuple[int, int]] = []
        self.gpu_mem_history: list[tuple[int, float]] = []
        self.wall_time_history: list[tuple[int, float]] = []
        self.result_dir: str = ""
        self._start_time: float = 0.0

    def setup(self, result_dir: str, **kw: Any) -> None:
        self.result_dir = result_dir
        self._start_time = time.time()

    def step(self, step: int, model: GaussianSplatModel, *, loss: float, train_psnr: float = 0.0, num_gs: int = 0, **kw: Any) -> None:
        self.loss_history.append((step, loss))
        self.psnr_history.append((step, train_psnr))
        self.gs_history.append((step, num_gs))
        self.gpu_mem_history.append((step, torch.cuda.max_memory_allocated() / 1024**3))
        self.wall_time_history.append((step, time.time() - self._start_time))
        if step % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if not self.loss_history:
            return
        self._save_json()
        self._plot_loss()
        self._plot_psnr()
        self._plot_gs_count()
        self._plot_gpu_mem()

    def _save_json(self) -> None:
        data = {
            "loss": [{"step": s, "value": v} for s, v in self.loss_history],
            "train_psnr": [{"step": s, "value": v} for s, v in self.psnr_history],
            "num_gaussians": [{"step": s, "value": v} for s, v in self.gs_history],
            "gpu_memory_gb": [{"step": s, "value": v} for s, v in self.gpu_mem_history],
            "wall_time_sec": [{"step": s, "value": v} for s, v in self.wall_time_history],
        }
        with open(os.path.join(self.result_dir, "train_metrics.json"), "w") as f:
            json.dump(data, f)

    def _plot_loss(self) -> None:
        steps, losses = zip(*self.loss_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, "r-", alpha=0.3, linewidth=0.5)
        window: int = max(1, len(losses) // 100)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smoothed, "r-", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.result_dir, "train_loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_psnr(self) -> None:
        steps, psnrs = zip(*self.psnr_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, psnrs, "g-", alpha=0.3, linewidth=0.5)
        window: int = max(1, len(psnrs) // 100)
        if window > 1:
            smoothed = np.convolve(psnrs, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smoothed, "g-", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train PSNR (dB)")
        ax.set_title("Training PSNR")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.result_dir, "train_psnr.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_gs_count(self) -> None:
        if not self.gs_history:
            return
        steps, counts = zip(*self.gs_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, [c / 1000 for c in counts], "m-", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Gaussians (k)")
        ax.set_title("Number of Gaussians")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.result_dir, "num_gaussians.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_gpu_mem(self) -> None:
        if not self.gpu_mem_history:
            return
        steps, mem_gb = zip(*self.gpu_mem_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, mem_gb, "c-", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("GPU Memory (GB)")
        ax.set_title("Peak GPU Memory")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.result_dir, "gpu_memory.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def finalize(self) -> None:
        self.flush()


class ValMetricsLogger:
    def __init__(self, every: int = 1000) -> None:
        self.every = every
        self.psnr_history: list[tuple[int, float, float]] = []
        self.result_dir: str = ""
        self.val_dataset: CaptureDataset | None = None
        self.device: str = "cuda"
        self._start_time: float = 0.0

    def setup(
        self, result_dir: str, val_dataset: CaptureDataset | None = None, device: str = "cuda", **kw: Any
    ) -> None:
        self.result_dir = result_dir
        self.val_dataset = val_dataset
        self.device = device
        self._start_time = time.time()

    @torch.no_grad()
    def step(self, step: int, model: GaussianSplatModel, **kw: Any) -> None:
        if step % self.every != 0 or self.val_dataset is None:
            return
        psnr_sum: float = 0.0
        for i in range(len(self.val_dataset)):
            sample: dict[str, Tensor | int] = self.val_dataset[i]
            c2w: Tensor = sample["camtoworld"].unsqueeze(0).to(self.device)
            K: Tensor = sample["K"].unsqueeze(0).to(self.device)
            gt: Tensor = sample["image"].unsqueeze(0).to(self.device)
            radial = self.val_dataset.radial_coeffs[i:i+1].to(self.device) if self.val_dataset.radial_coeffs is not None else None
            tangential = self.val_dataset.tangential_coeffs[i:i+1].to(self.device) if self.val_dataset.tangential_coeffs is not None else None

            rendered, _, _ = model(
                c2w, K, self.val_dataset.width, self.val_dataset.height,
                radial_coeffs=radial, tangential_coeffs=tangential,
            )
            rendered = rendered[..., :3].clamp(0, 1)
            mse: Tensor = F.mse_loss(rendered, gt)
            psnr_sum += -10.0 * torch.log10(mse).item()

        avg_psnr: float = psnr_sum / len(self.val_dataset)
        wall_time: float = time.time() - self._start_time
        self.psnr_history.append((step, avg_psnr, wall_time))
        print(f"\n[step {step}] val_psnr={avg_psnr:.2f} dB ({wall_time:.1f}s)")
        self.flush()

    def flush(self) -> None:
        if not self.psnr_history:
            return
        data = {"val_psnr": [{"step": s, "value": v, "wall_time_sec": t} for s, v, t in self.psnr_history]}
        with open(os.path.join(self.result_dir, "val_metrics.json"), "w") as f:
            json.dump(data, f)
        steps, psnrs, _ = zip(*self.psnr_history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, psnrs, "b-o", markersize=4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Val PSNR (dB)")
        ax.set_title("Validation PSNR")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(self.result_dir, "val_psnr.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def finalize(self) -> None:
        self.flush()


class ValImageLogger:
    def __init__(self, every: int = 1000) -> None:
        self.every = every
        self.render_dir: str = ""
        self.val_dataset: CaptureDataset | None = None
        self.device: str = "cuda"

    def setup(
        self, result_dir: str, val_dataset: CaptureDataset | None = None, device: str = "cuda", **kw: Any
    ) -> None:
        self.render_dir = os.path.join(result_dir, "val_renders")
        os.makedirs(self.render_dir, exist_ok=True)
        self.val_dataset = val_dataset
        self.device = device

    @torch.no_grad()
    def step(self, step: int, model: GaussianSplatModel, **kw: Any) -> None:
        if step % self.every != 0 or self.val_dataset is None:
            return
        for i in range(len(self.val_dataset)):
            sample: dict[str, Tensor | int] = self.val_dataset[i]
            c2w: Tensor = sample["camtoworld"].unsqueeze(0).to(self.device)
            K: Tensor = sample["K"].unsqueeze(0).to(self.device)
            radial = self.val_dataset.radial_coeffs[i:i+1].to(self.device) if self.val_dataset.radial_coeffs is not None else None
            tangential = self.val_dataset.tangential_coeffs[i:i+1].to(self.device) if self.val_dataset.tangential_coeffs is not None else None

            rendered, _, _ = model(
                c2w, K, self.val_dataset.width, self.val_dataset.height,
                radial_coeffs=radial, tangential_coeffs=tangential,
            )
            out: np.ndarray = (rendered[0, ..., :3].cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            path: str = os.path.join(self.render_dir, f"val_{sample['image_id']:04d}_step{step}.png")
            imageio.imwrite(path, out)

    def finalize(self) -> None:
        pass


class TestImageLogger:
    def __init__(self, cameras_path: str, every: int = 1000) -> None:
        self.cameras_path = cameras_path
        self.every = every
        self.render_dir: str = ""
        self.device: str = "cuda"
        self.camtoworlds: Tensor
        self.Ks: Tensor
        self.width: int
        self.height: int

    def setup(self, result_dir: str, device: str = "cuda", **kw: Any) -> None:
        self.render_dir = os.path.join(result_dir, "renders")
        os.makedirs(self.render_dir, exist_ok=True)
        self.device = device
        cameras_path: str = self.cameras_path
        if not os.path.isabs(cameras_path):
            import hydra
            cameras_path = hydra.utils.to_absolute_path(cameras_path)
        self.camtoworlds, self.Ks, self.width, self.height = load_cameras(cameras_path)

    @torch.no_grad()
    def step(self, step: int, model: GaussianSplatModel, **kw: Any) -> None:
        if step % self.every != 0:
            return
        for i in range(len(self.camtoworlds)):
            c2w: Tensor = self.camtoworlds[i : i + 1].to(self.device)
            K: Tensor = self.Ks[i : i + 1].to(self.device)

            rendered, _, _ = model(c2w, K, self.width, self.height)
            out: np.ndarray = (rendered[0, ..., :3].cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            path: str = os.path.join(self.render_dir, f"render_{i:04d}_step{step}.png")
            imageio.imwrite(path, out)

    def finalize(self) -> None:
        pass


class PlyLogger:
    def __init__(self, every: int = 30000) -> None:
        self.every = every
        self.result_dir: str = ""

    def setup(self, result_dir: str, **kw: Any) -> None:
        self.ply_dir = os.path.join(result_dir, "ply")
        os.makedirs(self.ply_dir, exist_ok=True)

    @torch.no_grad()
    def step(self, step: int, model: GaussianSplatModel, **kw: Any) -> None:
        if step % self.every != 0:
            return
        from gsplat import export_splats
        from dataset import rgb_to_sh

        splats = model.splats
        means = splats["means"]
        scales = splats["scales"]
        quats = splats["quats"]
        opacities = splats["opacities"]

        if "sh0" in splats:
            sh0 = splats["sh0"]
            shN = splats["shN"]
        else:
            if model.app_module is not None:
                rgb = model.app_module(
                    features=splats["features"],
                    embed_ids=None,
                    dirs=torch.zeros_like(means[None, :, :]),
                    sh_degree=model.model_cfg.sh_degree,
                )
                rgb = torch.sigmoid(rgb + splats["colors"]).squeeze(0)
            else:
                rgb = torch.sigmoid(splats["colors"])
            sh0 = rgb_to_sh(rgb).unsqueeze(1)
            shN = torch.empty(len(means), 0, 3, device=means.device)

        path: str = os.path.join(self.ply_dir, f"splats_{step}.ply")
        export_splats(
            means=means, scales=scales, quats=quats, opacities=opacities,
            sh0=sh0, shN=shN, format="ply", save_to=path,
        )
        print(f"PLY saved to {path}")

    def finalize(self) -> None:
        pass


class CheckpointLogger:
    def __init__(self, every: int = 30000) -> None:
        self.every = every
        self.result_dir: str = ""

    def setup(self, result_dir: str, **kw: Any) -> None:
        self.result_dir = result_dir

    def step(self, step: int, model: GaussianSplatModel, **kw: Any) -> None:
        if step % self.every != 0:
            return
        path: str = os.path.join(self.result_dir, f"ckpt_{step}.pt")
        model.save_ckpt(path, step)
        print(f"Checkpoint saved to {path}")

    def finalize(self) -> None:
        pass
