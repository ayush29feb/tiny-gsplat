"""Train Gaussian splats on a capture dataset.

Outputs go to outputs/<date>/<time>/ (managed by Hydra).

Usage:
  python train.py
  python train.py training.max_steps=30000
  python train.py train_data.val_ids=[20] val_data.val_ids=[20]
  python train.py model.app_opt=true model.antialiased=true
  python train.py 'strategies=[{_target_: gsplat.strategy.DefaultStrategy, start_step: 0, stop_step: 2000}, {_target_: gsplat.strategy.MCMCStrategy, start_step: 2000, stop_step: 30000}]'
"""

from __future__ import annotations

import os
import time

import hydra
import torch
import torch.nn.functional as F
import tqdm
from fused_ssim import fused_ssim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gsplat.strategy import DefaultStrategy, MCMCStrategy

from dataset import CameraData, set_random_seed
from lib_bilagrid import total_variation_loss
from model import GaussianSplatModel


class StrategySchedule:
    """Manages a sequence of gsplat strategies, each active for a step range."""

    def __init__(self, strategy_cfgs: list, splats, optimizers, scene_scale: float):
        self.entries: list[dict] = []
        for cfg in strategy_cfgs:
            cfg = dict(cfg)
            start = cfg.pop("start_step")
            stop = cfg.pop("stop_step")
            strategy = hydra.utils.instantiate(cfg)
            strategy.check_sanity(splats, optimizers)
            if isinstance(strategy, DefaultStrategy):
                state = strategy.initialize_state(scene_scale=scene_scale)
            elif isinstance(strategy, MCMCStrategy):
                state = strategy.initialize_state()
            self.entries.append({"strategy": strategy, "state": state, "start": start, "stop": stop})
        self._active_idx: int = 0

    @property
    def active(self):
        return self.entries[self._active_idx]

    @property
    def strategy(self):
        return self.active["strategy"]

    @property
    def state(self):
        return self.active["state"]

    def update(self, step: int):
        for i, e in enumerate(self.entries):
            if e["start"] <= step < e["stop"]:
                if i != self._active_idx:
                    self._active_idx = i
                    print(f"\n[step {step}] Switched to {type(e['strategy']).__name__}")
                return


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    device = cfg.device

    # Hydra manages output dir at outputs/<date>/<time>/
    result_dir: str = HydraConfig.get().runtime.output_dir

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Data
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    train_set = hydra.utils.instantiate(cfg.train_data, data_dir=data_dir)
    val_set = hydra.utils.instantiate(cfg.val_data, data_dir=data_dir)
    if len(val_set) == 0:
        val_set = None
    scene_scale = train_set.scene_scale * 1.1
    print(
        f"Loaded train={len(train_set)}, val={len(val_set) if val_set else 0}, "
        f"({train_set.width}x{train_set.height}), scene_scale={scene_scale:.3f}"
    )

    # Renderer + Model
    renderer = hydra.utils.instantiate(cfg.renderer)
    model = GaussianSplatModel(
        cfg.model,
        len(train_set), scene_scale,
        train_set.width, train_set.height, device,
        renderer=renderer,
    )
    print(f"Initialized {cfg.model.init_num_pts} Gaussians")

    # Optimizers + Schedulers
    splat_optimizers, aux_optimizer, schedulers = model.get_optimizers(cfg.lr, cfg.training.max_steps)

    # Strategy
    schedule = StrategySchedule(cfg.strategies, model.splats, splat_optimizers, scene_scale)

    max_steps = cfg.training.max_steps

    # Loggers — instantiate from config, then setup with runtime deps
    loggers = []
    for name, lg_cfg in cfg.loggers.items():
        lg = hydra.utils.instantiate(lg_cfg)
        lg.setup(result_dir=result_dir, val_dataset=val_set, device=device)
        loggers.append(lg)

    # Move training data to GPU
    train_cams = train_set.cameras.to(device)
    train_images = train_set.images.to(device)

    # Training loop
    global_tic = time.time()
    pbar = tqdm.tqdm(range(max_steps))

    batch_size: int = cfg.training.batch_size
    for step in pbar:
        batch_idx = torch.randint(0, len(train_set), (batch_size,))
        pixels = train_images[batch_idx]
        image_id = batch_idx.to(device)
        batch_cam = CameraData(
            camtoworld=train_cams.camtoworlds[batch_idx],
            K=train_cams.Ks[batch_idx],
            width=train_cams.width,
            height=train_cams.height,
            radial_coeffs=train_cams.radial_coeffs[batch_idx] if train_cams.radial_coeffs is not None else None,
            tangential_coeffs=train_cams.tangential_coeffs[batch_idx] if train_cams.tangential_coeffs is not None else None,
        )

        sh_degree_to_use = min(step // cfg.model.sh_degree_interval, cfg.model.sh_degree)

        strategy = schedule.strategy
        rendered, alphas, info = model(
            batch_cam,
            image_id=image_id, sh_degree=sh_degree_to_use,
            absgrad=(strategy.absgrad if isinstance(strategy, DefaultStrategy) else False),
        )

        # Strategy pre-backward
        strategy.step_pre_backward(
            params=model.splats, optimizers=splat_optimizers,
            state=schedule.state, step=step, info=info,
        )

        # Loss
        l1loss = F.l1_loss(rendered, pixels)
        ssimloss = 1.0 - fused_ssim(
            rendered.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid",
        )
        loss = torch.lerp(l1loss, ssimloss, cfg.training.ssim_lambda)
        if model.bg_module is not None:
            loss = loss + 10.0 * total_variation_loss(model.bg_module.grids)
        if cfg.training.opacity_reg > 0.0:
            loss = loss + cfg.training.opacity_reg * torch.sigmoid(model.splats["opacities"]).mean()
        if cfg.training.scale_reg > 0.0:
            loss = loss + cfg.training.scale_reg * torch.exp(model.splats["scales"]).mean()

        with torch.no_grad():
            mse = F.mse_loss(rendered, pixels)
            train_psnr: float = -10.0 * torch.log10(mse).item()

        loss.backward()

        pbar.set_description(
            f"loss={loss.item():.4f} | psnr={train_psnr:.1f} | sh={sh_degree_to_use} | gs={len(model.splats['means'])}"
        )

        # Optimizer step
        for opt in splat_optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if aux_optimizer is not None:
            aux_optimizer.step()
            aux_optimizer.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()

        # Strategy post-backward + schedule update
        if isinstance(strategy, DefaultStrategy):
            strategy.step_post_backward(
                params=model.splats, optimizers=splat_optimizers,
                state=schedule.state, step=step, info=info, packed=False,
            )
        elif isinstance(strategy, MCMCStrategy):
            strategy.step_post_backward(
                params=model.splats, optimizers=splat_optimizers,
                state=schedule.state, step=step, info=info,
                lr=schedulers[0].get_last_lr()[0],
            )
        schedule.update(step)

        # Loggers
        for lg in loggers:
            lg.step(step + 1, model, loss=loss.item(), train_psnr=train_psnr, num_gs=len(model.splats["means"]))

    elapsed = time.time() - global_tic
    print(f"Training done in {elapsed:.1f}s, {len(model.splats['means'])} Gaussians")

    for lg in loggers:
        lg.finalize()

    print("Done.")


if __name__ == "__main__":
    main()
