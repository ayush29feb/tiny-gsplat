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

from dataset import Batch, collate_samples, set_random_seed
from lib_bilagrid import total_variation_loss
from model import GaussianSplatModel
from strategy import StrategySchedule


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
    train_set = hydra.utils.instantiate(cfg.data, data_dir=data_dir, split="train")
    val_set = hydra.utils.instantiate(cfg.data, data_dir=data_dir, split="val")
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

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_samples,
    )
    train_iter = iter(train_loader)

    # Training loop
    global_tic = time.time()
    pbar = tqdm.tqdm(range(max_steps))

    for step in pbar:
        try:
            batch: Batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device)

        sh_degree_to_use = min(step // cfg.training.sh_degree_interval, cfg.model.sh_degree)

        strategy = schedule.strategy
        rendered, alphas, info = model(
            batch.cameras,
            image_id=batch.image_ids, sh_degree=sh_degree_to_use,
            absgrad=(strategy.absgrad if isinstance(strategy, DefaultStrategy) else False),
        )

        # Strategy pre-backward
        strategy.step_pre_backward(
            params=model.splats, optimizers=splat_optimizers,
            state=schedule.state, step=step, info=info,
        )

        # Loss
        l1loss = F.l1_loss(rendered, batch.images)
        ssimloss = 1.0 - fused_ssim(
            rendered.permute(0, 3, 1, 2), batch.images.permute(0, 3, 1, 2), padding="valid",
        )
        loss = torch.lerp(l1loss, ssimloss, cfg.training.ssim_lambda)
        if model.bg_module is not None:
            loss = loss + 10.0 * total_variation_loss(model.bg_module.grids)
        if cfg.training.opacity_reg > 0.0:
            loss = loss + cfg.training.opacity_reg * torch.sigmoid(model.splats["opacities"]).mean()
        if cfg.training.scale_reg > 0.0:
            loss = loss + cfg.training.scale_reg * torch.exp(model.splats["scales"]).mean()

        with torch.no_grad():
            mse = F.mse_loss(rendered, batch.images)
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
