from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from gsplat import export_splats
from gsplat.cuda._torch_impl import _eval_sh_bases_fast
from gsplat.rendering import rasterization
from lib_bilagrid import BilateralGrid, slice as bg_slice, total_variation_loss

from dataset import CameraData, knn, rgb_to_sh


@dataclass
class Splats:
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor
    sh_degree: int

    def export_ply(self, path: str, sh0: Tensor, shN: Tensor) -> None:
        """Write Gaussian splats to a .ply file. Expects raw (log-space scales, logit opacities)."""
        export_splats(
            means=self.means, scales=self.scales, quats=self.quats,
            opacities=self.opacities, sh0=sh0, shN=shN,
            format="ply", save_to=path,
        )


class SplatRenderer:
    """Wraps gsplat rasterization with fixed config (near/far plane, rasterize mode, distortion)."""

    def __init__(
        self,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        antialiased: bool = False,
        use_distortion: bool = False,
    ) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.rasterize_mode: str = "antialiased" if antialiased else "classic"
        self.use_distortion = use_distortion

    def __call__(
        self,
        splats: Splats,
        camera: CameraData,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        # Compute view matrices from camera-to-world
        camtoworld = camera.camtoworld
        if camtoworld.dim() == 2:
            camtoworld = camtoworld.unsqueeze(0)
        viewmats = torch.linalg.inv(camtoworld)

        # Resolve distortion coefficients
        radial = camera.radial_coeffs
        tangential = camera.tangential_coeffs
        with_ut = self.use_distortion and (radial is not None or tangential is not None)
        if not with_ut:
            radial = None
            tangential = None

        return rasterization(
            means=splats.means,
            quats=splats.quats,
            scales=splats.scales,
            opacities=splats.opacities,
            colors=splats.colors,
            viewmats=viewmats,
            Ks=camera.K if camera.K.dim() == 3 else camera.K.unsqueeze(0),
            width=camera.width,
            height=camera.height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            rasterize_mode=self.rasterize_mode,
            sh_degree=splats.sh_degree,
            packed=False,
            with_ut=with_ut,
            radial_coeffs=radial,
            tangential_coeffs=tangential,
            **kwargs,
        )


class AppearanceOptModule(torch.nn.Module):
    """Per-image appearance MLP that predicts view-dependent color corrections."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree

        # Per-image learned embedding
        self.embeds = torch.nn.Embedding(n, embed_dim)

        # Color prediction MLP
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width),
            torch.nn.ReLU(inplace=True),
        ]
        for _ in range(mlp_depth - 1):
            layers += [torch.nn.Linear(mlp_width, mlp_width), torch.nn.ReLU(inplace=True)]
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor | None, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        C, N = dirs.shape[:2]

        # Look up per-image embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)
        embeds = embeds[:, None, :].expand(-1, N, -1)
        features = features[None, :, :].expand(C, -1, -1)

        # Evaluate spherical harmonics basis
        dirs = F.normalize(dirs, dim=-1)
        num_bases_to_use: int = (sh_degree + 1) ** 2
        num_bases: int = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)

        # Concatenate and predict colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        return self.color_head(h)


class GaussianSplatModel(torch.nn.Module):
    """3D Gaussian splatting model with optional appearance optimization and bilateral grid."""

    def __init__(
        self,
        model_cfg: DictConfig,
        lr_cfg: DictConfig,
        training_cfg: DictConfig,
        n_train_images: int,
        scene_scale: float,
        width: int,
        height: int,
        device: str,
        renderer: SplatRenderer | None = None,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.lr_cfg = lr_cfg
        self.training_cfg = training_cfg
        self.device = device
        self.scene_scale = scene_scale

        feature_dim: int | None = 32 if model_cfg.app_opt else None
        N: int = model_cfg.init_num_pts

        # Random point cloud initialization
        points: Tensor = model_cfg.init_extent * scene_scale * (torch.rand((N, 3)) * 2 - 1)
        rgbs: Tensor = torch.rand((N, 3))

        # Scale from k-nearest-neighbor distances
        dist2_avg: Tensor = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg: Tensor = torch.sqrt(dist2_avg)
        scales: Tensor = torch.log(dist_avg * model_cfg.init_scale).unsqueeze(-1).repeat(1, 3)

        quats: Tensor = torch.rand((N, 4))
        opacities: Tensor = torch.logit(torch.full((N,), model_cfg.init_opa))

        # Register splat parameters with their learning rates
        self._splat_params: list[tuple[str, torch.nn.Parameter, float]] = [
            ("means", torch.nn.Parameter(points), lr_cfg.means * scene_scale),
            ("scales", torch.nn.Parameter(scales), lr_cfg.scales),
            ("quats", torch.nn.Parameter(quats), lr_cfg.quats),
            ("opacities", torch.nn.Parameter(opacities), lr_cfg.opacities),
        ]

        # Color representation: SH coefficients or learned features
        if feature_dim is None:
            colors = torch.zeros((N, (model_cfg.sh_degree + 1) ** 2, 3))
            colors[:, 0, :] = rgb_to_sh(rgbs)
            self._splat_params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), lr_cfg.sh0))
            self._splat_params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), lr_cfg.shN))
        else:
            features = torch.rand(N, feature_dim)
            self._splat_params.append(("features", torch.nn.Parameter(features), lr_cfg.sh0))
            colors_param = torch.logit(rgbs)
            self._splat_params.append(("colors", torch.nn.Parameter(colors_param), lr_cfg.sh0))

        self.splats = torch.nn.ParameterDict(
            {n: v for n, v, _ in self._splat_params}
        ).to(device)

        # Optional per-image appearance optimization
        self.app_module: AppearanceOptModule | None = None
        if model_cfg.app_opt:
            self.app_module = AppearanceOptModule(
                n_train_images, feature_dim, model_cfg.app_embed_dim, model_cfg.sh_degree
            ).to(device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)

        # Optional bilateral grid for per-image color correction
        self.bg_module: torch.nn.Module | None = None
        self._grid_xy: Tensor | None = None
        if model_cfg.bilateral_grid:
            self.bg_module = BilateralGrid(n_train_images).to(device)
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float32) + 0.5,
                torch.arange(width, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            grid_xy = torch.stack([pixel_x, pixel_y], dim=-1)
            grid_xy = grid_xy / torch.tensor([width, height], device=device, dtype=torch.float32)
            self._grid_xy = grid_xy.unsqueeze(0)

        self.renderer = renderer or SplatRenderer()

    def get_optimizers(
        self,
    ) -> tuple[dict[str, torch.optim.Adam], list[torch.optim.Adam], list[torch.optim.Adam]]:
        # Splat parameter optimizers
        splat_optimizers: dict[str, torch.optim.Adam] = {}
        for name, _, lr in self._splat_params:
            splat_optimizers[name] = torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr, "name": name}],
                eps=1e-15, betas=(0.9, 0.999), fused=True,
            )

        # Appearance module optimizers
        app_optimizers: list[torch.optim.Adam] = []
        if self.app_module is not None:
            app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=self.lr_cfg.app_opt * 10.0, weight_decay=1e-6,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=self.lr_cfg.app_opt,
                ),
            ]

        # Bilateral grid optimizers
        bg_optimizers: list[torch.optim.Adam] = []
        if self.bg_module is not None:
            bg_optimizers = [
                torch.optim.Adam(self.bg_module.parameters(), lr=2e-3, eps=1e-15)
            ]

        return splat_optimizers, app_optimizers, bg_optimizers

    def forward(
        self,
        camera: CameraData,
        image_id: Tensor | None = None,
        sh_degree: int | None = None,
        **raster_kwargs: Any,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        if sh_degree is None:
            sh_degree = self.model_cfg.sh_degree

        # Extract and activate splat parameters
        means: Tensor = self.splats["means"]
        quats: Tensor = self.splats["quats"]
        scales: Tensor = torch.exp(self.splats["scales"])
        opas: Tensor = torch.sigmoid(self.splats["opacities"])

        # Ensure batched camera pose
        camtoworld = camera.camtoworld
        if camtoworld.dim() == 2:
            camtoworld = camtoworld.unsqueeze(0)

        # Compute colors: appearance MLP or spherical harmonics
        if self.model_cfg.app_opt:
            app_colors: Tensor = self.app_module(
                features=self.splats["features"],
                embed_ids=image_id,
                dirs=means[None, :, :] - camtoworld[:, None, :3, 3],
                sh_degree=sh_degree,
            )
            colors: Tensor = torch.sigmoid(app_colors + self.splats["colors"])
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        # Rasterize
        splat_data = Splats(
            means=means, quats=quats, scales=scales,
            opacities=opas, colors=colors, sh_degree=sh_degree,
        )
        renders, alphas, info = self.renderer(
            splat_data, camera, **raster_kwargs,
        )
        out_colors: Tensor = renders[..., :3]

        # Apply bilateral grid color correction
        if self.bg_module is not None and image_id is not None:
            image_id_t = image_id if isinstance(image_id, Tensor) else torch.tensor([image_id], device=self.device)
            out_colors = bg_slice(
                self.bg_module,
                self._grid_xy.expand(out_colors.shape[0], -1, -1, -1),
                out_colors,
                image_id_t.unsqueeze(-1),
            )["rgb"]

        # Composite random background
        if self.model_cfg.random_bkgd:
            bkgd: Tensor = torch.rand(1, 3, device=self.device)
            out_colors = out_colors + bkgd * (1.0 - alphas)

        return out_colors, alphas, info

    def bilateral_grid_tv_loss(self) -> Tensor | float:
        if self.bg_module is not None:
            return total_variation_loss(self.bg_module.grids)
        return 0.0

    def reg_loss(self) -> Tensor | float:
        loss: Tensor | float = 0.0
        if self.training_cfg.opacity_reg > 0.0:
            loss = loss + self.training_cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
        if self.training_cfg.scale_reg > 0.0:
            loss = loss + self.training_cfg.scale_reg * torch.exp(self.splats["scales"]).mean()
        return loss

    def save_ckpt(self, path: str, step: int) -> None:
        data: dict[str, Any] = {"step": step, "splats": self.splats.state_dict()}
        if self.app_module is not None:
            data["app_module"] = self.app_module.state_dict()
        if self.bg_module is not None:
            data["bilateral_grid"] = self.bg_module.state_dict()
        torch.save(data, path)
