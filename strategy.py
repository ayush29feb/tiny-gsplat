from __future__ import annotations

import hydra
from gsplat.strategy import DefaultStrategy, MCMCStrategy


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
