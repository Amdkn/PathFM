"""Instance normalization shared by all released models."""

import torch


class FlooredInstanceNorm(torch.nn.Module):
    """Chronos-2 normalization with an explicit minimum context scale."""

    def __init__(self, inner: torch.nn.Module, floor: float):
        super().__init__()
        if floor <= 0:
            raise ValueError("floor must be positive")
        self.inner = inner
        self.floor = float(floor)

    @property
    def use_arcsinh(self) -> bool:
        return bool(self.inner.use_arcsinh)

    @property
    def eps(self) -> float:
        return float(self.inner.eps)

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is not None:
            return self.inner(x, loc_scale)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(
            (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
        )
        scale = scale.clamp(min=self.floor)
        scaled = (x - loc) / scale
        if self.use_arcsinh:
            scaled = torch.arcsinh(scaled)
        return (scaled.to(orig_dtype), (loc, scale))

    def inverse(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.inner.inverse(x, loc_scale)
