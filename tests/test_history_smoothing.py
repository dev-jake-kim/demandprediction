"""Valid temporal smoothing of local demand before tokenization."""

from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

from models.merged import MergedDemandConfig
from models.merged.modeling import LocalViewEncoder, MergedDemandModel


def _config(weights):
    return MergedDemandConfig(
        height=1, width=2, time_step=5, d_model=4, local_radius=0,
        weekday_dim=1, hour_dim=1, history_weights=weights,
        temperature_min=0.0, temperature_max=1.0, precipitation_max=1.0,
    )


def test_smoothed_local_matches_manual_valid_windows_and_context_alignment():
    torch.manual_seed(9)
    smoothed = LocalViewEncoder(_config([0.1, 0.2, 0.7])).eval()
    raw = LocalViewEncoder(_config(None)).eval()
    raw.load_state_dict(
        {key: value for key, value in smoothed.state_dict().items()
         if key != 'history_logits'}
    )
    demand = torch.tensor([[[[1., 10.]], [[2., 9.]], [[3., 8.]], [[4., 7.]], [[5., 6.]]]])
    context = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 20
    # conv1d performs cross-correlation: oldest -> newest = 0.1, 0.2, 0.7.
    manual = F.conv1d(
        demand.permute(0, 2, 3, 1).reshape(2, 1, 5),
        torch.tensor([[[0.1, 0.2, 0.7]]]),
    ).reshape(1, 1, 2, 3).permute(0, 3, 1, 2)
    torch.testing.assert_close(manual[:, :, 0, 0], torch.tensor([[2.6, 3.6, 4.6]]))
    torch.testing.assert_close(smoothed(demand, context), raw(manual, context[:, 2:]))
    smoothed(demand, context).square().sum().backward()
    assert smoothed.history_logits.grad is not None
    assert smoothed.history_logits.grad.abs().sum() > 0


def test_history_kernel_survives_checkpoint_roundtrip():
    model = MergedDemandModel(_config([0.1, 0.2, 0.7]))
    with torch.no_grad():
        model.local_view.history_logits.add_(torch.tensor([0.3, -0.2, 0.1]))
    with TemporaryDirectory() as path:
        model.save_pretrained(path)
        restored = MergedDemandModel.from_pretrained(path)
    assert restored.config.history_weights == [0.1, 0.2, 0.7]
    torch.testing.assert_close(restored.local_view.history_logits, model.local_view.history_logits)
