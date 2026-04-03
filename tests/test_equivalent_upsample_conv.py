import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from kokoro.istftnet import EquivalentUpsampleConv1d


def test_equivalent_upsample_conv_matches_convtranspose1d():
    channels = 8
    source = nn.ConvTranspose1d(
        channels,
        channels,
        kernel_size=3,
        stride=2,
        groups=channels,
        padding=1,
        output_padding=1,
    ).eval()
    replacement = EquivalentUpsampleConv1d(channels).eval()
    replacement.load_state_dict(source.state_dict())

    x = torch.randn(2, channels, 11)
    expected = source(x)
    actual = replacement(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_equivalent_upsample_conv_matches_weight_norm_state_dict():
    channels = 8
    source = weight_norm(
        nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            groups=channels,
            padding=1,
            output_padding=1,
        )
    ).eval()
    replacement = weight_norm(EquivalentUpsampleConv1d(channels)).eval()

    replacement.load_state_dict(source.state_dict())

    x = torch.randn(2, channels, 11)
    expected = source(x)
    actual = replacement(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)