"""AVFORGE - cross-dataset audio-visual deepfake detection."""

import os

# Must be set before torch dispatches its first MPS op, so it goes here rather
# than in device.py: a few ops (AdaptiveAvgPool3d, some grid samplers) still
# have no MPS kernel, and without this they raise instead of falling back.
# The streams avoid those ops deliberately (see models/syncnet.py); this is the
# safety net, not the plan.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# albumentations phones home on import to check for updates. Disabled: it is
# a network call from a library import (slow and noisy offline, e.g. on a
# Colab runtime with no egress) and it has no bearing on reproducibility.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

__version__ = "0.1.0"
