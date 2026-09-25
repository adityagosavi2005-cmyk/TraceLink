"""torchvision compatibility shim for the GFPGAN inference stack (Phase 8).

Upstream incompatibility: gfpgan 1.3.8 depends on basicsr 1.4.2,
whose ``basicsr/data/degradations.py`` does::

    from torchvision.transforms.functional_tensor import rgb_to_grayscale

``torchvision.transforms.functional_tensor`` was removed from
torchvision (0.15+). On the pinned torch==2.9.1+cpu /
torchvision==0.24.1+cpu runtime the unpatched import fails with::

    ModuleNotFoundError: No module named
    'torchvision.transforms.functional_tensor'

This module registers a synthetic module under exactly that name
whose ``rgb_to_grayscale`` delegates to the surviving, numerically
identical ``torchvision.transforms.functional.rgb_to_grayscale``
kernel (legacy vs v2 kernels agree to ~1e-07 on random input; this
is the only symbol basicsr/gfpgan import from the removed
namespace).

Scope: call :func:`ensure_gfpgan_compat` ONLY from the GFPGAN
adapter's lazy load path (gfpgan_adapter._load), immediately before
``from gfpgan import GFPGANer``. Never import this module at
application startup. No site-packages and no BasicSR sources are
modified.
"""

import sys
import types

_SHIM_MODULE_NAME = "torchvision.transforms.functional_tensor"


def ensure_gfpgan_compat() -> None:
    """Register the synthetic functional_tensor module (idempotent)."""
    if _SHIM_MODULE_NAME in sys.modules:
        return
    from torchvision.transforms import functional as _modern

    shim = types.ModuleType(_SHIM_MODULE_NAME)
    shim.__doc__ = (
        "Local TraceLink shim over "
        "torchvision.transforms.functional for the gfpgan/basicsr "
        "runtime (see app/services/gfpgan_compat.py)."
    )
    shim.rgb_to_grayscale = _modern.rgb_to_grayscale
    sys.modules[_SHIM_MODULE_NAME] = shim
    import torchvision.transforms as _transforms_pkg

    setattr(_transforms_pkg, "functional_tensor", shim)
