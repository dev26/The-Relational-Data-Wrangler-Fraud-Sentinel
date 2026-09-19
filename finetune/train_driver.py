#!/usr/bin/env python3
"""
LoRA training driver that works around a local environment collision:
this machine's anaconda ships MPICH, but MLX's default distributed init requires
Open MPI and hard-aborts (SIGABRT) when the C++ default-group path probes MPI
(triggered by all_sum / average_gradients inside mlx_lm's trainer).

Since this is a single-process (world size 1) run, every distributed collective is
an identity. We force the ring backend for init() and replace the collectives with
no-ops BEFORE importing the trainer, so MPI is never probed.
"""
import sys
import mlx.core as mx
import mlx.nn.utils as _nnu

# single-host ring group (size 1) — no MPI needed
_ring = mx.distributed.init(backend="ring")
mx.distributed.init = lambda *a, **k: _ring
# collectives are identities when world size == 1
mx.distributed.all_sum = lambda x, *a, **k: x
mx.distributed.all_gather = lambda x, *a, **k: x
if hasattr(mx.distributed, "all_reduce"):
    mx.distributed.all_reduce = lambda x, *a, **k: x
# gradient averaging across workers is a no-op with one worker
_nnu.average_gradients = lambda grads, *a, **k: grads

# hand remaining CLI args to mlx_lm's LoRA entrypoint (imported AFTER patching)
sys.argv = ["mlx_lm.lora"] + sys.argv[1:]
import mlx_lm.tuner.trainer as _t
_t.average_gradients = _nnu.average_gradients   # trainer bound it at import time
from mlx_lm.lora import main
main()
