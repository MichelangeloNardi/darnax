# test_conv_adapters.py
"""Runnable tests for conv adapters from the experiments/ folder.

This file ensures src/ is on sys.path when executing from experiments/
so imports like `darnax.modules.conv_adapters` work correctly.
"""
import sys
from pathlib import Path

# Add project `src/` directory to sys.path so `import darnax...` works
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import jax
import jax.numpy as jnp
import equinox as eqx

from darnax.modules.conv.conv_adapters import ConvAdapter, ConvRecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.layer_maps.sparse import LayerMap
from darnax.states.sequential import SequentialState
from darnax.modules.input_output import OutputLayer
from darnax.modules.fully_connected import FullyConnected

def test_conv_adapter():
    """Test ConvAdapter forward/backward."""
    key = jax.random.PRNGKey(0)
    
    # Create adapter: (28, 28, 1) → (28, 28, 16)
    adapt = ConvAdapter(
        h_in=28, w_in=28, c_in=1,
        c_out=16,
        kernel_size=3,
        h_out=28, w_out=28,
        strength=1.0,
        threshold=0.0,
        key=key,
        lr=0.1,
        weight_decay=0.001,
    )
    
    # Forward
    x = jax.random.normal(key, (4, 28*28*1))  # Batch of 4
    y = adapt(x)
    assert y.shape == (4, 28*28*16), f"Expected (4, 784), got {y.shape}"
    
    # Backward
    y_hat = jax.random.normal(key, y.shape)
    update = adapt.backward(x, y, y_hat)
    assert hasattr(update, 'kernel')
    assert update.kernel.shape == adapt.kernel.shape
    print("✓ ConvAdapter forward/backward OK")

def test_conv_recurrent():
    """Test ConvRecurrentDiscrete."""
    key = jax.random.PRNGKey(0)
    
    # Create recurrent: (28, 28, 16) → (28, 28, 16) with groups=16
    j_rec = ConvRecurrentDiscrete(
        h=28, w=28, channels=16,
        kernel_size=3,
        j_d=0.95,
        threshold=0.0,
        key=key,
        lr=0.1,
        weight_decay=0.001,
    )
    
    x = jax.random.normal(key, (4, 28*28*16))
    y = j_rec(x)
    assert y.shape == x.shape
    
    y_hat = jax.random.normal(key, y.shape)
    update = j_rec.backward(x, y, y_hat)
    assert hasattr(update, 'kernel')
    print("✓ ConvRecurrentDiscrete forward/backward OK")

def test_full_orchestrator():
    """Test full model with conv Win + conv J."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    
    # Build model: input(28×28×1) → hidden(28×28×16) → output(10)
    state = SequentialState((28*28*1, 28*28*16, 10))
    
    win = ConvAdapter(28, 28, 1, 16, 3, key=keys[0])
    j_rec = ConvRecurrentDiscrete(28, 28, 16, 3, j_d=0.95, key=keys[1])
    feedback = FullyConnected(10, 28*28*16, 1.0, 0.0, key=keys[2])
    output = FullyConnected(28*28*16, 10, 1.0, 0.0, key=keys[3])
    
    lmap = {1: {0: win, 1: j_rec, 2: feedback}, 2: {1: output, 2: OutputLayer()}}
    orch = SequentialOrchestrator(LayerMap.from_dict(lmap))
    
    # One step
    x = jax.random.normal(key, (4, 28*28*1))
    y = jax.random.normal(key, (4, 10)) * 2 - 1  # {-1, +1}
    
    rng = jax.random.PRNGKey(1)
    new_state, new_rng = orch.step(state.init(x), rng)
    assert new_state[1].shape == (4, 28*28*16)
    
    pred_state, new_rng2 = orch.predict(new_state, new_rng)
    pred = pred_state[-1]
    assert pred.shape == (4, 10)

    updates = orch.backward(pred_state, new_rng2)
    assert updates.lmap[1][0].kernel.shape == win.kernel.shape
    
    print("✓ Full orchestrator OK")

if __name__ == "__main__":
    test_conv_adapter()
    test_conv_recurrent()
    test_full_orchestrator()
    print("\n✅ All tests passed!")