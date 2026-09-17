# PAN baselines

Memory baselines for [PAN](https://github.com/beimingli0626/mapnav), the pose-attention navigator. Each one replaces PAN's pose-indexed spatial context with a different way of storing the same tour, and is trained and evaluated by PAN's own pipeline, on the same frozen keyframes, readout and action decoder — the memory is the only axis that differs.

| arch | memory |
|---|---|
| `lstm` | cuDNN LSTM over the keyframes in tour order |
| `sru` | Spatially-Enhanced Recurrent Unit (Yang et al., IJRR 2025) |
| `sru_gate` | the SRU cell with the refine-gate forget update |
| `xfmr` | decoder-only transformer over the tour as a causal token stream, 1-D RoPE by rank |
| `baselines.classical` | OctoMap occupancy grid of the tour's depth frames + A*, no learning |

## Installation

Install [PAN](https://github.com/beimingli0626/mapnav) first — it brings the simulator, the encoder and the data pipeline. Then, in the same environment:

```bash
pip install -e .
```

## Usage

`python -m baselines` is the PAN command with these archs registered, so every PAN flag applies unchanged:

```bash
torchrun --nproc_per_node=1 -m baselines train --arch lstm --run-name lstm <pan train flags>
python -m baselines eval --ckpt logs/lstm/checkpoints/final.pt <pan eval flags>
python -m baselines eval_accum ...
```

`--help` lists every field with its default; the architecture defaults live in `RecurrentConfig` (`baselines/recurrent.py`) and `SeqFormerConfig` (`baselines/seqformer.py`). `--no-motion-input` drops the keyframe-to-keyframe motion input, leaving the memory no frame-to-frame adjacency signal.

The classical baseline has no checkpoint and runs on its own:

```bash
python -m baselines.classical --token-root $DATA/tokens/val --traj-root $DATA/traj/val \
    --bank $DATA/banks/bank_val.npz --eta 10 --exec-horizon 4 --workers 16 --out logs/classical/eval.json
```

It reads the learned planners' evaluation inputs — the context frames' depth and camera poses, the agent's pose at each query, the goal position — and the navmesh only moves and scores the agent.
