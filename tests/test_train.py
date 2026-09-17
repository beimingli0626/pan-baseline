"""The baseline archs in PAN's TrainConfig: registration, arch-specific flags, checkpoint
round trips."""
import pytest
import torch

import baselines  # noqa: F401  (registers the archs)
from baselines.recurrent import RecurrentConfig
from baselines.seqformer import SeqFormerConfig
from pan.planner.train import build_model, load_planner, parse_config, save_checkpoint

REQUIRED = "--run-name r --token-root t --val-token-root v".split()


def test_model_flags_follow_the_arch():
    cfg, _ = parse_config(REQUIRED + ("--arch sru --rnn-hidden 64 --rnn-head-dim 32 --rnn-compile --summary-dim 96 "
                                      "--rnn-readout-dim 48 --rnn-readout-blocks 2 --act-emb-dim 16").split())
    assert cfg.model == RecurrentConfig(cell="sru", summary_dim=96, hidden=64, head_hidden=32, readout_dim=48,
                                        readout_blocks=2, act_emb_dim=16, compile_step=True)
    cfg, _ = parse_config(REQUIRED + "--arch xfmr --xfmr-layers 2 --model-dim 64".split())
    assert isinstance(cfg.model, SeqFormerConfig) and (cfg.model.layers, cfg.model.model_dim) == (2, 64)
    assert cfg.model.motion_input
    cfg, _ = parse_config(REQUIRED + "--arch xfmr --no-xfmr-motion-input".split())
    assert not cfg.model.motion_input
    cfg, _ = parse_config(REQUIRED + "--arch sru --no-rnn-motion-input".split())
    assert isinstance(cfg.model, RecurrentConfig) and not cfg.model.motion_input
    with pytest.raises(SystemExit):                      # the Planner's flag is not the lstm's
        parse_config(REQUIRED + "--arch lstm --pos-mode temporal".split())


READOUT = "--rnn-readout-dim 16 --rnn-readout-blocks 1 --act-emb-dim 8"


@pytest.mark.parametrize("arch,flags", [
    ("lstm", f"--rnn-hidden 16 --summary-dim 32 {READOUT}"),
    ("sru_gate", f"--rnn-hidden 16 --summary-dim 32 {READOUT}"),
    ("xfmr", "--model-dim 32 --summary-dim 32 --xfmr-layers 1 --xfmr-num-heads 2"),
])
def test_checkpoint_roundtrip(tmp_path, arch, flags):
    cfg, _ = parse_config(REQUIRED + f"--arch {arch} {flags}".split())
    torch.manual_seed(0)
    model = build_model(cfg)
    save_checkpoint(str(tmp_path / "ck.pt"), model, cfg, step=3)
    loaded, loaded_cfg = load_planner(str(tmp_path / "ck.pt"))
    assert loaded_cfg == cfg and type(loaded) is type(model)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
