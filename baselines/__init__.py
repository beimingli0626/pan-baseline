"""Baselines for the PAN planner, trained and evaluated with PAN's own pipeline:
recurrent memories (lstm, sru, sru_gate) and a sequence transformer (xfmr).

    torchrun --nproc_per_node=2 -m baselines train --arch lstm <pan train flags>
    python -m baselines eval --ckpt <ckpt> ...        python -m baselines eval_accum ...

Importing the package registers its archs with `pan.planner.train.ARCHS`.
"""
from pan.planner.train import ARCHS

from .recurrent import CELLS, RecurrentConfig, RecurrentPlanner
from .seqformer import SeqFormer, SeqFormerConfig

ARCHS.update({cell: (RecurrentPlanner, RecurrentConfig, "rnn", {"cell": cell}) for cell in CELLS})
ARCHS["xfmr"] = (SeqFormer, SeqFormerConfig, "xfmr", {})
