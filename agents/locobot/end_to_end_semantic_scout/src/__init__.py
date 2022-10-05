from typing import Dict
from .policy import (
    Net,
    Policy,
)
from .policy import Seq2SeqModel
from .sem_seg_policy import SemSegSeqModel
from .sem_seg_ft_policy import SemSegSeqFTModel, ObjectNavILPolicy
from .sem_seg_hm3d_policy import SemSegSeqHM3DModel

POLICY_CLASSES = {
    "Seq2SeqPolicy": Seq2SeqModel,
    "SemSegSeq2SeqPolicy": SemSegSeqModel,
    "SemSegFTSeq2SeqPolicy": SemSegSeqFTModel,
    "SemSegSeqHM3DPolicy": SemSegSeqHM3DModel,
    "ObjectNavILPolicy": ObjectNavILPolicy,
}

__all__ = [
    "Policy",
    "Net",
]
