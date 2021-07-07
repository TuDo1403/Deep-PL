from typing import Any
import pytorch_lightning as pl

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from libs.graphs.sequence_model import *
from libs.graphs.attention_model import *


class LanguageTranslation(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        self.net = eval(cfg.net.name)(**cfg.net.kwargs)
        self.criterion = getattr(nn, cfg.criterion)()
        