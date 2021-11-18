import argparse
import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from transformers import (BartForConditionalGeneration,
                          PreTrainedTokenizerFast)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from kobart_transformers import get_kobart_tokenizer

from .model import Base, KoBARTGenerationModel
from .dataset import CommentDataModule


class KoBARTGenerationTrainer():
    def __init__(self, args):
        self.args=args
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def train(self):
        logging.info(self.args)
        print(self.args)

        train_model = KoBARTGenerationModel(self.args)

        dm = CommentDataModule(self.args.train_file,
                            self.args.test_file,
                            os.path.join(self.args.tokenizer_path, 'model.json'),
                            max_seq_len=self.args.max_seq_len,
                            num_workers=self.args.num_workers)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                            filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                            verbose=True,
                                                            save_last=True,
                                                            mode='min',
                                                            save_top_k=-1
                                                            )
        tb_logger = pl_loggers.TensorBoardLogger(os.path.join('tb_logs'))
        lr_logger = pl.callbacks.LearningRateMonitor()
        trainer = pl.Trainer.from_argparse_args(self.args, logger=tb_logger,
                                                callbacks=[checkpoint_callback, lr_logger])
        trainer.fit(train_model, dm)   

        train_model.model.eval()

        torch.save({
            'model_state_dict': train_model.state_dict()
        }, 'output.pth')
        
    