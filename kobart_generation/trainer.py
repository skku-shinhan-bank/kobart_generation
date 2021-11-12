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

# import model
# import dataset
from .model import Base, KoBARTGenerationModel
from .dataset import CommentDataModule

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='/content/drive/MyDrive/신한은행/training-data/generate_data/kobart_crawled_shinhan_data.csv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='/content/drive/MyDrive/신한은행/training-data/generate_data/kobart_testdata.csv',
                            help='test file')

        parser.add_argument('--tokenizer_path',
                            type=str,
                            default="hyunwoongko/kobart",
                            help='tokenizer')
        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_seq_len',
                            type=int,
                            default=256,
                            help='max seq len')
        return parser

class KoBARTGenerationTrainer():
    def __init__(self, args):
        self.args=args
        # self.parser = argparse.ArgumentParser(description='KoBART Comment Generation')

        # self.parser.add_argument('--checkpoint_path',
        #                     type=str,
        #                     help='checkpoint path')

        # self.parser.add_argument('--chat',
        #                     action='store_true',
        #                     default=False,
        #                     help='response generation on given user input')

        # self.parser = self.hparams
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def train(self):
        # self.parser = Base.add_model_specific_args(self.parser)
        # self.parser = ArgsBase.add_model_specific_args(self.parser)
        # self.parser = CommentDataModule.add_model_specific_args(self.parser)
        self.args = pl.Trainer.add_argparse_args(self.args)
        self.args = self.parser.parse_args()
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
        torch.save({
            'model_state_dict': train_model.state_dict()
        }, 'output.pth')
        # torch.save(train_model.state_dict(), 'output.pth')
        print(train_model)


    