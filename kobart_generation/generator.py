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

class KoBARTCommentGenerator(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTCommentGenerator, self).__init__(hparams, **kwargs)
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(ctx)

        print(self.hparams.model_path)
        print(self.hparams.max_seq_len)
        print("=================")

        ckpt = torch.load(self.hparams.model_path)

        kobart_model= KoBARTGenerationModel(self.hparams)
        kobart_model.load_state_dict(ckpt['model_state_dict'])
        kobart_model.eval()

        self.generation_model = kobart_model
        print(self.generation_model)

    def print_comment(self):
        while 1:
            q = input()
            if q=='quit':
                break
            print(self.generation_model.chat(q))
    
    def make_comment_excel(self, file_path):
        predict_output = []
        test_data = pd.read_excel(file_path)
        for sentence in test_data['review']:
            row = []
            cnt = cnt + 1
            print(cnt)
            row.append(sentence)
            row.append(self.generation_model.chat(sentence))
            predict_output.append(row)
        predict_output = pd.DataFrame(predict_output) #데이터 프레임으로 전환
        predict_output.to_excel(excel_writer='KoBART_predict_data.xlsx', encoding='utf-8') 

    def print_nbest_comment(self, review):
        self.generation_model.chat_nbest(review)
