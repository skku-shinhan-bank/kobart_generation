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

        ckpt = torch.load(self.hparams.model_path)
        kobart_model= KoBARTGenerationModel(self.hparams)
        kobart_model.load_state_dict(ckpt['model_state_dict'])
        kobart_model.eval()
        self.generation_model = kobart_model

    def print_comment(self, review, issue_id):
        return self.generation_model.chat(review, issue_id)
    
    def print_nbest_comment(self, review, issue_id):
        return self.generation_model.chat_nbest(review, issue_id)

    def make_comment_excel(self, file_path):
        predict_output = []
        test_data = pd.read_excel(file_path)
        cnt=0
        for i in range(len(test_data)):
            sentence = test_data['review'][i]
            issue_id = test_data['issue_id'][i]
            row = []
            cnt = cnt + 1
            print(cnt)
            row.append(sentence)
            row.append(self.generation_model.chat(sentence, issue_id))
            predict_output.append(row)
        predict_output = pd.DataFrame(predict_output) #데이터 프레임으로 전환
        predict_output.to_excel(excel_writer='KoBART_predict_data.xlsx', encoding='utf-8') 
