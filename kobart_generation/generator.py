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

import model
import dataset


class KoBARTCommentGenerator(model.Base):
    def __init__(self, model_path, max_seq_len, test_file):
        super(KoBARTCommentGenerator, self).__init__(model_path, max_seq_len)
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.test_file = test_file
        self.model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
        self.model = torch.load(self.model_path)
        self.model.eval()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = get_kobart_tokenizer()

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'],
                          labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def chat(self, text):
        input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        res_ids = self.model.generate(torch.tensor([input_ids]),
                                            max_length=self.max_seq_len,
                                            num_beams=5,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            bad_words_ids=[[self.tokenizer.unk_token_id]])        
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace('<s>', '').replace('</s>', '')

    def print_comment(self):
        while 1:
            q = input()
            if q=='quit':
                break
            print(self.model.chat(q))
    
    def making_comment_excel(self):
        predict_output = []
        test_data = pd.read_excel(self.test_file)
        for sentence in test_data['review']:
            row = []
            cnt = cnt + 1
            print(cnt)
            row.append(sentence)
            row.append(self.model.chat(sentence))
            predict_output.append(row)
        predict_output = pd.DataFrame(predict_output) #데이터 프레임으로 전환
        predict_output.to_excel(excel_writer='KoBART_predict_data.xlsx', encoding='utf-8') 

