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
        # kobart_model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")
        kobart_model= KoBARTGenerationModel(self.hparams)

        kobart_model.load_state_dict(ckpt['model_state_dict'])
        # kobart_model.model.save_pretrained("working_dir")
        # checkpoint = torch.load(self.hparams.model_path, map_location=device)
        # kobart_model.load_state_dict(checkpoint['state_dict'])
        # kobart_model.load_state_dict(ckpt)
        # kobart_model.load_state_dict(checkpoint, strict=False)
        # kobart_model.load_state_dict(torch.load(self.hparams.model_path))
        kobart_model.eval()

        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = get_kobart_tokenizer()
        self.generation_model = kobart_model
        print(self.generation_model)

    # def chat_nbest(self, text):
    #     input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
    #     res_ids = self.generation_model.generate(torch.tensor([input_ids]),
    #                                         max_length=self.hparams.max_seq_len,
    #                                         num_beams=5,
    #                                         eos_token_id=self.tokenizer.eos_token_id,
    #                                         bad_words_ids=[[self.tokenizer.unk_token_id]])
    #     result = []
    #     for i in range(0, 3):
    #         print(i)
    #         print(self.tokenizer.batch_decode(res_ids.tolist()))
    #         a = self.tokenizer.batch_decode(res_ids.tolist())[0]
    #         a.replace('<usr>', '').replace('<s>', '').replace('</s>', '')
    #         print(a)
    #         print("\n")
    #         result.append(a)
        
    #     print(result)
    #     return result

    def print_nbest_comment(self, review):
        self.generation_model.chat_nbest(review)

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

