# import argparse
# import logging
# import os

# import numpy as np
# import pandas as pd
# import pytorch_lightning as pl
# import torch
# import transformers
# from pytorch_lightning import loggers as pl_loggers
# from torch.utils.data import DataLoader, Dataset
# from transformers import (BartForConditionalGeneration,
#                           PreTrainedTokenizerFast)
# from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
# from kobart_transformers import get_kobart_tokenizer

# from .model import Base

# class KoBARTConditionalGeneration(Base):
#     def __init__(self, hparams, **kwargs):
#         super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
#         self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
#         self.model.train()
#         self.bos_token = '<s>'
#         self.eos_token = '</s>'
#         self.tokenizer = get_kobart_tokenizer()

#     def chat(self, text):
#         input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
#         res_ids = self.model.generate(torch.tensor([input_ids]),
#                                             max_length=self.hparams.max_seq_len,
#                                             num_beams=5,
#                                             eos_token_id=self.tokenizer.eos_token_id,
#                                             bad_words_ids=[[self.tokenizer.unk_token_id]])        
#         a = self.tokenizer.batch_decode(res_ids.tolist())[0]
#         return a.replace('<s>', '').replace('</s>', '')