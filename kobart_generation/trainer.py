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

class KoBARTConditionalGeneration(model.Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
        self.model.train()
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

    # def chat(self, text):
    #     input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
    #     res_ids = self.model.generate(torch.tensor([input_ids]),
    #                                         max_length=self.hparams.max_seq_len,
    #                                         num_beams=5,
    #                                         eos_token_id=self.tokenizer.eos_token_id,
    #                                         bad_words_ids=[[self.tokenizer.unk_token_id]])        
    #     a = self.tokenizer.batch_decode(res_ids.tolist())[0]
    #     return a.replace('<s>', '').replace('</s>', '')


parser = argparse.ArgumentParser(description='KoBART Comment Generation')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = model.Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = dataset.CommentDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    train_model = KoBARTConditionalGeneration(args)

    dm = dataset.CommentDataModule(args.train_file,
                        args.test_file,
                        os.path.join(args.tokenizer_path, 'model.json'),
                        max_seq_len=args.max_seq_len,
                        num_workers=args.num_workers)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1
                                                       )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(train_model, dm)
    torch.save(train_model.state_dict(), 'output.pth')
    
    # if args.chat:
    #     train_model.model.eval()

    #     predict_output = []
    #     cnt=0
    #     train_model.model.eval()
    #     predict_data_path = input()
    #     predict_data= pd.read_excel(predict_data_path)
    #     for sentence in predict_data['review']:
    #         row = []

    #         cnt = cnt + 1
    #         print(cnt)
    #         row.append(sentence)
    #         row.append(train_model.chat(sentence))

    #         predict_output.append(row)
        
    #     predict_output = pd.DataFrame(predict_output) #데이터 프레임으로 전환
    #     predict_output.to_excel(excel_writer='KoBART_predict_data.xlsx', encoding='utf-8') #엑셀로 저장          
        