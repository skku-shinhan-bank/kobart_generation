# kobart_generation  
kobart generation  
## Install  
```  
pip install git+https://github.com/skku-shinhan-bank/koelectra_classification.git  
```    
## Train  
```python  
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html pytorch_lightning == 1.4.9

from kobart_generation import KoBARTGenerationTrainer

import easydict
import logging

args = easydict.EasyDict({
    'train_file' : 'train_data.csv',
    'test_file' : 'test_data.csv',
    'max_seq_len' : 128,
    'gradient_clip_val' : 1.0,
    'max_epochs' : 3,
    'model_path' : "hyunwoongko/kobart",
    'gpus' : 1,
    'tokenizer_path' : " ",
    'num_workers' : 5,
    'batch_size' : 32,
    'lr' : 5e-5,
    'warmup_ratio' : 0.1,
    'num_nodes' : 1
})

trainer = KoBARTGenerationTrainer(args)
trainer.train()
```  
  
**Train_data, Test_data form**  
  
|     Q    |     A     |
|:--------:|:---------:|
| Review 1 | Comment 1 |
| Review 2 | Comment 2 |
| Review 3 | Comment 3 |
| ...      | ...       |  
  
## Generate  
```python  
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html pytorch_lightning

import easydict

args = easydict.EasyDict({
    'model_path':'model_path.pth',
    'max_seq_len': 128
})

from kobart_generation import KoBARTCommentGenerator

<<<<<<< HEAD
comment_generator = KoBARTCommentGenerator(args)
=======
comment_generator = KoBARTCommentGenerator(args)  
>>>>>>> 93a92a294f5458ef9cd7ea499ae44a9531e1772f

# Chat form : Review -> Generate and print comment  
comment_generator.print_comment()  

# Review file -> Generate comments and store Excel file  
comment_generator.make_comment_excel('reveiws.xlsx')  
