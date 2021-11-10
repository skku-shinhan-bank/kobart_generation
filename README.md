# kobart_generation  
kobart generation  
## Install  
```  
pip install git+https://github.com/skku-shinhan-bank/koelectra_classification.git  
```  
```  
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html pytorch_lightning==1.4.9  
```  
## Train  
```python  
python trainer.py --train_file 'train_file_path' --test_file 'test_file_path' --max_seq_len 128 --gradient_clip_val 1.0 --max_epochs 3 --default_root_dir logs --chat --gpus 1  
  
train_data, test_data form  
  
|   Q    |    A    |
|---|---|
|Review 1|Comment 1|
|Review 2|Comment 2|
|Review 3|Comment 3|
|  ...   |   ...   |  

```  

## Generate  
```python  
import easydict  
  
args = easydict.EasyDict({
    'model_path':'model_path_from_trainer',
    "max_seq_len": 128
})  

import generator  

comment_generator = generator.KoBARTCommentGenerator(args)  

# Chat form : Review -> Generate and print comment
comment_generator.print_comment() 

# Review file -> Generate comments and store Excel file
comment_generator.make_comment_excel('file_path') 
