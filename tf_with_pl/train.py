import os
import numpy
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from model.my_tf import MyTF
from preprocessing import pl_dataloader
if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('--data_preprocessing',type=bool,default=True)
    parser.add_argument('--weight_path',type=str,default='weights/')
    parser.add_argument('--monitor',type=str,default='val_loss',choices=['val_loss'])
    parser.add_argument('--gpus',type=int,default=1)
    parser.add_argument('--params_saving_path',type=str,default='params/')
    parser.add_argument('--train_batch_size',type=int,default=192)
    #384 for RTX 3090
    parser.add_argument('--max_epochs', type=int, default=512)
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='saved_models/')
    parser.add_argument('--top_k', type=int, default=3)
    
    parser.add_argument('--num_workers',type=int,default=14)
    parser.add_argument('--precision',type=int, default=16,choices=[16,32])



    #Phrases longer than this will be truncated, shorter ones will be padded.
    parser.add_argument('--num_steps',type=int,default=50)
    
    #example
    parser.add_argument('--num_examples',type=int,default=None)
    parser=MyTF.add_model_specific_args(parser)
    args=parser.parse_args()
    #print(args)
    
    if(args.data_preprocessing==True):
        from preprocessing.data_preprocessor import Data_preprocessor
        data_preprocessor=Data_preprocessor(batch_size=args.train_batch_size,num_steps=args.num_steps,num_examples=args.num_examples,num_workers=args.num_workers)
        data_preprocessor.process_data()
        
    path=os.getcwd()
    
    
    #lst=torch.load(os.path.join(path,'processed_dataset\\scr_array___src_valid_len___tgt_array___tgt_valid_len.pt'))
    '''s=0
    for X,X_valid_len, Y, Y_valid_len in lst:
        s+=1
        print(s)
        print(X)
        print(X.type())

        if(s==10):
            break'''

    from preprocessing.pl_dataloader import PL_DataLoader
    train_data_loader=torch.load(os.path.join(path,'processed_dataset/train_data_loader.pt'))
    val_data_loader=torch.load(os.path.join(path,'processed_dataset/valid_data_loader.pt'))
    dictionary_size_list=torch.load(os.path.join(path,'processed_dataset/dictionary_size_list.pt'))
    src_vocab_size=dictionary_size_list[0]
    tgt_vocab_size=dictionary_size_list[1]
    pl_dataloader=PL_DataLoader()
    pl_dataloader.naive_setup(train_data_loader,val_data_loader)

    #print src_vocab_size and tgt_vocab_size
    #print('src_vocab_size:',src_vocab_size) 10012
    #print('tgt_vocab_size:',tgt_vocab_size) 17851

    #part of the arg parser is model specific, so we need to pass the partial args to the model
    model=MyTF(
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        N=args.N,
        seq_len=args.num_steps,
        dictionary_size=src_vocab_size,
        tgt_dictionary_size=tgt_vocab_size,
        batch_size=args.train_batch_size,
    )

    #checkpoint
    checkpoint_callback=pl.callbacks.ModelCheckpoint(
        monitor=args.monitor,
        dirpath=args.save_dir,
        save_top_k=args.save_top_k,
        mode='min',
        every_n_epochs=1,
    )


    #Train.
    trainer=pl.Trainer(callbacks=[checkpoint_callback],accelerator="gpu",devices="auto",strategy="ddp_find_unused_parameters_false",max_epochs=args.max_epochs,precision=args.precision)
    trainer.fit(model,pl_dataloader)

