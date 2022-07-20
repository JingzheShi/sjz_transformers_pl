import os
import numpy
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from model.my_tf import MyTF

from preprocessing.data_preprocessor import Data_preprocessor
def delete_ith_element(tensor_a,idx):
    lft=tensor_a[:idx]
    if(idx==tensor_a.size(0)-1):
        return lft
    else:
        rgt=tensor_a[idx+1:]
        return torch.cat((lft,rgt),dim=0)
def read_file(file_path):
    with open(file_path,'r') as f:
        return f.read()
if(__name__=='__main__'):
    arg_parser=ArgumentParser()
    arg_parser.add_argument('--realative_save_path',type=str,default='epoch=389-step=152880.ckpt')
    arg_parser.add_argument('--relative_translation_source_path',type=str,default='inpt.txt')
    arg=arg_parser.parse_args()
    path=os.getcwd()
    src_vocab=torch.load(os.path.join(path,'processed_dataset/src_vocab.pt'))
    tgt_vocab=torch.load(os.path.join(path,'processed_dataset/tgt_vocab.pt'))
    #obtain raw_txt from relative_translation_source_path
    raw_txt=read_file(os.path.join(path,arg.relative_translation_source_path))
    my_tf_model=MyTF(num_heads=8)
    #checkpoint=torch.load(os.path.join(path,arg.realative_save_path))
    #print(checkpoint.state_dict['d_model'])
    my_tf_model.load_from_checkpoint(os.path.join(path,arg.realative_save_path))
    my_tf_model.eval()

    #preprocess raw_txt

    data_preprocessor=Data_preprocessor(batch_size=384,num_steps=50,num_examples=None,num_workers=0)
    txt=data_preprocessor.preprocess_nmt(raw_txt)
    txt_source=data_preprocessor.tokenize_nmt_while_using(txt)
    src_array,src_valid_len=data_preprocessor.build_array_nmt(txt_source,src_vocab,data_preprocessor.num_steps)

    print(src_array.shape)
    print(src_valid_len.shape)
    print(src_array)
    print(src_valid_len)


    tgt_txt_0=[]
    for i in range(src_array.size(0)):
        tgt_txt_0.append([])
    
    
    tgt_array_0,tgt_valid_len_0=data_preprocessor.build_array_nmt(tgt_txt_0,tgt_vocab,data_preprocessor.num_steps)
    tgt_valid_len_0-=1

    print(tgt_array_0.shape)
    print(tgt_valid_len_0.shape)
    print(tgt_array_0)
    print(tgt_valid_len_0)

    final_tgt_lst=tgt_array_0.clone()
    final_valid_lst=tgt_valid_len_0.clone()
    final_valid_lst+=1
    #e.g. '我是学生。', final_vlid_lst[i]=7



    
    #otpt_2: batch_size * len_sequence
    #transform otpt_2 to 
    

    finish_flag=False
    while(finish_flag==False):
        otpt_1=torch.nn.functional.softmax(my_tf_model.linear(my_tf_model.forward(src_array,src_valid_len,tgt_array_0,tgt_valid_len_0)),dim=2)
        otpt_2=torch.argmax(otpt_1,dim=2)
        tgt_valid_len_0+=1
        to_be_deleted=[]
        for i in range(otpt_2.size(0)):
            #otpt_2[i] is of size [len_sequence]
            #otpt_3[i] is of size [len_sequence]
            #otpt_3[i][j]=:
                #if j==0:
                    #otpt_3[i][j]=2
                #else:
                    #otpt_3[i][j]=otpt_2[i][j-1]
            otpt_3=otpt_2[i].tolist()
            otpt_3=[2]+otpt_3[:-1]
            tgt_array_0[i]=torch.tensor(otpt_3)

            if(otpt_3[tgt_valid_len_0[0]]==3):
                final_tgt_lst[i]=tgt_array_0[i].clone()
                final_valid_lst[i]=tgt_valid_len_0[0]
                to_be_deleted.append(i)
        #sort to_be_deleted in descending order
        to_be_deleted.sort(reverse=True)
        if(len(to_be_deleted)==tgt_array_0.size(0)):
            finish_flag=True
        else:
            for i in to_be_deleted:
                tgt_array_0=delete_ith_element(tgt_array_0,i)
                tgt_valid_len_0=delete_ith_element(tgt_valid_len_0,i)
                


        

        print(otpt_2.shape)
        print(tgt_array_0)

    
    
    



    
