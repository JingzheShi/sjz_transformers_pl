import torch
import os
from d2l import torch as d2l
import pytorch_lightning as pl
class Data_preprocessor():
    def __init__(self,batch_size=64,num_steps=50,num_examples=None,num_workers=0,valid_ratio=0.1):
        self.valid_ratio=valid_ratio
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.num_examples=num_examples
        self.num_workers=num_workers
    def save_dataset(self):
        d2l.DATA_HUB['fra-eng']=(d2l.DATA_URL+'fra-eng.zip','94646ad1522d915e7b0f9296181140edcf86a4f5114514')
    def read_data_nmt(self):
        try:
            data_dir=os.getcwd()+"\\..\\data\\fra-eng"
            with open(os.path.join(data_dir,'fra.txt'),'r',encoding='UTF-8') as f:
                raw_txt=f.read()
                return raw_txt
        except:
            data_dir=d2l.download_extract('fra-eng')
            # print(data_dir)
            # ..\data\frag-eng
            self.save_dataset()
            with open(os.path.join(data_dir,'fra.txt'),'r',encoding='UTF-8') as f:
                raw_txt=f.read()
                return raw_txt
    def preprocess_nmt(self,text):
        """Preprocess the English-French dataset."""
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        # Replace non-breaking space with space, and convert uppercase letters to
        # lowercase ones
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # Insert space between words and punctuation marks
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text)]
        return ''.join(out)
    def tokenize_nmt(self,text):
        num_examples=self.num_examples
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if num_examples and i > num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))
        return source, target
    
    def tokenize_nmt_while_using(self,text):
        num_examples=self.num_examples
        source=[]
        for i, line in enumerate(text.split('\n')):
            if num_examples and i > num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 1:
                source.append(parts[0].split(' '))
        return source

    

    def truncate_pad(self,line,num_steps,padding_token):
        if len(line)>num_steps:
            return line[:num_steps]
        return line+[padding_token]*(num_steps-len(line))

    def build_array_nmt(self,lines,vocab,num_steps):
        #print('vocab[pad]==',end='')
        #print(vocab['<pad>']) printed:1
        #print('vocab[bos]==',end='')
        #print(vocab['<bos>']) printed:2
        #print('vocab[eos]==',end='')
        #print(vocab['<eos>']) printed:3
        lines=[vocab [l] for l in lines]
        lines=[[vocab['<bos>']]+l+[vocab['<eos>']] for l in lines]
        array=torch.tensor([self.truncate_pad(l,num_steps,vocab['<pad>']) for l in lines])
        valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
        return array, valid_len

    def load_data_nmt(self):
        text=self.preprocess_nmt(self.read_data_nmt())
        source,target=self.tokenize_nmt(text)
        
        src_vocab=d2l.Vocab(source,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])
        tgt_vocab=d2l.Vocab(target,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])
        src_array, src_valid_len = self.build_array_nmt(source, src_vocab, self.num_steps)
        tgt_array, tgt_valid_len = self.build_array_nmt(target, tgt_vocab, self.num_steps)

        src_dictionary_size=len(src_vocab)
        tgt_dictionary_size=len(tgt_vocab)

        #data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

        #permute src_array, src_valid_len, tgt_array, tgt_valid_len in dimension 0 randomly but in the same way.
        perm = torch.randperm(len(src_array))
        src_array=src_array[perm]
        src_valid_len=src_valid_len[perm]
        tgt_array=tgt_array[perm]
        tgt_valid_len=tgt_valid_len[perm]

        src_valid_mask = torch.ones([src_valid_len.size(0), self.num_steps, self.num_steps], dtype=torch.int32)
        for i in range(src_valid_len.size(0)):
            src_valid_mask[i, :src_valid_len[i], :src_valid_len[i]] = 0
        tgt_valid_mask = torch.ones([tgt_valid_len.size(0), self.num_steps, self.num_steps], dtype=torch.int32)
        for i in range(tgt_valid_len.size(0)):
            tgt_valid_mask[i, :tgt_valid_len[i]-1, :tgt_valid_len[i]-1] = 0
        #tgt: valid_len-1 tokens -> valid_len-1 otpts.

        #split the data into train and valid
        num_train=int(len(src_array)*(1-self.valid_ratio))
        train_data=(src_array[:num_train],src_valid_len[:num_train],tgt_array[:num_train],tgt_valid_len[:num_train],src_valid_mask[:num_train],tgt_valid_mask[:num_train])
        valid_data=(src_array[num_train:],src_valid_len[num_train:],tgt_array[num_train:],tgt_valid_len[num_train:],src_valid_mask[num_train:],tgt_valid_mask[num_train:])


        
        train_dataset=torch.utils.data.TensorDataset(*train_data)
        valid_dataset=torch.utils.data.TensorDataset(*valid_data)
        train_iter=torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        valid_iter=torch.utils.data.DataLoader(valid_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
        #train_iter,valid_iter=torch.utils.data.random_split(data_iter,(int(len(data_iter)*0.8),int(len(data_iter)*0.2)))
        return train_iter,valid_iter,src_vocab,tgt_vocab,src_dictionary_size,tgt_dictionary_size
    def to_cuda_(self,lst):
        for item in lst:
            item=item.cuda()
    def process_data(self):
        train_data_iter,valid_data_iter,src_vocab,tgt_vocab,src_vocab_size,tgt_vocab_size=self.load_data_nmt()
        #data_iter,src_vocab,tgt_vocab=data_preprocessor.to_cuda_(data_preprocessor.load_data_nmt())

        current_path=os.getcwd()+'/processed_dataset/'
        #save to /current_path/processed_dataset
        #print(src_vocab.to_tokens(1)) printed: <pad>
        #print(src_vocab['<pad>']) printed: 1
        torch.save(train_data_iter,os.path.join(current_path,'train_data_loader.pt'))
        torch.save(valid_data_iter,os.path.join(current_path,'valid_data_loader.pt'))
        
        torch.save(src_vocab,os.path.join(current_path,'src_vocab.pt'))
        torch.save(tgt_vocab,os.path.join(current_path,'tgt_vocab.pt'))

        dictionary_size_list=[src_vocab_size,tgt_vocab_size]
        torch.save(dictionary_size_list,os.path.join(current_path,'dictionary_size_list.pt'))



'''
The five .pt files are:
train_data_loader.pt: the training data loader
    each batch in the data loader has four tensors:
        src_array: the source sentence array, batch_size*len_sequence, begin with vocab['<bos>']=2, end with vocab['<eos>']=3, and padding with vocab['<pad>']=1
        src_valid_len: the valid length of the source sentence, batch_size*1. The valid length means the length of the original sentence +2(<bos> and <eos>).
            For example: if the original sentence is "I am a student.", the valid length is 7 (including '.').
        tgt_array: the target sentence array, batch_size*len_sequence, begin with vocab['<bos>']=2, end with vocab['<eos>']=3, and padding with vocab['<pad>']=1
        tgt_valid_len: the valid length of the target sentence, batch_size*1. The valid length means the length of the original sentence +2(<bos> and <eos>).
            For example: if the original sentence is "我是学生。", the valid length is 7 (including '。').

valid_data_loader.pt
    similar to train_data_loader.pt

src_vocab.pt
tgt_vocab.pt

'''








if(__name__=='__main__'):
    data_preprocessor=Data_preprocessor(batch_size=3,num_steps=8,num_examples=600)
    train_data_iter,valid_data_iter,src_vocab,tgt_vocab=data_preprocessor.load_data_nmt()
    #data_iter,src_vocab,tgt_vocab=data_preprocessor.to_cuda_(data_preprocessor.load_data_nmt())

    current_path=os.getcwd()+'/processed_dataset/'
    #save to /current_path/processed_dataset
    print(src_vocab.to_tokens(1))
    print(src_vocab['<pad>'])
    torch.save(train_data_iter,os.path.join(current_path,'train_data_loadder.pt'))
    torch.save(valid_data_iter,os.path.join(current_path,'valid_data_loader.pt'))
    torch.save(src_vocab,os.path.join(current_path,'src_vocab.pt'))
    torch.save(tgt_vocab,os.path.join(current_path,'tgt_vocab.pt'))

