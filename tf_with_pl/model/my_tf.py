import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.my_encoder_and_decoder import My_Encoder,My_Decoder


class MyTF(pl.LightningModule):
    def __init__(self,
                num_heads: int,
                dictionary_size=10000,
                tgt_dictionary_size=10000,
                padding_idx=1,
                d_model=512,
                seq_len=50,
                max_norm=15,
                dropout=0,
                batch_size=64,
                N=6,
                d_v=64,
                d_ff=2048,
                lr=0.00002
                    ):
        super(MyTF,self).__init__()
        self.padding_idx=padding_idx

        #embedding_dim when doing initial embedding
        self.d_model=torch.nn.Parameter(torch.tensor(d_model),requires_grad=False)
        embedding_dim=d_model
        

        #positional_encoding is batch_size * len_sequence * embedding_dim
        self.save_hyperparameters()
        self.embedding=nn.Embedding(num_embeddings=dictionary_size,embedding_dim=embedding_dim,padding_idx=padding_idx,max_norm=max_norm)
        self.tgtembedding=nn.Embedding(num_embeddings=tgt_dictionary_size,embedding_dim=embedding_dim,padding_idx=padding_idx,max_norm=max_norm)
        self.positional_encoding=torch.zeros([1,seq_len,d_model])
        for pos in range(seq_len):
            for i in range(int(d_model/2)):
                self.positional_encoding[0,pos,2*i]=torch.sin(torch.tensor(pos)/(10000**((2*i)/d_model)))
                self.positional_encoding[0,pos,2*i+1]=torch.cos(torch.tensor(pos)/(10000**((2*i+1)/d_model)))
        self.positional_encoding=torch.nn.Parameter(self.positional_encoding,requires_grad=False)

        self.encoder=My_Encoder(d_model=d_model,d_ff=d_ff,N=N,dropout=dropout,batch_size=batch_size,seq_len=seq_len)
        self.decoder=My_Decoder(d_model=d_model,d_ff=d_ff,N=N,dropout=dropout,batch_size=batch_size,seq_len=seq_len)
        self.linear=nn.Linear(d_model,tgt_dictionary_size,bias=True)
        #tgt_dictionary_size!!
        
        self.sftmax=nn.Softmax(dim=2)
        self.sqrt_d_model=torch.sqrt(torch.tensor(d_model))
        self.sqrt_d_model=torch.nn.Parameter(self.sqrt_d_model,requires_grad=False)

        self.lr=lr
        
    def forward(self,org_data,org_valid_mask,tgt_data,tgt_valid_mask):


        #data is batch_size * len_sequence

        #first, embed the data
        embedded_data=self.embedding(org_data)
        embedded_data=self.sqrt_d_model*embedded_data
        embedded_tgt_data=self.tgtembedding(tgt_data)
        embedded_tgt_data=self.sqrt_d_model*embedded_tgt_data
        #weights *=\sqrt{self.d_model}, so that the norm of embedded_data is approx 10*d_model, similar to the norm of positional_encoding

        #secondly, do positional encoding
        encoder_inpt=embedded_data+self.positional_encoding
        decoder_inpt=embedded_tgt_data+self.positional_encoding
        
        #thirdly, do encoder
        encoder_outpt=self.encoder(encoder_inpt,org_valid_mask)

        #fourthly, do decoder
        decoder_outpt=self.decoder(encoder_outpt,decoder_inpt,tgt_valid_mask)


        return decoder_outpt

    def computing_loss(self,otpt,tgt_data):
        #otpt:[batch_size,len_sequence,dictionary_size]
        #tgt_data:[batch_size,len_sequence]
        #tgt_data_valid_len:[batch_size]
        #loss is a scalar
        loss=F.cross_entropy(otpt[:,:-1,:].reshape(-1,otpt.shape[-1]),tgt_data[:,1:].reshape(-1).long(),ignore_index=self.padding_idx,reduction='sum')
        
        return loss
    
    def training_step(self,batch,batch_idx):
        org_data,org_data_valid_len,tgt_data,tgt_data_valid_len,org_valid_mask,tgt_valid_mask=batch
        otpt=self.forward(org_data,org_valid_mask,tgt_data,tgt_valid_mask)
        #otpt:[batch_size,len_sequence,d_model]
        otpt=otpt-self.positional_encoding
        otpt=self.linear(otpt)
        #otpt:[batch_size,len_sequence,dictionary_size]

        #otpt:[batch_size,len_sequence,dictionary_size]
        loss=self.computing_loss(otpt,tgt_data)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss':loss}
    
    def validation_step(self,batch,batch_idx):
        org_data,org_data_valid_len,tgt_data,tgt_data_valid_len,org_valid_mask,tgt_valid_mask=batch
        otpt=self.forward(org_data,org_valid_mask,tgt_data,tgt_valid_mask)
        #otpt:[batch_size,len_sequence,d_model]
        otpt=otpt-self.positional_encoding
        otpt=self.linear(otpt)
        #otpt:[batch_size,len_sequence,dictionary_size]
        
        #otpt:[batch_size,len_sequence,dictionary_size]
        loss=self.computing_loss(otpt,tgt_data)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss':loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if(epoch % 30==29):
            self.lr/=1.8
        optimizer.step(closure=optimizer_closure)

        






    @staticmethod
    def add_model_specific_args(parent_parser):
        parser_group=parent_parser.add_argument_group('MyTF')
        parser_group.add_argument('--num_heads',type=int,default=8)
        parser_group.add_argument('--d_model',type=int,default=512)
        parser_group.add_argument('--d_ff',type=int,default=2048)
        parser_group.add_argument('--N',type=int,default=6)
        return parent_parser

    