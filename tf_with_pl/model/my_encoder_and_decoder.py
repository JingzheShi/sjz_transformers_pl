import torch
import torch.nn as nn
def get_attn_subsequence_mask(seq):
    #seq: [batch_size,seq_len,d_model]
    #mask: [batch_size,seq_len,seq_len], a upper-triangular matrix, with 1s on the upper triangle and 0s on the lower triangle and 0s on the diagonal
    mask=torch.zeros(seq.shape[0],seq.shape[1],seq.shape[1])

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,dropout=0.1,d_k=8,d_v=8,n_head=8):
        self.d_k=d_k
        super(ScaledDotProductAttention,self).__init__()
        self.softmax=torch.nn.Softmax(dim=-1)
        self.dropout=torch.nn.Dropout(dropout)
    def forward(self,Q,K,V,attn_mask):
        '''
        Q:[batch_size,n_heads,seq_len,d_k]
        K:[batch_size,n_heads,seq_len,d_k]
        V:[batch_size,n_heads,seq_len,d_v]
        attn_mask:[batch_size,seq_len,seq_len]
        '''
        scores=torch.matmul(Q,K.transpose(2,3))/torch.sqrt(torch.tensor(self.d_k))
        #scores: [batch_size,n_heads,seq_len,seq_len]
        scores.masked_fill(attn_mask>0.5,-1e4)
        
        #attn=self.dropout(dim=-1)(scores)

        #attn=self.dropout(attn)

        attn=self.softmax(scores)
        context=torch.matmul(attn,V)

        return context

class My_MultiheadAttn_with_addnorm(torch.nn.Module):
    def __init__(self,seq_len=50,d_model=512,n_heads=8,dropout=0.1,d_v=64):
        super(My_MultiheadAttn_with_addnorm,self).__init__()
        self.W_Q=torch.nn.Linear(d_model,d_model,bias=False)
        self.W_K=torch.nn.Linear(d_model,d_model,bias=False)
        self.W_V=torch.nn.Linear(d_model,d_model,bias=False)
        self.fc=nn.Linear(n_heads*d_v, d_model, bias=False)
        self.n_heads=n_heads
        self.d_k=int(d_model/n_heads)
        self.d_v=d_v
        self.layernorm=nn.LayerNorm(d_model)
    def forward(self,Q,K,V,attn_mask):
        '''
        Q:[batch_size,len_q,d_model]
        K:[batch_size,len_k,d_model]
        V:[batch_size,len_v=len_k,d_model]
        attn_mask:[batch_size,seq_len,seq_len]
        '''
        residual,batch_size=Q,Q.size(0)
        Q=self.W_Q(Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K=self.W_K(K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V=self.W_V(V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)

        #attn_mask:[batch_size,seq_len,seq_len]->[batch_size,n_heads,seq_len,seq_len]
        attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)

        #context: [batch_size, n_heads, len_q, d_v], attn_mask:[batch_size,n_heads,len_q,len_q]
        context= ScaledDotProductAttention(dropout=0.1,d_k=self.d_k)(Q,K,V,attn_mask)

        context=context.transpose(1,2).reshape(batch_size,-1,self.n_heads*self.d_v)

        otpt=self.fc(context) #[batch_size,len_q,d_model]

        return self.layernorm(otpt+residual)

class My_pointwise_feedforward_network_with_addnorm(torch.nn.Module):
    def __init__(self,d_model=512,d_ff=2048,dropout=0.1):
        super(My_pointwise_feedforward_network_with_addnorm,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff,bias=False)
        self.fc2=nn.Linear(d_ff,d_model,bias=False)
        self.relu=nn.ReLU()
        self.layernorm=nn.LayerNorm(d_model)
    def forward(self,x):
        return self.layernorm(x+self.fc2(self.relu(self.fc1(x))))


class My_EncoderLayer(nn.Module):
    def __init__(self,seq_len=50,d_model=512,n_heads=8,dropout=0.1,d_v=64,d_ff=2048):
        super(My_EncoderLayer, self).__init__()
        self.enc_self_attn=My_MultiheadAttn_with_addnorm(seq_len,d_model,n_heads,dropout,d_v)
        self.pos_ffn=My_pointwise_feedforward_network_with_addnorm(d_model,d_ff,dropout)
    def forward(self, enc_inpt, enc_self_attn_mask):
        enc_output=self.enc_self_attn(enc_inpt,enc_inpt,enc_inpt,enc_self_attn_mask)
        enc_output=self.pos_ffn(enc_output)
        return enc_output

class My_Encoder(nn.Module):
    def __init__(self,batch_size=64,N=6,seq_len=50,d_model=512,n_heads=8,dropout=0.1,d_v=64,d_ff=2048):
        super(My_Encoder,self).__init__()
        self.N=N
        self.layers=torch.nn.ModuleList([My_EncoderLayer(seq_len,d_model,n_heads,dropout,d_v,d_ff) for _ in range(N)])
        #self.device=device
        self.enc_self_attn_mask=torch.ones((batch_size,seq_len,seq_len)).long()
        self.enc_self_attn_mask=torch.nn.Parameter(self.enc_self_attn_mask,requires_grad=False)
        self.batch_size=batch_size
    def forward(self,enc_inpt,enc_mask):
        #enc_inpt:[batch_size,seq_len,d_model], enc_valid_length: [batch_size]
        #for enc_inpt[i][j]=torch.zeros[d_model], let enc_self_attn_mask[i][j]=torch.ones[seq_len]
        #for i in range(enc_valid_length.size(0)):
        #    self.enc_self_attn_mask[i,:enc_valid_length[i]-1,:enc_valid_length[i]-1]-=1
        #print(self.enc_self_attn_mask.size())
        '''
            for example,
            e.g. for ith sentence: [2,6,78,7,2017,28,3090,4,3,1,1,1,...], enc_valid_length[i]=9.
            Then enc_self_attn_mask[i][0][:9]=[0,0,0,0,0,0,0,0,1]
        '''
        for i in range(self.N):
            enc_inpt=self.layers[i](enc_inpt,enc_mask)
        #for i in range(enc_valid_length.size(0)):
        #    self.enc_self_attn_mask[i,:enc_valid_length[i]-1,:enc_valid_length[i]-1]+=1
        return enc_inpt

class My_DecoderLayer(nn.Module):
    def __init__(self,batch_size=64,seq_len=50,d_model=512,n_heads=8,dropout=0.1,d_v=64,d_ff=2048):
        super(My_DecoderLayer,self).__init__()
        self.dec_self_attn=My_MultiheadAttn_with_addnorm(seq_len,d_model,n_heads,dropout,d_v)
        self.dec_attn=My_MultiheadAttn_with_addnorm(seq_len,d_model,n_heads,dropout,d_v)
        self.pos_ffn=My_pointwise_feedforward_network_with_addnorm(d_model,d_ff,dropout)
        self.subsequence_mask=torch.zeros(seq_len,seq_len).long()
        #subsequence_mask: a upper-triangular matrix, with 1 on the upper triangle, and 0 elsewhere
        for i in range(seq_len):
            self.subsequence_mask[i,i+1:]=1
        '''
        For example, for a sentence of length 10, the upper-triangular matrix is:
        [[0,1,1,1,1,1,1,1,1,1],
         [0,0,1,1,1,1,1,1,1,1],
        [0,0,0,1,1,1,1,1,1,1],
        [0,0,0,0,1,1,1,1,1,1],
        [0,0,0,0,0,1,1,1,1,1],
        [0,0,0,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0]]
        '''
        self.subsequence_mask=self.subsequence_mask.view(1,seq_len,seq_len).repeat(batch_size,1,1)
        #self.subsequence_mask:[batch_size,seq_len,seq_len]
        self.subsequence_mask=torch.nn.Parameter(self.subsequence_mask,requires_grad=False)
    
    def forward(self,dec_inpt,enc_otpt,dec_self_attn_mask):
        dec_output=self.dec_self_attn(dec_inpt,dec_inpt,dec_inpt,dec_self_attn_mask+self.subsequence_mask[:dec_self_attn_mask.size(0)])
        dec_output=self.dec_attn(dec_output,enc_otpt,enc_otpt,dec_self_attn_mask)
        dec_output=self.pos_ffn(dec_output)
        return dec_output

class My_Decoder(nn.Module):
    def __init__(self,N=6,batch_size=64,seq_len=50,d_model=512,n_heads=8,dropout=0.1,d_v=64,d_ff=2048):
        super(My_Decoder,self).__init__()
        self.batch_size=batch_size
        self.N=N
        self.layers=torch.nn.ModuleList([My_DecoderLayer(batch_size,seq_len,d_model,n_heads,dropout,d_v,d_ff) for _ in range(N)])
        self.dec_self_attn_mask=torch.ones((batch_size,seq_len,seq_len)).long()
        self.dec_self_attn_mask=torch.nn.Parameter(self.dec_self_attn_mask,requires_grad=False)
    def forward(self,dec_inpt,enc_otpt,tgt_mask):
        #dec_inpt:[batch_size,seq_len,d_model], dec_valid_length: [batch_size]
        #for dec_inpt[i][j]=torch.zeros[d_model], let dec_self_attn_mask[i][j]=torch.ones[seq_len]
        #for i in range(dec_valid_length.size(0)):
        #    self.dec_self_attn_mask[i,:dec_valid_length[i]-1,:dec_valid_length[i]-1]-=1
        
        for i in range(self.N):
            dec_inpt=self.layers[i](dec_inpt,enc_otpt,tgt_mask)
        #for i in range(dec_valid_length.size(0)):
        #    self.dec_self_attn_mask[i,:dec_valid_length[i]-1,:dec_valid_length[i]-1]+=1
        return dec_inpt

