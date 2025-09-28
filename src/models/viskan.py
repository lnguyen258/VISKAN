import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import *  


# B --> Batch Size
# C --> Number of Input Channels
# IH --> Image Height
# IW --> Image Width
# P --> Patch Size
# E --> Embedding Dimension
# N --> Number of Patches = IH/P * IW/P
# S --> Sequence Length   = IH/P * IW/P + 1 or N + 1 (extra 1 is of Classification Token)
# Q --> Query Sequence length (equal to S for self-attention)
# K --> Key Sequence length   (equal to S for self-attention)
# V --> Value Sequence length (equal to S for self-attention)
# H --> Number of heads
# HE --> Head Embedding Dimension = E/H
# CL --> Number of Classes

class EmbedLayer(nn.Module):
    """
    1. Break image into patches
    2. Embed patches into numbers using a Conv2D 
    3. Add a learnable positional embedding vector to all patches
    4. Finally, a classification token is added to classify

    Parameters:
        n_channels (int) : Number of channels of the input image
        embed_dim  (int) : Embedding dimension
        image_size (int) : Image size
        patch_size (int) : Patch size
        dropout  (float) : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH
    
    Returns:
        Tensor: Embedding of the image of shape B, S, E
    """  

    # Define layers
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout):
        super().__init__()

        # patch encoding layer
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # positional embedding layer
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size//patch_size)**2, embed_dim), requires_grad=True)

        # classification tokenlayer
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # dropout 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]          
        x = self.conv1(x)                      # B,C,IH,IW --> B,E,IH/P,IW/P
        x = x.reshape([batch_size, x.shape[1], -1])     # B,E,IH/P,IW/P --> B,E,IH/P*IW/P
        x = x.permute(0, 2, 1)                 # B,E,N --> B,N,E
        x = x + self.pos_embedding             # B,N,E --> B,N,E
        x = torch.cat((torch.repeat_interleave(self.cls_token,batch_size,0),x), dim=1)
                                               # B,N,E --> B,N+1,E
        x = self.dropout(x)
        return x
    
class SelfAttention(nn.Module):

    """
    Class for computing self attention 

    Parameters:
        embed_dim (int)        : Embedding dimension
        n_attention_heads (int): Number of attention heads to use for performing MultiHeadAttention
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output after Self-Attention Module of shape B, S, E
    """ 
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.n_attention_heads * self.head_embed_dim)
        self.keys = nn.Linear(self.embed_dim, self.n_attention_heads * self.head_embed_dim)
        self.values = nn.Linear(self.embed_dim, self.n_attention_heads * self.head_embed_dim)

        self.out_projection = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)

    def forward(self, x):
        b, s, e = x.shape
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      -->  B, Q, (H*HE)  -->  B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  -->  B, H, Q, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      -->  B, K, (H*HE)  -->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  -->  B, H, K, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      -->  B, V, (H*HE)  -->  B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  -->  B, H, V, HE

        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  -->  B, H, HE, K
        x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE  *   B, H, HE, K   -->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  -->  A, B, C, F   if D==E)

        x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

        x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

        x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  -->  B, H, Q, HE     Compute Attention product with Values

        # Format the output
        x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE --> B, Q, H, HE
        x = x.reshape(b, s, e)                                                              # B, Q, H, HE --> B, Q, (H*HE)

        x = self.out_projection(x)                                                          # B, Q,(H*HE) --> B, Q, E
        return x
    
def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1

class SineKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        super(SineKANLayer,self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052
        
        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)
            
        if is_first:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim  / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim  / self.grid_norm_factor)

        grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        # self.phase = torch.nn.Parameter(phase)
        self.register_buffer('phase', phase)
        
        if self.add_bias:
            self.bias  = torch.nn.Parameter(torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))

        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y

class Encoder(nn.Module):
    """
    Class for creating an encoder layer

    Parameters:
        embed_dim (int)         : Embedding dimension
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout (float)         : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """    
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))                                # Skip connections
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))           # Skip connections
        return x
    
class Encoder_SineKAN(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.skan = SineKANLayer(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))                                # Skip connections
        x = x + self.dropout2(self.skan(self.activation(self.fc1(self.norm2(x)))))           # Skip connections
        return x
    
class Classifier(nn.Module):
    """
    Classification module of the Vision Transformer. Uses the embedding of the classification token to generate logits.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, CL
    """    

    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # New architectures skip fc1 and activations and directly apply fc2.
        self.fc1        = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2        = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]              # B, S, E --> B, E         
        x = self.fc1(x)             # B, E    --> B, E
        x = self.activation(x)      # B, E    --> B, E    
        x = self.fc2(x)             # B, E    --> B, CL
        return x
        

class VisionSKANformer(nn.Module):
    """
    Vision Transformer Class.

    Parameters:
        n_channels (int)        : Number of channels of the input image
        embed_dim  (int)        : Embedding dimension
        n_layers   (int)        : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        image_size (int)        : Image size
        patch_size (int)        : Patch size
        n_classes (int)         : Number of classes
        dropout  (float)        : dropout value
    
    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """    
    def __init__(self, n_channels = 3, embed_dim = 64, n_layers = 4, n_attention_heads = 8, forward_mul = 4, image_size = 32, patch_size = 4, n_classes = 10, dropout=0.1):
        super().__init__()
        self.embedding  = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout) for _ in range(n_layers - 1)])
        self.skan_encoder = Encoder_SineKAN(embed_dim, n_attention_heads, forward_mul, dropout)
        self.encoder.append(self.skan_encoder)
        self.norm = nn.LayerNorm(embed_dim)                                         
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)                                                

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x
    
def vit_init_weights(m): 
    """
    function for initializing the weights of the Vision Transformer.
    """    

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)