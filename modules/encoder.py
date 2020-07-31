import torch.nn as nn
import torchvision

class AttentionUnit(nn.Module):
    """
    Section 3.1.2 Decoder: Attention Unit
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.at_encoder = nn.Linear(encoder_dim, attention_dim)
        self.at_decoder = nn.Linear(decoder_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded, decoded):
        enc = self.at_encoder(encoded)
        dec = self.at_decoder(decoded)
        attend = enc + dec.unsqueeze(1)
        attend = self.relu(attend)
        attend = self.attention(attend).squeeze(2)
        alpha = self.softmax(attend)
        out = (enc * alpha.unsqueeze(2)).sum(dim=1)
        return out, alpha


class Decoder(nn.Module):
    """
    Section 3.1.2 Decoder: Decoder Unit with LSTM and Attention
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=1024, dropout=0.6, embeddings=None, fine_tune_embed=False):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embeddings:
            self.embedding.weight = nn.Parameter(embeddings)
            print("Embeddings Loaded")
        if fine_tune_embed:
            for p in self.embedding.parameters():
                p.requires_grad = True

        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, enc_out):
        mean = enc_out.mean(dim=1)
        h = self.init_h(mean)
        c = self.init_c(mean)
        return h, c
    
    def forward(self, x):
        raise NotImplementedError