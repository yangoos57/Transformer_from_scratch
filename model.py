import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class selfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        """
        embed_size : input 토큰 개수, 논문에서는 512개로 사용
        heads : multi_head의 개수, 논문에서는 8개 사용
        Self Attention은 특정 단어(query)와 다른 단어(key) 간의 중요도를 파악하는 매커니즘이다.
        """
        super().__init__()
        self.embed_size = embed_size  # 512차원
        self.heads = heads  # 8개
        self.head_dim = embed_size // heads  # 64차원(개별 attention의 차원)
        """
        query는 기준이 되는 token 모음
        key는 문장 내 token 모음
        모든 token이 query로 활용되므로
        모든 token의 개별 token 간 연관성 파악이 가능
        """
        # input feature, output feature
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 64 => 64
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 64 => 64
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 64 => 64
        # Multi-headed attention을 만듬
        # fully connected out
        # input feature = outfut feature
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)  # 64 * 8 => 512

    def forward(self, value, key, query, mask):
        """
        # query, key, value: (문장 개수(n) x 최대 token 개수(=100) x embeding 차원(=512) )
        """

        N_batch = query.shape[0]  # 총 문장 개수
        value_len = value.shape[1]  # token 개수
        key_len = key.shape[1]  # token 개수
        query_len = query.shape[1]  # token 개수

        value = value.reshape(
            N_batch, self.heads, value_len, self.head_dim
        )  # (n x h x value_len x d_k)
        key = key.reshape(
            N_batch, self.heads, key_len, self.head_dim
        )  # (n x h x key_len x d_k)
        query = query.reshape(
            N_batch, self.heads, query_len, self.head_dim
        )  # (n x h x query_len x d_k)

        # Q,K,V 구하기
        V = self.value(value)
        K = self.key(key)
        Q = self.query(query)

        # score = Q dot K^T
        # score = torch.einsum("nqhd,nkhd->nhqk", [query,key])
        score = torch.matmul(Q, K.transpose(-2, -1))
        # query shape : (n x h x query_len x d_k)
        # key shape : (n x h x d_k x key_len)
        # score shape : (n x h x query_len x key_len)
        # Pad 부분을 0 => -inf로 변환하는 과정
        #
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-1e20"))
            """
            mask = 0 인 값에 대해서 -inf 대입
            -1e20 = -inf
            -inf이기 때문에 softmax 계산시 값 0을 부여받음
            """
        # attention 정의
        # parameter dim은 몇번째 값에 softmax를 수행하는지 설정함.
        softmax_score = torch.softmax(score / (self.embed_size ** (1 / 2)), dim=3)
        # out = torch.einsum("nhql,nlhd -> nqhd",[attention, value]).reshape(
        #     N,query_len,self.heads * self.head_dim
        #     )
        # out = torch.matmul(softmax_score,V).reshape(
        #     N,query_len,self.heads * self.head_dim
        #     )
        out = torch.matmul(softmax_score, V).reshape(
            N_batch, query_len, self.heads * self.head_dim
        )

        # softmax_score shape : (n x h x query_len x key_len)
        # value shape : (n x h x value_len x d_k)
        # (value_len과 key_len은 size가 같음.)
        # out shape : (n x h x query_len x d_k)
        # transpose shape : (n x query_len x embed_size)
        # concat all heads
        out = self.fc_out(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        """
        embed_size : token 개수 | 논문 512개
        heads : attention 개수 | 논문 8개
        dropout : 개별 Node를 골고루 학습하기 위한 방법론
        forward_expansion : forward 계산시 차원을 얼마나 늘릴 것인지 결정, 임의로 결정하는 값
                            forward_차원 계산은 forward_expension * embed_size
                            논문에서는 4로 정함. 총 2048차원으로 늘어남.
        """
        super().__init__()
        # Attention 정의
        self.attention = selfAttention(embed_size, heads)
        ### Norm & Feed Forward
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forawrd = nn.Sequential(
            # 차원을 512 -> 2048로 증가
            nn.Linear(embed_size, forward_expansion * embed_size),
            # 차원을 ReLU 연산
            nn.ReLU(),
            # 차원 2048 -> 512로 축소
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    ### Encoder Block 구현
    def forward(self, value, key, query, mask):
        # self Attention
        attention = self.attention(value, key, query, mask)
        # Add & Normalization
        x = self.dropout(self.norm1(attention + query))
        # Feed_Forward
        forward = self.feed_forawrd(x)
        # Add & Normalization
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ) -> None:
        """
        src_vocab_size : input vocab 개수
        num_layers : Encoder block 구현할 개수
        dropout : dropout 비율 0 ~ 1사이
        max_length : 문장 내 최대 token 개수
        """
        super().__init__()
        self.embed_size = embed_size
        self.device = device

        # 시작부분 구현(input + positional_embeding)
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)  # row / col

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)

        # Encoder Layer 구현
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        # dropout = 0 ~ 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _, seq_len = x.size()

        pos_embed = self.pos_embed[:, :seq_len, :]  # 2 -> 3차원으로 늘림
        out = self.dropout(self.word_embedding(x) + pos_embed)
        for layer in self.layers:
            # Q,K,V,mask
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device) -> None:
        """
        embed_size : token 개수 | 논문 512개
        heads : attention 개수 | 논문 8개
        dropout : 개별 Node를 골고루 학습하기 위한 방법론
        forward_expansion : forward 계산시 차원을 얼마나 늘릴 것인지 결정, 임의로 결정하는 값
                            forward_차원 계산은 forward_expension * embed_size
                            논문에서는 4로 정함. 총 2048차원으로 늘어남.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = selfAttention(embed_size, heads=heads)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, target_mask):
        # src_mask : Multi-head attention에서 Pad에 대한 Mask 수행
        # output에 대한 attention 수행
        # target_mask : Masked Multi-head attention에서 Teacher Forcing을 위한 Mask 수행
        attention = self.attention(x, x, x, target_mask)
        # add & Norm
        query = self.dropout(self.norm(attention + x))

        # encoder_decoder attention + feed_forward
        out = self.encoder_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ) -> None:
        """
        trg_vocab_size : input vocab 개수
        embed_size : embedding_size
        num_layers : Encoder block 구현할 개수
        dropout : dropout 비율 0 ~ 1사이
        max_length : 문장 내 최대 token 개수
        """
        super().__init__()
        self.device = device

        # 시작부분 구현(input + positional_embeding)
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)

        # Decoder Layer 구현
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # N, seq_length = x.shape
        # positional embedding
        _, seq_len = x.size()
        pos_embed = self.pos_embed[:, :seq_len, :]
        out = self.dropout(self.word_embedding(x) + pos_embed)
        for layer in self.layers:
            # Decoder Input, Encoder K, Encoder V , src_mask, trg_mask
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)
        return out


class transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ) -> None:
        super().__init__()
        self.Encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.Decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        # Probability Generlator
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N,1,1,src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # trg = triangle
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.Encoder(src, src_mask)
        out = self.Decoder(trg, enc_src, src_mask, trg_mask)
        # Linear Layer
        out = self.fc_out(out)  # num of sentence x max_length x trg_vocab_size

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out
