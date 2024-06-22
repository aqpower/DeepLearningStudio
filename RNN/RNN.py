#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import Dataset, DataLoader


# In[61]:


file_path = "./data/train.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

paragraphs = [para.strip() for para in text.split("<|endoftext|>") if para.strip()]

print(paragraphs[:5])


# In[62]:


char_to_id = {}
id_to_char = {}


# 遍历数据，更新字符映射
chars = sorted(set(text))
char_to_id = {ch: i + 2 for i, ch in enumerate(chars)}
id_to_char = {i + 2: ch for i, ch in enumerate(chars)}

char_to_id["<pad>"] = 0
char_to_id["<eos>"] = 1
id_to_char[0] = "<pad>"
id_to_char[1] = "<eos>"

vocab_size = len(char_to_id)
print("字典大小: {}".format(vocab_size))


# In[63]:


# df["char_id_list"] = df["Comment"].apply(
# lambda text: [char_to_id[char] for char in list(text)] + [char_to_id["<eos>"]]
# )
# df.head()

char_id_lists = []
for item in paragraphs:
    char_ids = [char_to_id[char] for char in item] + [char_to_id["<eos>"]]
    char_id_lists.append(char_ids)

print(char_id_lists[:5])


# In[64]:


batch_size = 32
epochs = 100
embed_dim = 50
hidden_dim = 30
lr = 0.001
grad_clip = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("now using device: ", device)


# In[65]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # x = self.sequences.iloc[index][:-1]
        # y = self.sequences.iloc[index][1:]
        x = self.sequences[index][:-1]
        y = self.sequences[index][1:]
        return x, y


def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    pad_batch_x = torch.nn.utils.rnn.pad_sequence(
        batch_x, batch_first=True, padding_value=char_to_id["<pad>"]
    )

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(
        batch_y, batch_first=True, padding_value=char_to_id["<pad>"]
    )

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens


# In[66]:


# dataset = Dataset(df["char_id_list"])
dataset = Dataset(char_id_lists)


# In[67]:


data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)


# In[68]:


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize the weights
        self.W_xh = np.random.randn(hidden_size, input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size)
        # initialize the hidden state
        self.h = np.zeros((hidden_size, 1))

    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y


# In[69]:


class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=char_to_id["<pad>"],
        )

        self.rnn_layer1 = torch.nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.rnn_layer2 = torch.nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size),
        )

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        batch_x_lens = batch_x_lens.cpu()
        batch_x = torch.nn.utils.rnn.pack_padded_sequence(
            batch_x, batch_x_lens, batch_first=True, enforce_sorted=False
        )

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=50, top_n=5):
        char_list = [char_to_id[start_char]]
        next_char = None

        while len(char_list) < max_len:
            x = torch.LongTensor(char_list).unsqueeze(0)
            x = self.embedding(x)
            _, (ht, _) = self.rnn_layer1(x)
            _, (ht, _) = self.rnn_layer2(ht)
            y = self.linear(ht.squeeze(0))

            # 获取前 top_n 大的字符的索引
            top_n_values, top_n_indices = torch.topk(y, top_n)
            top_n_indices = top_n_indices.cpu().numpy()

            # 随机选择一个索引
            if top_n > 1:
                next_char = np.random.choice(top_n_indices[0])
            else:
                next_char = top_n_indices[0][0]

            if next_char == char_to_id["<eos>"]:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]


# In[70]:


torch.manual_seed(2)
model = CharRNN(vocab_size, embed_dim, hidden_dim)
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=char_to_id["<pad>"], reduction="mean"
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[74]:


def train(model, num_epochs, data_loader, optimizer, criterion, vocab_size, grad_clip=1.0):
    ###################
    # 训练 #
    ###################
    min_loss = np.Inf
    model.train()
    for epoch in range(1, epochs + 1):
        model = model.to(device)
        for batch_idx, (batch_x, batch_y, batch_x_lens, batch_y_lens) in enumerate(data_loader):
            optimizer.zero_grad()

            # 将数据移动到GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # batch_x_lens = batch_x_lens.to(device)
            # batch_y_lens = batch_y_lens.to(device)

            batch_pred_y = model(batch_x, batch_x_lens)

            batch_pred_y = batch_pred_y.view(-1, vocab_size)
            batch_y = batch_y.view(-1)

            loss = criterion(batch_pred_y, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()

            if not batch_idx % 100:
                print(
                    f"Epoch: {epoch:03d}/{num_epochs:03d} | Batch {batch_idx:04d}/{len(data_loader):04d} | Loss: {loss:.4f}"
                )

        torch.save(model.state_dict(), "char_rnn_model.pth")
        # 每个epoch结束后进行生成测试
        with torch.no_grad():
            model.eval()
            model.cpu()
            generated_text = model.generator("月")
            print("".join(generated_text))
            model.train()

        torch.cuda.empty_cache()


train(model, epochs, data_loader, optimizer, criterion, vocab_size)


# In[ ]:


with torch.no_grad():
    for i in range(10):
        print("".join(model.generator("月")))
        print()
    for i in range(10):
        print("".join(model.generator("天")))
        print()
    for i in range(10):
        print("".join(model.generator("人")))
        print()


# In[ ]:


torch.save(model.state_dict(), "char_rnn_model.pth")

