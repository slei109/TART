import torch
import torch.nn as nn

from embedding.rnn import RNN
import torch.nn.functional as F



class ModelG(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG, self).__init__()

        self.args = args

        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim

        self.rnn = RNN(300, 128, 1, True, 0)

        self.fc = nn.Linear(256, self.args.n_train_class, bias=False)

        self.seq = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, data, return_score=False):
        ebd = self.ebd(data)
        w2v = ebd
        avg_sentence_ebd = torch.mean(w2v, dim=1)

        # Embedding
        ebd_rnn = self.rnn(ebd, data['text_len'])
        avg_sentence_ebd_rnn = torch.mean(ebd_rnn, dim=1)
        ebd_rnn_loss = 0
        ebd = self.seq(ebd_rnn).squeeze(-1)  # [b, text_len, 256] -> [b, text_len]
        word_weight = F.softmax(ebd, dim=-1)
        sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        ebd_final = (torch.unsqueeze(word_weight, dim=-1)) * w2v

        reverse_feature = word_weight

        if reverse_feature.shape[1] < 500:
            zero = torch.zeros((reverse_feature.shape[0], 500-reverse_feature.shape[1]))
            if self.args.cuda != -1:
               zero = zero.cuda(self.args.cuda)
            reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
        else:
            reverse_feature = reverse_feature[:, :500]

        if self.args.ablation == '-IL':
            sentence_ebd = torch.cat((avg_sentence_ebd, sentence_ebd), 1)
            print("%%%%%%%%%%%%%%%%%%%%This is ablation mode: -IL%%%%%%%%%%%%%%%%%%")


        return sentence_ebd, reverse_feature, avg_sentence_ebd, ebd_rnn_loss, ebd_final