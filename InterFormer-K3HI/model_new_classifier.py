import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
import scipy.io
import os

# ----------------------------------------------------------------------------------------------------------------------
class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        # Attention
        self.attention = Attention(128)

        # Classifier
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes)
        # )

        self.classifier_1 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
        )
        self.classifier_2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_padded, x_lengths):
        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, _ = self.gru1(x_packed)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)

        #
        cwd=os.getcwd()
        cwd=cwd.replace("\\", "/")
        scipy.io.savemat(cwd+'/features/sequences_lenghts', dict([('lengths', x_lengths.cpu().detach().numpy())]))
        #scipy.io.savemat('features_ablation_200000_pre_classif', dict([('features', output.data.cpu().detach().numpy())]))
        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        rer = self.classifier_1(attn_output)
        scipy.io.savemat(cwd+'/features/features_InterFormer', dict([('features', rer.data.cpu().detach().numpy())]))
        return self.classifier_2(rer)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(128, 128, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output
