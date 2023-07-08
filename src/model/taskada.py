import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from classifier.base import BASE

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class TaskAda(BASE):

    def __init__(self, ebd_dim, args):
        super(TaskAda, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.args = args
        self.scale_cls = 7

        self.reference = nn.Linear(self.ebd_dim, self.args.way, bias=True)
        nn.init.kaiming_normal_(self.reference.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.reference.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def Transformation_Matrix(self, prototype):
        C = prototype
        eps = 1e-6
        R = self.reference.weight

        power_R = ((R * R).sum(dim=1, keepdim=True)).sqrt()
        R = R / (power_R + eps)

        power_C = ((C * C).sum(dim=1, keepdim=True)).sqrt()
        C = C / (power_C + eps)

        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(1, 0)
        return P

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def forward(self, XS, YS, XQ, YQ, query_data=None):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)
        prototype = self._compute_prototype(XS, YS)
        P = self.Transformation_Matrix(prototype)
        weight = P.view(P.size(0), P.size(1), 1)
        prototype_transformed = F.conv1d(prototype.squeeze(0).unsqueeze(2), weight).squeeze(2)
        XQ_transformed = F.conv1d(XQ.squeeze(0).unsqueeze(2), weight).squeeze(2)

        pred = -self._compute_cos(prototype_transformed, XQ_transformed)

        discriminative_loss = 0.0

        for j in range(self.args.way):
            for k in range(self.args.way):
                if j != k:
                    sim = -self._compute_cos(prototype_transformed[j].unsqueeze(0),
                                            prototype_transformed[k].unsqueeze(0))
                    discriminative_loss = discriminative_loss + sim

        loss = F.cross_entropy(pred, YQ) + 0.5 * discriminative_loss

        acc = BASE.compute_acc(pred, YQ)


        if query_data is not None:
            y_hat = torch.argmax(pred, dim=1)
            X_hat = query_data[y_hat != YQ]
            return acc, loss, X_hat

        return acc, loss, discriminative_loss