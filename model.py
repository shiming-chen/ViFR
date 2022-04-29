#author: akshitac8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Encoder
class Encoder(nn.Module):

    def __init__(self, opt):

        super(Encoder,self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

#Decoder/Generator
class Generator(nn.Module):

    def __init__(self, opt):

        super(Generator,self).__init__()

        layer_sizes = opt.decoder_layer_sizes
        latent_size=opt.latent_size
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z,c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x

#conditional discriminator for inductive
class Discriminator(nn.Module):
    def __init__(self, opt): 
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h
        
#Feedback Modules
class Feedback(nn.Module):
    def __init__(self,opt):
        super(Feedback, self).__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self,x):
        self.x1 = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(self.x1))
        return h

class Post_FR(nn.Module):
    def __init__(self, opt, attSize):
        super(Post_FR, self).__init__()
        self.embedSz = 0
        self.hidden = None
        self.lantent = None
        self.latensize=opt.latensize
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize*2)
        # self.encoder_linear = nn.Linear(opt.resSize, opt.latensize*2)
        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, feat, train_G=False):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = self.fc3(self.hidden)
        mus,stds = self.lantent[:,:self.attSize],self.lantent[:,self.attSize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        h= encoder_out
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        return mus, stds, dis_out, pred, encoder_out, h
        
    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu


class Pre_FR(nn.Module):  
    def __init__(self, opt, init_w2v_att, att, seenclass, unseenclass):  
        super(Pre_FR, self).__init__()  
        self.config = opt
        self.dim_f = opt.resSize
        self.dim_v = opt.pre_dim_v
        self.dim_att = att.shape[1]  
        self.nclass = att.shape[0]  
        self.hidden = self.dim_att//2
        self.init_w2v_att = init_w2v_att
        device = opt.device
        self.normalize_V = opt.pre_normalize_V
        # init parameters
        self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att))
        self.V = nn.Parameter(self.init_w2v_att.clone().to(device))
        self.att = F.normalize(torch.tensor(att)).to(device)
        # visual-to-semantic mapping
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)).to(device))
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)).to(device))
        # for loss
        self.weight_ce = torch.eye(self.nclass).float().to(device)
        self.seenclass = seenclass  
        self.unseenclass = unseenclass
        self.log_softmax_func = nn.LogSoftmax(dim=1)  
                       
    def compute_aug_cross_entropy(self, in_package):
        batch_label = in_package['batch_label'] 
        
        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label]  
        
        S_pp = in_package['S_pp']  
        
        Labels = batch_label
        
        S_pp = S_pp[:,self.seenclass]  
        Labels = Labels[:,self.seenclass]  
        assert S_pp.size(1) == len(self.seenclass)  
        
        Prob = self.log_softmax_func(S_pp)  
          
        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss  
    
    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]
        loss = self.compute_aug_cross_entropy(in_package)
        return {'loss': loss}
    
    def extract_attention(self, Fs):
        shape = Fs.shape
        Fs = Fs.reshape(shape[0], shape[1], shape[2]*shape[3])
        V_n = F.normalize(self.V) if self.normalize_V else self.V
        Fs = F.normalize(Fs, dim=1)
        A = torch.einsum('iv,vf,bfr->bir', V_n, self.W_2, Fs)
        A = F.softmax(A, dim=-1)
        Hs = torch.einsum('bir,bfr->bif', A, Fs)
        return {'A': A, 'Hs': Hs}

    def compute_attribute_embed(self, Hs):
        V_n = F.normalize(self.V) if self.normalize_V else self.V
        S_p = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, Hs)
        S_pp = torch.einsum('ki,bi->bik', self.att, S_p)
        S_pp = torch.sum(S_pp, axis=1)
        return {'S_pp': S_pp, 'A_p': None}

    def forward(self, Fs):
        package_1 = self.extract_attention(Fs)
        Hs = package_1['Hs']
        package_2 = self.compute_attribute_embed(Hs)
        return {'A': package_1['A'],
                'A_p': package_2['A_p'],
                'S_pp': package_2['S_pp']}
