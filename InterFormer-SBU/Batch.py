import torch
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask[:,0,1]=0 # to take first frame into account
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, opt):
    
    src_mask = (src[:,:,0] != opt.src_pad).unsqueeze(1)

    if trg is not None:
        trg_mask = (trg[:,:,0] != opt.trg_pad)
        trg_mask = torch.unsqueeze(trg_mask,1)
        size = trg.shape[1] # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        #if trg.is_cuda:
        #    np_mask.cuda()
        if opt.device == 0:
            trg_mask = trg_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

