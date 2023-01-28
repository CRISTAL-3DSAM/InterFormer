import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math
import numpy as np


def init_vars(src, model,init_pos, opt):
    
    init_tok = np.zeros((opt.nb_joints*3,1))
    src_mask = (src[:,:,0] != opt.src_pad).unsqueeze(1)
    e_output,att_enc = model.encoder(src, src_mask,opt.is_test)
    
    outputs = torch.LongTensor(init_tok).transpose(1,0).unsqueeze(0)
    if opt.device == 0:
        outputs = outputs.cuda()

    out = init_pos

    
    outputs = torch.zeros(src.shape[0],opt.max_len,opt.nb_joints*3,dtype=torch.float32)
    if opt.device == 0:
        outputs = outputs.cuda()
    outputs[:, 0,:] = torch.from_numpy(init_tok).transpose(1,0)
    outputs[:, 1,:] = out[:,:,1]
    
    e_outputs = torch.zeros(src.shape[0],e_output.size(-2),e_output.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :,:] = e_output
    
    return outputs, e_outputs,att_enc


def get_distance(action,reaction,dim):
    action2 = torch.reshape(action,(reaction.shape[0],reaction.shape[1],dim,int(reaction.shape[2]/dim)))
    reaction2 = torch.reshape(reaction,(reaction.shape[0],reaction.shape[1],dim,int(reaction.shape[2]/dim)))
    dist = torch.zeros((reaction.shape[0],reaction.shape[1],dim,int(reaction.shape[2]/dim),int(reaction.shape[2]/dim)))
    for j in range(reaction2.shape[-1]):
        x = action2-reaction2[:,:,:,j].unsqueeze(-1).repeat(1,1,1,15)
        dist[:,:,:,j,:]=x
    dist=torch.norm(dist,dim=2)
    dist_res =torch.reshape(dist,(dist.shape[0],dist.shape[1],dist.shape[2]*dist.shape[2]))
    dist_res = F.softmax(-dist_res, dim=-1)
    dist =torch.reshape(dist_res,(dist.shape[0],dist.shape[1],dist.shape[2],dist.shape[2]))
    return dist


def beam_search(src, model,init_pos, opt):
    

    outputs, e_outputs,att_enc = init_vars(src, model,init_pos, opt)
    eos_tok = np.ones((opt.nb_joints*3,1))
    src_mask = (src[:,:,0] != opt.src_pad).unsqueeze(1)
    ind = None
    att_dec = {}
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)
        dist = get_distance(src[:,:i],outputs[:,:i],3)
        dist = dist.cuda()
        out_tmp, att_dec_tmp = model.decoder(outputs[:,:i],e_outputs,dist, src_mask,trg_mask,opt.is_test)
        out = model.out(out_tmp)
        str_val = 'frame' + str(i)
        att_dec[str_val] = att_dec_tmp

        outputs[:,i,:] = out[:,-1,:]
        del out,out_tmp
    return outputs,att_enc,att_dec
