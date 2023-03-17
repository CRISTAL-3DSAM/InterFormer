
from Process import *
import argparse
from Models import get_model
from Beam import beam_search
import matplotlib.pyplot as plt
import Load
import torch
import numpy as np
import imageio
import Load_test
import math
from celluloid import Camera
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
import time
from scipy.io import loadmat
from scipy.linalg import sqrtm
import Load_recursive
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import TSNE
from collections import Counter
import copy
import shutil


def plot_skeleton(skeleton, frame, color,ax):
    ax.plot(skeleton[0:15,frame ], -skeleton[15:30,frame], skeleton[30:45,frame], color + 'o')

    body = np.asarray([12, 14, 13]) #for head and body
    ax.plot(skeleton[body, frame], -skeleton[body+15, frame],skeleton[body+30, frame], color + '-')
    arm_left =  np.asarray([14, 10, 8, 6])
    arm_rigth =  np.asarray([14,11,9,7])
    ax.plot(skeleton[arm_left, frame],-skeleton[arm_left+15, frame],skeleton[arm_left+30, frame], color + '-', alpha=0.3)
    ax.plot(skeleton[arm_rigth, frame],-skeleton[arm_rigth+15, frame],skeleton[arm_rigth+30, frame], color + '-')
    leg_left =  np.asarray([13,4,2,0])
    leg_right =  np.asarray([13,5,3,1])
    ax.plot(skeleton[leg_left, frame],-skeleton[leg_left+15, frame],skeleton[leg_left+30, frame], color + '-', alpha=0.3)
    ax.plot(skeleton[leg_right, frame],-skeleton[leg_right+15, frame],skeleton[leg_right+30, frame], color + '-')


def show_skeleton(reaction,action,GT,nb_seq,label):#,#L2_norm,MPJPE_norm,SME_norm,AME_norm,FID_norm,GT_smooth,reaction_smooth,diversity,MPJPE_average,MPJPE_classes,SME_average,SME_classes,FID_class,labels):
    images = []
    fig = plt.figure(figsize = (12,8))
    plt.ioff()
    fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(label,fontsize=24)
    for i in range(1,reaction.shape[1]-1):
        for k in range(0,len(axs)):
            ax= axs[k]
            angle = 0
            ax.clear()
            if k == 0:
                plot_skeleton(GT, i, 'g',ax)
                ax.set_title('Ground Truth')
            else:
                plot_skeleton(reaction, i, 'r',ax)
                ax.set_title('Generated')
            plot_skeleton(action, i, 'b',ax)
            ax.axis('off')
            ax.set_xlim3d([-3, 2])
            ax.set_ylim3d([-2, 2])
            ax.set_zlim3d([0, 3])
            ax.view_init(-90,-90)
            plt.axis('off')

        if not os.path.exists(cwd+'/visual/visual_tmp'):
            os.makedirs(cwd+'/visual/visual_tmp')
        name =cwd+'/visual/visual_tmp/visual_' + str(i) + '.png'
        plt.savefig(name)
        images.append(imageio.imread(name))
        if np.array_equal(reaction[:,i],GT[:,i]) and i>1:
            break
    imageio.mimsave(cwd+'/visual/movie_'+str(nb_seq)+'.gif', images)
    shutil.rmtree(cwd+'/visual/visual_tmp')
    plt.close('all')


def get_reaction(sentence, init_pos, model, opt):

    model.eval()
    sentence=torch.from_numpy(sentence)
    if opt.device == 0:
        sentence = sentence.cuda()
    sentence = sentence.float()
    sentence = torch.transpose(sentence,1,2)
    init_pos = torch.from_numpy(init_pos)
    init_pos = init_pos.float()
    reaction,att_enc,att_dec = beam_search(sentence.cuda(), model,init_pos, opt)
    torch.cuda.empty_cache()
    gr = torch.greater_equal(reaction,0.91)
    lw = torch.less_equal(reaction,1.09)
    to = torch.logical_and(gr,lw)
    for b in range (reaction.shape[0]):
        for t in range(reaction.shape[1]):
            t_ten = torch.count_nonzero(to[b,t,:])
            if torch.greater(t_ten,40):
                reaction[b,t,:] = torch.ones(reaction.shape[2])

    return reaction.cpu().detach().numpy(),att_enc,att_dec

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response


def class_name(nb):
    if nb=='cha-cha':
        name='01'
    elif nb=='jive':
        name='02'
    elif nb=='rumba':
        name='03'
    elif nb=='salsa':
        name='04'
    elif nb=='samba':
        name='05'
    else:
        name='error'
    return name

def create_SBU_files(action,generated,labels):
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    shutil.rmtree(cwd+'/data/sbu/SBU-Kinect-Interaction/s00s00/')
    count_cl = [0,0,0,0,0]
    for b in range(generated.shape[0]):
        cl_name = labels[b]
        cl_name = class_name(cl_name)
        count_cl[int(cl_name[1])-1] +=1
        path = cwd+'/data/sbu/SBU-Kinect-Interaction/s00s00/'+cl_name+'/'+str(count_cl[int(cl_name[1])-1]).zfill(3)
        os.makedirs(path)
        ACT=action[b]
        GEN=generated[b]
        with open(path+'/skeleton_pos.txt', 'w+') as f:
            for i in range(1,generated.shape[1]):
                if np.sum(ACT[i,:])==45.0:
                    break
                gen = GEN[i,:]
                act = ACT[i,:]
                act_block = ''
                gen_block =''
                for j in range(int(generated.shape[2]/3)):
                    act_block=act_block+str(act[j])+','+str(act[j+15])+','+str(act[j+30])+','
                    if j!=int(generated.shape[2]/3)-1:
                        gen_block=gen_block+str(gen[j])+','+str(gen[j+15])+','+str(gen[j+30])+','
                    else:
                        gen_block=gen_block+str(gen[j])+','+str(gen[j+15])+','+str(gen[j+30])

                if i!=generated.shape[1]-1:
                    full_block = str(i+1)+','+gen_block+'\n'
                else:
                    full_block = str(i+1)+','+gen_block
                f.write(full_block)




def get_full_label(lab):
    if lab=='cha':
        label='cha-cha'
    elif lab=='jiv':
        label='jive'
    elif lab=='rum':
        label='rumba'
    elif lab=='sal':
        label='salsa'
    elif lab=='sam':
        label='samba'
    else:
        label='error'

    return label

def show_skeleton_single(reaction,action,nb_seq,label):
    images = []
    fig = plt.figure(figsize = (12,8))
    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=1,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(label,fontsize=24)
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    for i in range(2,reaction.shape[1]):
        ax.clear()
        plot_skeleton(reaction, i, 'r',ax)
        plot_skeleton(action, i, 'b',ax)

        ax.axis('off')

        ax.set_xlim3d([-3, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([0, 3])
        ax.view_init(-90,-90)
        plt.axis('off')

        if not os.path.exists(cwd+'/visual_single/visual_tmp'):
            os.makedirs(cwd+'/visual_single/visual_tmp')
        name =cwd+'/visual_single/visual_tmp/visual_' + str(i) + '.png'
        plt.savefig(name)
        images.append(imageio.imread(name))
        if np.sum(action[:,i])==45.0 and i>1:
            break
    imageio.mimsave(cwd+'/visual_single/movie_'+str(nb_seq)+'.gif', images)
    shutil.rmtree(cwd+'/visual_single/visual_tmp')
    plt.close('all')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-data_dir', type=str,default='data')
    parser.add_argument('-test_file', type=str,default='data/Test.txt')
    parser.add_argument('-nb_joints', type=int, default=15)
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-max_len', type=int, default=52)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=3)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-denorm', action='store_true',default=False)




    opt = parser.parse_args()
    opt.is_test = True
    opt.links  =np.asarray([[1,2],[2,4],[4,5],[5,6],[2,7],[7,8],[8,9],[2,3],[3,10],[10,11],[11,12],[3,13],[13,14],[14,15]])-1
    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.max_len > 10

    #opt.data_input,opt.data_output,opt.Labels,opt.max_frames = Load_recursive.Load_test(opt.test_file,opt.data_dir,opt.nb_joints) # modifier taille max harcoder
    opt.data_input,opt.data_output,opt.Labels,opt.max_frames = Load_test.Load_Data(opt.test_file,opt.data_dir,opt.nb_joints) # modifier taille max harcoder
    opt.src_pad = 1 # 1 is EOS and PAD
    opt.trg_pad = 1
    model = get_model(opt)


    sentences = opt.data_input

    nb_batch = sentences.shape[0]//opt.batch_size
    full_batch=False
    if ((sentences.shape[0]%opt.batch_size)==0):
        full_batch=True
    else:
        nb_batch=nb_batch+1

    generated= np.zeros([sentences.shape[0],sentences.shape[2],sentences.shape[1]])



    for i in range(nb_batch):
        if not full_batch and i==nb_batch-1:
            sentence = sentences[i*opt.batch_size:,:,:]
            init_pos = opt.data_output[i*opt.batch_size:,:,:]
        else:
            sentence = sentences[i*opt.batch_size:(i+1)*opt.batch_size,:,:]
            init_pos = opt.data_output[i*opt.batch_size:(i+1)*opt.batch_size,:,:]


        gen,encoder_attention,decoder_attention = get_reaction(sentence,init_pos,model, opt)
        if not full_batch and i==nb_batch-1:
            generated[i*opt.batch_size:,:,:] = gen
        else:
            generated[i*opt.batch_size:(i+1)*opt.batch_size,:,:]=gen


    create_SBU_files(np.transpose(opt.data_input,(0,2,1)),generated,opt.Labels)
    classes = ['cha','jiv','rum','sal','sam']
    classes_nb = [0,0,0,0,0]
    for b in range(len(generated)):
        global reaction
        reaction = np.transpose(generated[b])
        action = opt.data_input[b,:,:]
        GT = opt.data_output[b,:,:]
        lab = opt.Labels[b]
        label = get_full_label(lab)
        idx = classes.index(lab)
        if np.sum(action[:,1])==45.0:
            continue
        show_skeleton_user(reaction,action,False,classes_nb[idx],label)
        classes_nb[idx]+=1

        show_skeleton(reaction,action,GT,b,label)




if __name__ == '__main__':
    main()
