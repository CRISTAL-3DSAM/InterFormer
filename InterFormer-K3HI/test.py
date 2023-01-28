
import argparse
from Models import get_model
from Beam import beam_search
import matplotlib.pyplot as plt
import torch
import imageio
import Load_test
import os

import numpy as np
import shutil

def plot_skeleton(skeleton, frame, color,ax):
    ax.plot(skeleton[:15, frame], skeleton[30:45, frame], skeleton[15:30, frame], color + 'o')
    # [0, 1, 2] for head and body
    ax.plot(np.append(skeleton[:2, frame],skeleton[8, frame]),np.append(skeleton[30:32, frame],skeleton[38, frame]), np.append(skeleton[15:17, frame],skeleton[23, frame]), color + '-')
    # [1, 3, 4, 5] and [1, 6, 7, 8] for arms
    ax.plot(np.append(np.append(np.append(skeleton[1, frame],skeleton[2, frame]), skeleton[4, frame]), skeleton[6, frame]), np.append(np.append(np.append(skeleton[31, frame],skeleton[32, frame]), skeleton[34, frame]), skeleton[36, frame]), np.append(np.append(np.append(skeleton[16, frame],skeleton[17, frame]), skeleton[19, frame]), skeleton[21, frame]), color + '-', alpha=0.3)
    ax.plot(np.append(np.append(np.append(skeleton[1, frame],skeleton[3, frame]), skeleton[5, frame]), skeleton[7, frame]),np.append(np.append(np.append(skeleton[31, frame],skeleton[33, frame]), skeleton[35, frame]), skeleton[37, frame]), np.append(np.append(np.append(skeleton[16, frame],skeleton[18, frame]), skeleton[20, frame]), skeleton[22, frame]), color + '-')
    # [2, 9, 10, 11] et [2, 12, 13, 14] for legs
    ax.plot(np.append(np.append(np.append(skeleton[8, frame],skeleton[9, frame]), skeleton[11, frame]), skeleton[13, frame]),np.append(np.append(np.append(skeleton[38, frame],skeleton[39, frame]), skeleton[41, frame]), skeleton[43, frame]), np.append(np.append(np.append(skeleton[23, frame],skeleton[24, frame]), skeleton[26, frame]), skeleton[28, frame]), color + '-', alpha=0.3)
    ax.plot(np.append(np.append(np.append(skeleton[8, frame],skeleton[10, frame]), skeleton[12, frame]), skeleton[14, frame]),np.append(np.append(np.append(skeleton[38, frame],skeleton[40, frame]), skeleton[42, frame]), skeleton[44, frame]), np.append(np.append(np.append(skeleton[23, frame],skeleton[25, frame]), skeleton[27, frame]), skeleton[29, frame]), color + '-')

def show_skeleton(reaction,action,GT,nb_seq,label):
    images = []
    fig = plt.figure(figsize = (12,8))
    plt.ioff()
    fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(label,fontsize=24)
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    for i in range(2,reaction.shape[1]):
        for k in range(0,len(axs)):
            ax= axs[k]
            ax.clear()
            if k == 0:
                plot_skeleton(GT, i, 'g',ax)
                ax.set_title('ground truth')
            else:
                plot_skeleton(reaction, i, 'r',ax)
                ax.set_title('generated')
            plot_skeleton(action, i, 'b',ax)

            ax.axis('off')

            ax.set_xlim3d([-0, 2])
            ax.set_ylim3d([-2, 2])
            ax.set_zlim3d([0, 1])
            ax.view_init(180,90)
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

        ax.set_xlim3d([-0, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([0, 1])
        #ax.set_title('generated')
        ax.view_init(180,90)
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

def class_name(nb):
    if nb=='approa':
        name='01'
    elif nb=='depart':
        name='02'
    elif nb=='kickin':
        name='03'
    elif nb=='pushin':
        name='04'
    elif nb=='shakin':
        name='05'
    elif nb=='exchan':
        name='06'
    elif nb=='punchi':
        name='07'
    elif nb=='pointi':
        name='08'
    else:
        name='error'
    return name

def create_SBU_files(action,generated,labels):
    cwd=os.getcwd()
    cwd=cwd.replace("\\", "/")
    shutil.rmtree(cwd+'/data/sbu/SBU-Kinect-Interaction/s00s00/')
    count_cl = [0,0,0,0,0,0,0,0]
    for b in range(generated.shape[0]):
        ACT=action[b]
        GEN=generated[b]
        if np.sum(ACT[1,:])==45.0:
            continue
        cl_name = labels[b]
        cl_name = class_name(cl_name)
        count_cl[int(cl_name[1])-1] +=1
        path = cwd+'/data/sbu/SBU-Kinect-Interaction/s00s00/'+cl_name+'/'+str(count_cl[int(cl_name[1])-1]).zfill(3)
        os.makedirs(path)

        with open(path+'/skeleton_pos.txt', 'w') as f:
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
                act_block =str(i+1)+','+act_block
                # if i!=generated.shape[1]-1:
                #     full_block = act_block+gen_block+'\n'
                # else:
                #     full_block = act_block+gen_block

                if i!=generated.shape[1]-1:
                    full_block = str(i+1)+','+gen_block+'\n'
                else:
                    full_block = str(i+1)+','+gen_block
                f.write(full_block)

def get_full_label(lab):
    if lab=='approa':
        label='approaching'
    elif lab=='depart':
        label='departing'
    elif lab=='kickin':
        label='kicking'
    elif lab=='pushin':
        label='pushing'
    elif lab=='shakin':
        label='shaking'
    elif lab=='exchan':
        label='exchanging'
    elif lab=='punchi':
        label='punching'
    elif lab=='pointi':
        label='pointing'
    else:
        label='error'
    return label


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-data_dir', type=str,default='data')
    parser.add_argument('-test_file', type=str,default='data/Test.txt')
    parser.add_argument('-nb_joints', type=int, default=15)
    parser.add_argument('-dim', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-max_len', type=int, default=106)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=3)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-visual', action='store_true')

    opt = parser.parse_args()
    opt.is_test = True
    opt.links  =np.asarray([[1,2],[2,4],[4,5],[5,6],[2,7],[7,8],[8,9],[2,3],[3,10],[10,11],[11,12],[3,13],[13,14],[14,15]])-1
    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.max_len > 10

    opt.data_input,opt.data_output,opt.Labels,opt.max_frames = Load_test.Load_Data(opt.test_file,opt.data_dir,opt.nb_joints)
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
    print("generating reaction")
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

    if opt.visual:
        print("creating visuals")
        for b in range(len(generated)):
            global reaction
            reaction = np.transpose(generated[b])
            action = opt.data_input[b,:,:]
            GT = opt.data_output[b,:,:]

            classes = ['approa','depart','kickin','pushin','shakin','exchan','punchi','pointi']
            lab = opt.Labels[b]
            idx = classes.index(lab)
            label = get_full_label(lab)
            show_skeleton(reaction,action,GT,b,label)
            show_skeleton_single(reaction,action,b,label)
    print("DONE")

if __name__ == '__main__':
    main()
