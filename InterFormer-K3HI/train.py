import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
import Load
import Load_test
import numpy as np
import Beam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import imageio

def get_reaction(sentences, model,out_part, opt):

    #reactions = torch.zeros([sentences.shape[0],sentences.shape[2],sentences.shape[1]]).cuda()
    init_pos = torch.from_numpy(out_part)
    init_pos = init_pos.float()
    reaction,_,_ = Beam.beam_search(sentences.cuda(), model,init_pos, opt)
    gr = torch.greater_equal(reaction,0.91)
    lw = torch.less_equal(reaction,1.09)
    to = torch.logical_and(gr,lw)
    for t in range(reaction.shape[1]):
        t_ten = torch.count_nonzero(to[0,t,:])
        if torch.greater(t_ten,40):
            reaction[0,t,:] = torch.ones(reaction.shape[2])

    reaction = torch.transpose(reaction,2,1)
    reaction = torch.squeeze(reaction)

    return reaction


def plot_skeleton(skeleton, frame, color,ax):
    ax.plot(skeleton[:15, frame], skeleton[30:45, frame], skeleton[15:30, frame], color + 'o')
    # [0, 1, 2] for head and body
    ax.plot(skeleton[:3, frame], skeleton[30:33, frame],skeleton[15:18, frame], color + '-')
    # [1, 3, 4, 5] and [1, 6, 7, 8] for arms
    ax.plot(np.append(skeleton[1, frame], skeleton[3:6, frame]), np.append(skeleton[31, frame], skeleton[33:36, frame]), np.append(skeleton[16, frame], skeleton[18:21, frame]), color + '-', alpha=0.3)
    ax.plot(np.append(skeleton[1, frame], skeleton[6:9, frame]), np.append(skeleton[31, frame], skeleton[36:39, frame]), np.append(skeleton[16, frame], skeleton[21:24, frame]), color + '-')
    # [2, 9, 10, 11] et [2, 12, 13, 14] for legs
    ax.plot(np.append(skeleton[2, frame], skeleton[9:12, frame]), np.append(skeleton[32, frame], skeleton[39:42, frame]), np.append(skeleton[17, frame], skeleton[24:27, frame]), color + '-', alpha=0.3)
    ax.plot(np.append(skeleton[2, frame], skeleton[12:15, frame]), np.append(skeleton[32, frame], skeleton[42:45, frame]), np.append(skeleton[17, frame], skeleton[27:30, frame]), color + '-')


def show_skeleton(reaction,GT):
    reaction=reaction.detach().cpu().numpy()
    GT=GT.detach().cpu().numpy()
    #reaction=np.transpose(reaction,(1,0))
    GT=np.transpose(GT,(1,0))
    GT=GT[:,1:]
    images = []
    fig = plt.figure(figsize = (12,8))
    plt.ioff()
    fig, axs = plt.subplots(nrows=1, ncols=2,subplot_kw=dict(projection="3d"))
    fig.set_size_inches(18.5, 10.5)
    for i in range(reaction.shape[1]):
        for k in range(0,len(axs)):
            ax= axs[k]
            angle = 0
            ax.clear()
            if k == 0:
                plot_skeleton(GT, i, 'g',ax)
            else:
                plot_skeleton(reaction, i, 'r',ax)

            ax.axis('off')

            ax.set_xlim3d([-0, 2])
            ax.set_ylim3d([-2, 2])
            ax.set_zlim3d([0, 1])
            ax.view_init(180,90)
            plt.axis('off')

        if not os.path.exists('visual/visual'+str(0)):
            os.makedirs('visual/visual'+str(0))
        name ='visual/visual'+str(0)+'/visual_' + str(i) + '.png'
        plt.savefig(name)
        images.append(imageio.imread(name))
        if np.array_equal(reaction[:,i],GT[:,i]) and i>1:
            break
    imageio.mimsave('visual/movie_'+str(0)+'.gif', images)
    plt.close('all')

def get_distance(action,reaction,dim):
    action = torch.transpose(torch.from_numpy(action[:,:,:-1]),1,2)
    reaction = torch.transpose(torch.from_numpy(reaction[:,:,:-1]),1,2)
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

def train_model(model, opt):
    
    print("training model...")
    model.train()
    start = time.time()
    avg_loss = 100
    if opt.checkpoint > 0:
        cptime = time.time()

    if opt.eval:
        per_eval=0.05
        nb_samples = opt.data_input.shape[0]
        nb_samples_eval = round(nb_samples*per_eval)
        input_train = opt.data_input
        output_train = opt.data_output
    else:
        input_train = opt.data_input
        output_train = opt.data_output

    nb_save = int(opt.epochs/100)
    loss_train_hist = np.zeros(nb_save)
    loss_eval_hist = np.zeros(nb_save)
    nb_hist=0

    writer = SummaryWriter()
    distances = get_distance(input_train,output_train,3)#add opt.dim

    for epoch in range(opt.epochs):

        max_normal_epoch = 2
        total_loss = 0
        total_loss_train = 0
        total_loss_eval = 0
        total_loss_ff=0
        loss_ff_print=0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        nb_batch = input_train.shape[0]//opt.batchsize
        full_batch=False
        if ((input_train.shape[0]%opt.batchsize)==0):
            full_batch=True
        else:
            nb_batch=nb_batch+1

        for i in range(nb_batch):

            if not full_batch and i==nb_batch-1:
                src = input_train[i*opt.batchsize:,:,:]
                trg = output_train[i*opt.batchsize:,:,:]
                dist = distances[i*opt.batchsize:,:,:]
            else:
                src = input_train[i*opt.batchsize:(i+1)*opt.batchsize,:,:]
                trg = output_train[i*opt.batchsize:(i+1)*opt.batchsize,:,:]
                dist = distances[i*opt.batchsize:(i+1)*opt.batchsize,:,:]

            src=torch.from_numpy(src)
            src = src.float()
            trg=torch.from_numpy(trg)
            trg = trg.float()
            src = torch.transpose(src,1,2)
            trg = torch.transpose(trg,1,2)
            trg_input = trg[:,:-1,:]
            if opt.device == 0:
                src=src.cuda()
                trg=trg.cuda()
                trg_input=trg_input.cuda()
                dist=dist.cuda()
            src_mask, trg_mask = create_masks(src, trg_input, opt)



            preds = model(src, trg_input,dist, src_mask, trg_mask,opt.is_test)


            #show_skeleton(preds[0,:,:],trg[0,:,:])
            ys = trg[:,1:,:]
            opt.optimizer.zero_grad()

            loss = F.mse_loss(preds, ys,reduction='mean')
            loss = torch.mul(loss,100000)
            total_loss_train+=loss.item()
            #loss.backward(retain_graph=True)


            loss_ff = F.mse_loss(preds[:,1,:]-preds[:,0,:],ys[:,1,:]-ys[:,0,:],reduction='mean')
            loss_ff = torch.mul(loss_ff,100000)
            total_loss_ff+=loss_ff.item()
            loss_ff_print =loss_ff.item()
            current_loss = loss+loss_ff
            current_loss.backward()

            if opt.diverse:
                preds_2 = model(src, trg_input, src_mask, trg_mask,opt.is_test)
                loss_diverse = -F.mse_loss(preds, preds_2,reduction='mean')*10000
                loss_diverse.backward()


            if epoch>max_normal_epoch and  opt.eval:
                eval_select = np.random.choice(nb_samples,size=nb_samples_eval,replace=False)
                input_eval = opt.data_input[eval_select,:,:]
                input_eval = np.transpose(input_eval,(0,2,1))
                input_eval = torch.from_numpy(input_eval).float()
                output_eval = opt.data_output[eval_select,:,:]
                preds_with_test_eval = get_reaction(input_eval,model,output_eval,opt)
                output_eval_torch = torch.from_numpy(output_eval)
                loss_eval = F.mse_loss(preds_with_test_eval.cpu(), output_eval_torch.float(),reduction='sum')
                loss_eval.backward()
                total_loss_eval = loss_eval.item()
                #show_skeleton(preds_with_test_eval[0,:,:],input_eval[0,:,:])

            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()


            total_loss += loss.item()+total_loss_eval+loss_ff_print

            writer.add_scalar('Loss/train', loss.item(), epoch+i)
            writer.add_scalar('Loss/eval', total_loss_eval, epoch+i)


            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / nb_batch)
                 avg_loss = loss.item()
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  loss_train = %.3f loss_ff =%.3f loss_eval = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, loss,loss_ff_print, total_loss_eval), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  loss_train = %.3f loss_ff =%.3f loss_eval = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss,loss,loss_ff_print, total_loss_eval))

            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()


        avg_loss = total_loss/nb_batch
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f  loss_train = %.3f loss_ff =%.3f loss_eval = %.3f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss, total_loss_train/nb_batch,total_loss_ff/nb_batch, total_loss_eval))
        total_loss = 0
        if epoch%5000==0:
            dst= 'temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
        if epoch==max_normal_epoch:
            dst= 'temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights_after_normal_train')
        if epoch>max_normal_epoch and epoch%100 and  opt.eval:
            dst= 'temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights_eval')
    writer.flush()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str,default='data')
    parser.add_argument('-train_file', type=str,default='data/Train.txt')
    parser.add_argument('-nb_joints', type=int, default=15)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=32. )
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=3)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=128)
    parser.add_argument('-printevery', type=int, default=1)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-eos_dist', type=float, default=0.05)
    parser.add_argument('-eval' , action='store_true',default=False)
    parser.add_argument('-spatial' , action='store_true',default=False)
    parser.add_argument('-diverse' , action='store_true',default=False)
    parser.add_argument('-new_arch' , action='store_true',default=False)
    parser.add_argument('-speed_loss' , action='store_true',default=False)
    parser.add_argument('-multihead_new_arch', action='store_true',default=False)

    parser.add_argument('-max_len', type=int, default=48) # ajouté pour test, à intégrer dans le code

    opt = parser.parse_args()
    opt.is_test = False
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    opt.src_pad = 1 # 1 is EOS and PAD
    opt.trg_pad = 1
    opt.data_input,opt.data_output,opt.max_frames = Load.Load_Data(opt.train_file,opt.data_dir,opt.nb_joints)
    model = get_model(opt)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    #if opt.load_weights is not None and opt.floyd is not None:
    #    os.mkdir('weights')
    
    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            #if saved_once == 0:

            #    saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
