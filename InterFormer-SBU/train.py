import argparse
import time
import torch
from Models import get_model
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import Load
from torch.utils.tensorboard import SummaryWriter
import os


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

    input_train = opt.data_input
    output_train = opt.data_output

    nb_save = int(opt.epochs/100)

    writer = SummaryWriter()
    distances = get_distance(input_train,output_train,3)#add opt.dim


    for epoch in range(opt.epochs):

        max_normal_epoch = 100000
        total_loss = 0
        total_loss_train = 0
        total_loss_eval = 0
        total_loss_ff=0
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

            ys = trg[:,1:,:]
            opt.optimizer.zero_grad()

            loss = F.mse_loss(preds, ys,reduction='mean')
            loss = torch.mul(loss,100000)
            total_loss_train+=loss.item()

            loss_ff = F.mse_loss(preds[:,1,:]-preds[:,0,:],ys[:,1,:]-ys[:,0,:],reduction='mean')
            loss_ff = torch.mul(loss_ff,100000)
            total_loss_ff+=loss_ff.item()
            loss_ff_print =loss_ff.item()

            current_loss = loss+loss_ff
            current_loss.backward()


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
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  loss_train = %.3f loss_ff =%.3f  loss_eval = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, loss,loss_ff_print, total_loss_eval), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f  loss_train = %.3f loss_ff =%.3f  loss_eval = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss,loss,loss_ff_print, total_loss_eval))

            cwd=os.getcwd()
            cwd=cwd.replace("\\", "/")
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), cwd+'weights/model_weights')
                cptime = time.time()


        avg_loss = total_loss/nb_batch
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f  loss_train = %.3f loss_ff =%.3f  loss_eval = %.3f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss, total_loss_train/nb_batch,total_loss_ff/nb_batch, total_loss_eval))
        total_loss = 0
        if epoch%5000==0:
            dst= 'temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), cwd+f'{dst}/model_weights')
        if epoch==max_normal_epoch:
            dst= 'temp_save'
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), cwd+f'{dst}/model_weights_after_normal_train')
    writer.flush()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str,default='data')
    parser.add_argument('-train_file', type=str,default='data/Train.txt')
    parser.add_argument('-nb_joints', type=int, default=15)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
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
