import numpy as np
from scipy.io import loadmat
import os

def load_text_file( file):
    # get path to data file x = input, y= condition , frame = first frame of reaction sequence, gram = path to folder with gram matrix for each frame
    X = []
    y = []
    for line in open(file, 'r'):
        data = line.split()
        X.append(data[0])
        y.append(data[1])
    seed = 2021
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    return X, y

def read_frames(path):
    data_ = loadmat(path)
    data = data_['curve_B']
    return data


def add_sos_eos(data_in,data_out,max_frames,nb_joint):
    SOS = np.zeros((nb_joint*3,1)) # start of sequence pose replace with T pose
    EOS = np.ones((nb_joint*3,1)) # end of sequence pose replace with H pose
    missing_in = max_frames-data_in.shape[1]  # number of padding frames
    missing_out = max_frames-data_out.shape[1]
    data_in = np.concatenate((SOS,data_in,EOS),axis=1)
    data_out = np.concatenate((SOS,data_out,EOS),axis=1)
    pad_in=np.ones((nb_joint*3,missing_in))
    pad_out=np.ones((nb_joint*3,missing_out))
    data_in = np.concatenate((data_in,pad_in),axis=1)
    data_out = np.concatenate((data_out,pad_out),axis=1)

    return data_in,data_out

def Load_Data(path,data_dir,nb_joint):

    Data_in, Data_out = load_text_file(path)
    Data_in = [os.path.join(data_dir, x) for x in Data_in]
    Data_out = [os.path.join(data_dir, y) for y in Data_out]

    seed = 2019
    np.random.seed(seed)
    np.random.shuffle(Data_in)
    np.random.seed(seed)
    np.random.shuffle(Data_out)

    max_frames = 50
    for ind in range(len(Data_in)):
        batch_in = Data_in[ind]
        batch_in = read_frames(batch_in)
        batch_in = np.array(batch_in).astype(np.float32)
        if max_frames<batch_in.shape[1]:
            max_frames=batch_in.shape[1]
    all_data_in = np.zeros((len(Data_in),nb_joint*3,max_frames+2))
    all_data_out = np.zeros((len(Data_in),nb_joint*3,max_frames+2))
    for ind in range(len(Data_in)):
        batch_in = Data_in[ind]
        batch_in = read_frames(batch_in)
        batch_in = np.array(batch_in).astype(np.float32)
        batch_out = Data_out[ind]
        batch_out = read_frames(batch_out)
        batch_out = np.array(batch_out).astype(np.float32)
        data_in, data_out =add_sos_eos(batch_in,batch_out,max_frames,nb_joint)
        all_data_in[ind,:,:] = data_in
        all_data_out[ind,:,:] = data_out
    return all_data_in,all_data_out,max_frames+2


