import tensorflow_gan as tfgan
from scipy.io import loadmat
import numpy as np
from scipy.linalg import sqrtm


def calculate_fvd(real_activations,
                  generated_activations):
    """Returns a list of ops that compute metrics as funcs of activations.
    Args:
      real_activations: <float32>[num_samples, embedding_size]
      generated_activations: <float32>[num_samples, embedding_size]
    Returns:
      A scalar that contains the requested FVD.
    """
    return tfgan.eval.frechet_classifier_distance_from_activations(
        real_activations, generated_activations)

def calculate_diversity(features,lengths):
    sequences = []
    start = 0
    for k in lengths:
        sequences.append(features[start:start+k,:])
        start +=k

    distances = []
    for i in range(len(sequences)):
        for j in range(len(sequences)):
            if i==j:
                continue
            A=sequences[i]
            B=sequences[j]
            if A.shape[0]<B.shape[0]:
                B=B[:A.shape[0],:]
            elif A.shape[0]>B.shape[0]:
                A=A[:B.shape[0],:]

            distances.append(np.linalg.norm(A-B))

    DIST = np.mean(distances)
    return DIST


def calculate_diversity_2(features):

    distances = []
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            if i==j:
                continue
            A=features[i,:]
            B=features[j,:]
            distances.append(np.linalg.norm(A-B))

    DIST = np.mean(distances)
    return DIST

GT = loadmat('features/features_MIXMATCH_GT_classif')
GT = GT['features']
OURS = loadmat('features/features_Interformer')
OURS = OURS['features']
PAPER = loadmat('features/features_VRNN_classif')
VRNN = PAPER['features']
MIXMATCH = loadmat('features/features_MIXMATCH_classif')
MIXMATCH = MIXMATCH['features']
PGBIG = loadmat('features/features_PGBIG')
PGBIG = PGBIG['features']
STT = loadmat('features/features_STT')
STT = STT['features']

lengths = loadmat('features/sequences_lenghts')
lengths = lengths['lengths']
lengths = np.squeeze(lengths)

FVD_PAPER = calculate_fvd(GT,PAPER)
FVD_OURS = calculate_fvd(GT,OURS)
FVD_MIXMATCH = calculate_fvd(GT,MIXMATCH)
FVD_VRNN = calculate_fvd(GT,VRNN)
FVD_STT = calculate_fvd(GT,STT)
FVD_PGBIG = calculate_fvd(GT,PGBIG)

div_GT=calculate_diversity_2(GT)
div_VRNN=calculate_diversity_2(VRNN)
div_OURS=calculate_diversity_2(OURS)
div_MIXMATCH=calculate_diversity_2(MIXMATCH)
div_STT=calculate_diversity_2(STT)
div_PGBIG=calculate_diversity_2(PGBIG)



print('FVD of MIXMATCH:')
print(FVD_MIXMATCH.numpy())
print('\n')
print('FVD of VRNN:')
print(FVD_VRNN.numpy())
print('\n')
print('FVD of STT:')
print(FVD_STT.numpy())
print('\n')
print('FVD of PGBIG:')
print(FVD_PGBIG.numpy())
print('\n')
print('FVD of our method:')
print(FVD_OURS.numpy())
print('\n')


print('diversity of VRNN:')
print(100*np.abs(div_VRNN-div_GT)/div_GT)
print('\n')
print('diversity  of MIXMATCH:')
print(100*np.abs(div_MIXMATCH-div_GT)/div_GT)
print('\n')
print('diversity of STT:')
print(100*np.abs(div_STT-div_GT)/div_GT)
print('\n')
print('diversity of PGBIG:')
print(100*np.abs(div_PGBIG-div_GT)/div_GT)
print('\n')
print('diversity  of our method:')
print(100*np.abs(div_OURS-div_GT)/div_GT)
print('\n')

