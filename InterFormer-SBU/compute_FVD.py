import tensorflow_gan as tfgan
from scipy.io import loadmat
import numpy as np


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

GT = loadmat('features/features_GT')
GT = GT['features']
OURS = loadmat('features/features_InterFormer')
OURS = OURS['features']
PAPER = loadmat('features/features_VRNN')
PAPER = PAPER['features']
MIXMATCH = loadmat('features/features_MIXMATCH')
MIXMATCH = MIXMATCH['features']

lengths = loadmat('features/sequences_lenghts')
lengths = lengths['lengths']
lengths = np.squeeze(lengths)

FVD_PAPER = calculate_fvd(GT,PAPER)
FVD_OURS = calculate_fvd(GT,OURS)
FVD_MIXMATCH = calculate_fvd(GT,MIXMATCH)

div_GT=calculate_diversity_2(GT)
div_PAPER=calculate_diversity_2(PAPER)
div_OURS=calculate_diversity_2(OURS)
div_MIXMATCH=calculate_diversity_2(MIXMATCH)


print('FVD of VRNN:')
print(FVD_PAPER.numpy())
print('\n')
print('\n')
print('FVD of MIXMATCH:')
print(FVD_MIXMATCH.numpy())
print('\n')
print('FVD of INTERFORMER:')
print(FVD_OURS.numpy())
print('\n')

print('diversity of VRNN:')
print(100*np.abs(div_PAPER-div_GT)/div_GT)
print('\n')
print('diversity  of MIXMATCH:')
print(100*np.abs(div_MIXMATCH-div_GT)/div_GT)
print('\n')
print('diversity  of INTERFORMER:')
print(100*np.abs(div_OURS-div_GT)/div_GT)
print('\n')
