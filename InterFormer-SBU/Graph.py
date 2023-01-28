import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph():
    num_node = 15
    self_link = [(i, i) for i in range(num_node)]
    inward_ori_index = [(1, 2), (2, 3), (4,2), (5, 4), (6,5), (7, 2), (8, 7),
                        (9, 8), (10, 3), (11, 10), (12, 11), (13, 3), (14, 13),
                        (15, 14)]
    inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
    outward = [(j, i) for (i, j) in inward]
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A