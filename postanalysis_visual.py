import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(
    'Visualize the distribution of learned edges between residues.')
parser.add_argument('--num-residues', type=int, default=388,
                    help='Number of residues of the PDB.')
parser.add_argument('--windowsize', type=int, default=10,
                    help='window size')
parser.add_argument('--threshold', type=float, default=0.6,
                    help='threshold for plotting')
parser.add_argument('--dist-threshold', type=int, default=12,
                    help='threshold for shortest distance')
parser.add_argument('--filename', type=str, default='logs/out_probs_train.npy',
                    help='File name of the probs file.')
args = parser.parse_args()


def getEdgeResults(threshold=False):
    a = np.load(args.filename)
    b = a[:, :, 1]
    c = a[:, :, 2]
    d = a[:, :, 3]

    # There are four types of edges, eliminate the first type as the non-edge
    probs = b+c+d
    # For default residue number 77, residueR2 = 77*(77-1)=5852
    residueR2 = args.num_residues*(args.num_residues-1)
    probs = np.reshape(probs, (args.windowsize, residueR2))

    # Calculate the occurence of edges
    edges_train = probs/args.windowsize

    results = np.zeros((residueR2))
    for i in range(args.windowsize):
        results = results+edges_train[i, :]

    if threshold:
        # threshold, default 0.6
        index = results < (args.threshold)
        results[index] = 0

    # Calculate prob for figures
    edges_results = np.zeros((args.num_residues, args.num_residues))
    count = 0
    for i in range(args.num_residues):
        for j in range(args.num_residues):
            if not i == j:
                edges_results[i, j] = results[count]
                count += 1
            else:
                edges_results[i, j] = 0

    return edges_results

def getDomainEdges(edges_results, domainName):

    if domainName == 'dislp1':#'b1':
        startLoc = 1
        endLoc = 12
    elif domainName == 'dislp2':#'b1':
        startLoc = 255
        endLoc = 316
    elif domainName == 'helcomm':#'diml':
        startLoc = 13
        endLoc = 47
    elif domainName == 'comm':#'disl':
        startLoc = 97
        endLoc = 184
    elif domainName == 'accommh': #'zl':
        startLoc = 81
        endLoc = 96
    elif domainName == 'hel1':#'b2':
        startLoc = 185
        endLoc = 220
    elif domainName == 'hel2':#'b2':
        startLoc = 226
        endLoc = 246
    elif domainName == 'b1':#'el':
        startLoc = 48
        endLoc = 80
    elif domainName == 'b2':#'el':
        startLoc = 221
        endLoc = 225
    elif domainName == 'b3':#'el':
        startLoc = 246
        endLoc = 254
    elif domainName == 'b4':#'el':
        startLoc = 317
        endLoc = 374
    elif domainName == 'end':#'b3':
        startLoc = 375
        endLoc = 388


    edges_results_dislp1 = edges_results[:12, startLoc:endLoc]
    edges_results_dislp2 = edges_results[255:316, startLoc:endLoc]
    edges_results_helcomm = edges_results[13:47, startLoc:endLoc]
    edges_results_comm = edges_results[97:184, startLoc:endLoc]
    edges_results_accommh = edges_results[81:96, startLoc:endLoc]
    edges_results_hel1 = edges_results[185:220, startLoc:endLoc]
    edges_results_hel2 = edges_results[226:246, startLoc:endLoc]
    edges_results_b1 = edges_results[48:60, startLoc:endLoc]
    edges_results_b2 = edges_results[221:225, startLoc:endLoc]
    edges_results_b3 = edges_results[246:254, startLoc:endLoc]
    edges_results_b4 = edges_results[317:374, startLoc:endLoc]
    edges_results_end = edges_results[375:-1, startLoc:endLoc]


    edge_num_dislp1 = edges_results_dislp1.sum(axis=0)
    edge_num_dislp2 = edges_results_dislp2.sum(axis=0)
    edge_num_helcomm = edges_results_helcomm.sum(axis=0)
    edge_num_comm = edges_results_comm.sum(axis=0)
    edge_num_accommh = edges_results_accommh.sum(axis=0)
    edge_num_hel1 = edges_results_hel1.sum(axis=0)
    edge_num_hel2 = edges_results_hel2.sum(axis=0)
    edge_num_b1 = edges_results_b1.sum(axis=0)
    edge_num_b2 = edges_results_b2.sum(axis=0)
    edge_num_b3 = edges_results_b3.sum(axis=0)
    edge_num_b4 = edges_results_b4.sum(axis=0)
    edge_num_end = edges_results_end.sum(axis=0)


    if domainName == 'dislp1':
        edge_average_dislp1 = 0
    else:
        edge_average_dislp1 = edge_num_dislp1.sum(axis=0)/(12*(endLoc-startLoc))
    
    if domainName == 'dislp2':
        edge_average_dislp2 = 0
    else:
        edge_average_dislp2 = edge_num_dislp2.sum(axis=0)/(62*(endLoc-startLoc))
    if domainName == 'helcomm':
        edge_average_helcomm = 0
    else:
        edge_average_helcomm = edge_num_helcomm.sum(axis=0)/(35*(endLoc-startLoc))
    if domainName == 'comm':
        edge_average_comm = 0
    else:
        edge_average_comm = edge_num_comm.sum(axis=0)/(88*(endLoc-startLoc))
    if domainName == 'accommh':
        edge_average_accommh = 0
    else:
        edge_average_accommh = edge_num_accommh.sum(axis=0)/(16*(endLoc-startLoc))
    if domainName == 'hel1':
        edge_average_hel1 = 0
    else:
        edge_average_hel1 = edge_num_hel1.sum(axis=0)/(36*(endLoc-startLoc))
    if domainName == 'hel2':
        edge_average_hel2 = 0
    else:
        edge_average_hel2 = edge_num_hel2.sum(axis=0)/(21*(endLoc-startLoc))
    if domainName == 'b1':
        edge_average_b1 = 0
    else:
        edge_average_b1 = edge_num_b1.sum(axis=0)/(33*(endLoc-startLoc))
    if domainName == 'b2':
        edge_average_b2 = 0
    else:
        edge_average_b2 = edge_num_b2.sum(axis=0)/(5*(endLoc-startLoc))
    if domainName == 'b3':
        edge_average_b3 = 0
    else:
        edge_average_b3 = edge_num_b3.sum(axis=0)/(9*(endLoc-startLoc))
    if domainName == 'b4':
        edge_average_b4 = 0
    else:
        edge_average_b4 = edge_num_b4.sum(axis=0)/(58*(endLoc-startLoc))
    if domainName == 'end':
        edge_average_end = 0
    else:
        edge_average_end = edge_num_end.sum(axis=0)/(14*(endLoc-startLoc))

    edges_to_all = np.hstack((edge_average_dislp1, edge_average_dislp2, edge_average_helcomm,
                              edge_average_comm, edge_average_accommh, edge_average_hel1, edge_average_hel2, edge_average_b1, 
                              edge_average_b2, edge_average_b3, edge_average_b4, edge_average_end))


    return edges_to_all

# Load distribution of learned edges
edges_results_visual = getEdgeResults(threshold=True)
pd.DataFrame(edges_results_visual).to_csv('logs/edges_results.png')

# Step 1: Visualize results
ax = sns.heatmap(edges_results_visual, linewidth=0.5,
                 cmap="Blues", vmax=1.0, vmin=0.0)
plt.savefig('logs/probs.png', dpi=600)
# plt.show()
plt.close()

# Step 2: Get domain specific results
# According to the distribution of learned edges between residues, we integrated adjacent residues as blocks for a more straightforward observation of the interactions.
# For example, the residues in SOD1 structure are divided into seven domains (β1, diml, disl, zl, β2, el, β3).
edges_results = getEdgeResults(threshold=False)

dislp1 = getDomainEdges(edges_results, 'dislp1')
dislp2 = getDomainEdges(edges_results, 'dislp2')
helcomm = getDomainEdges(edges_results, 'helcomm')
comm = getDomainEdges(edges_results, 'comm')
accommh = getDomainEdges(edges_results, 'accommh')
hel1 = getDomainEdges(edges_results, 'hel1')
hel2 = getDomainEdges(edges_results, 'hel2')
b1 = getDomainEdges(edges_results, 'b1')
b2 = getDomainEdges(edges_results, 'b2')
b3 = getDomainEdges(edges_results, 'b3')
b4 = getDomainEdges(edges_results, 'b4')
end = getDomainEdges(edges_results, 'end')


edges_results = np.vstack((dislp1, dislp2, helcomm, comm, accommh, hel1, hel2, b1, b2, b3, b4, end))

edges_results_T = edges_results.T
index = edges_results_T < (args.threshold)
edges_results_T[index] = 0

# Visualize
ax = sns.heatmap(edges_results_T, linewidth=1,
                 cmap="Blues", vmax=1.0, vmin=0.0)
#ax.set_ylim([7, 0])
plt.savefig('logs/edges_domain.png', dpi=600)
# plt.show()
plt.close()

