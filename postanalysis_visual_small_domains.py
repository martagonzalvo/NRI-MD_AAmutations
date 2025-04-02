

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

# gets edge_results into csvs and generates .pngs images individually
# usage:
#   python postan_vis_sm_domains.py ----name-files='NAME'

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
parser.add_argument('--name-files', type=str, default='',
                    help='name of .pdb, useful if running for + than 1 pdb')
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


####### FIGURE OUT WHAT WINDOWSIZE IS!
    print('FIGURE OUT WHAT WINDOWSIZE IS!')

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

# Load distribution of learned edges
edges_results_visual = getEdgeResults(threshold=True)



# Step 1: Visualize results
ax = sns.heatmap(edges_results_visual, linewidth=0.5,
                 cmap="Blues", vmax=1.0, vmin=0.0)
plt.savefig('logs/probs_{}.png'.format(args.name_files), dpi=600)
# plt.show()
plt.close()


# Step 2: Get domain specific results
# According to the distribution of learned edges between residues, we integrated adjacent residues as blocks for a more straightforward observation of the interactions.
# For example, the residues in SOD1 structure are divided into seven domains (β1, diml, disl, zl, β2, el, β3).

domain_list = [[1,12],
[13,31],
[32,50],
[51,64],
[65,77],
[78,96],
[97,106],
[107,122],
[123,131],
[132,148],
[149,157],
[158,177],
[178,184],
[185,201],
[202,216],
[217,226],
[227,243],
[244,255],
[256,270],
[271,275],
[276,282],
[283,298],
[299,306],
[307,315],
[316,324],
[325,334],
[335 ,348],
[349,359],
[360,371],
[372,388]]


edges_results = getEdgeResults(threshold=False)
pd.DataFrame(edges_results).to_csv('logs/edges_results_{}.csv'.format(args.name_files))

heatmap_vals = np.zeros(len(domain_list), len(domain_list))
for j in domain_list:
    j1, j2=j
    sizej= j2-j1
    for i in domain_list:
        i1, i2 = i
        sizei = i2-i1
        value = edges_results[i1:i2,j1:j2].sum()/(sizei*sizej)
        heatmap_vals[j,i] = value


edges_results_T = heatmap_vals.T
index = edges_results_T < (args.threshold)
edges_results_T[index] = 0

# Visualize
###### ADD LABELS!!?
ax = sns.heatmap(edges_results_T, linewidth=1,
                 cmap="Blues", vmax=1.0, vmin=0.0)
#ax.set_ylim([7, 0])                    ####### changed this
plt.savefig('logs/edges_domain_{}.png'.format(args.name_files), dpi=600)
# plt.show()
plt.close()

