'''
Author: Anthony Badea
Date: September 13, 2022
Purpose: Compute matching efficiencies using unmatched filler value of -1 from outputs of CreateH5files.py
'''

import h5py
import numpy as np
import argparse
import os
import glob
import matplotlib.pyplot as plt

def main():

    ops = options()

    if ops.plot:
        if "viaN1" in ops.dsidList:
            plot2x5(ops.dsidList, ops.inFile, ops.outDir)
        else:
            plot2x3(ops.dsidList, ops.inFile, ops.outDir)
        return

    files = handleInput(ops.inFile)

    # compute efficiencies
    eff = []
    m = []
    for file in files: 
        # get dsid
        dsid = int(os.path.basename(file).split(".")[2])
        # load file and gluinos
        x = h5py.File(file, "r")
        # determine quark list
        qs = [i for i in list(x['g1'].keys()) if 'q' in i]
        g1 = np.stack([x[f'g1/{i}'] for i in qs],-1)
        g2 = np.stack([x[f'g2/{i}'] for i in qs],-1)
        
        # compute masses
        p = np.stack([x[f'source/{i}'] for i in ['eta','mass','phi','pt']],-1)
        pz = p[:,:,3] * np.sinh(p[:,:,0])
        py = p[:,:,3] * np.sin(p[:,:,2])
        px = p[:,:,3] * np.cos(p[:,:,2])
        e = np.sqrt(p[:,:,1]**2 + px**2 + py**2 + pz**2)
        p = np.stack([e,px,py,pz],-1)
        g1p = np.take_along_axis(p,np.expand_dims(g1,-1).repeat(4,-1),1)
        mask = (np.expand_dims(g1,-1).repeat(4,-1) == -1)
        g1p[mask] = 0
        g2p = np.take_along_axis(p,np.expand_dims(g2,-1).repeat(4,-1),1)
        mask = (np.expand_dims(g2,-1).repeat(4,-1) == -1)
        g2p[mask] = 0
        gp = np.stack([g1p,g2p],1).sum(2)
        gm = np.sqrt(gp[:,:,0]**2 - gp[:,:,1]**2 - gp[:,:,2]**2 - gp[:,:,3]**2)
        m.append(gm)

        # plt.hist(gm.flatten(),bins=np.linspace(0,4000,50),histtype="step");plt.show()

        # -1 indicates a missing match
        g1 = (g1==-1).sum(-1)
        g2 = (g2==-1).sum(-1)
        # compute efficiencies
        full = ((g1+g2) == 0).sum() / g1.shape[0]
        partial = np.logical_or(np.logical_and(g1==0, g2!=0), np.logical_and(g1!=0, g2==0)).sum() / g1.shape[0]
        none = np.logical_and(g1!=0, g2!=0).sum() / g1.shape[0]
        print(f"{dsid}: Full ({full:.3f}), Partial ({partial:.3f}), None ({none:.3f})")
        eff.append([dsid, full, partial, none])

    # convert to matrix
    eff = np.array(eff)
    # save masses
    n = max([i.shape[0] for i in m])
    m = [np.pad(i, [(n-i.shape[0],0),(0,0)]) for i in m]
    m = np.stack(m,0)
    print(eff.shape)
    print(m.shape)

    # save to file
    outFileName = os.path.join(ops.outDir, f"eff_2x{len(qs)}.h5")
    with h5py.File(outFileName, 'w') as hf:
        hf.create_dataset('eff', data=eff)
        hf.create_dataset('masses', data=m)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFile", default=None, help="Input file")
    parser.add_argument("-o", "--outDir", default="./", help="Output directory")
    parser.add_argument("-d", "--dsidList", default=None, help="List of files with dsid's")
    parser.add_argument("-p", "--plot", action="store_true")
    return parser.parse_args()

def handleInput(data):
    if os.path.isfile(data) and ".h5" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data,"r")])
    elif os.path.isdir(data):
        return sorted([os.path.join(data,f) for f in os.listdir(data)])
    elif "*" in data:
        return sorted(glob.glob(data))
    return []

def plot2x5(dsidList, effFile, outDir):

    # parse dsid lists
    with open(dsidList, "r") as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        dsids = {}
        for line in lines:
            split = line.split(".")
            dsid = int(split[1])
            gluino_mass = int(split[2].split("_")[5])
            neutralino_mass = int(split[2].split("_")[6])
            dsids[dsid] = [gluino_mass, neutralino_mass]

    # load efficiencies file
    with h5py.File(effFile, "r") as f:
        effData = np.array(f["eff"])

    udb = effData[effData[:,0] < 512876]
    uds = effData[effData[:,0] >= 512876]
    
    plt.rcParams["figure.figsize"] = (12,9)

    # make plot
    for eff, dname in [(udb, "UDB"), (uds, "UDS")]:

        print(f"Looking at {dname} with {eff.shape[0]} samples.")

        # make grid
        x, y = [], []
        for i in eff:
            dsid = i[0]
            # fill x (gluino mass) and y (neutralino mass)
            x.append(dsids[dsid][0])
            y.append(dsids[dsid][1])

        # make arrays
        x = np.array(x)
        y = np.array(y)

        # make grid
        x_bins = [ 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500]
        y_bins = np.linspace(0,2400,49) - 25

        for iE, name in [(1,"Full"), (2,"Partial"), (3, "None"), ([1,2],"Full+Partial")]:
            low = 1e-6
            # print(eff[:,iE].flatten())
            if isinstance(iE, int):
                weights = eff[:,iE].flatten() + low
            else:
                weights = eff[:,iE].sum(-1).flatten() + low
            hist, xbins, ybins, im = plt.hist2d(x, y, bins=[x_bins,y_bins], weights=weights, cmap=plt.cm.Blues, vmin=0, vmax=1)

            # add text to used bins
            for i in range(len(ybins)-1):
                for j in range(len(xbins)-1):
                    if hist.T[i,j] > low:
                        # print(xbins[j]+0.5, ybins[i]+0.5, f"{hist.T[i,j]:.2f}")
                        color = "black" # if hist.T[i,j] < 0.75 else "white"
                        plt.text(xbins[j]+100, ybins[i]+23, f"{hist.T[i,j]:.4f}", color=color, ha="center", va="center", fontweight="bold", fontsize=11)

            plt.xlabel(r"$\tilde{g}$ mass [TeV]", fontsize=18, labelpad=10)
            plt.ylabel(r"$\chi$ mass [TeV]", fontsize=18, labelpad=10)

            plt.xticks(xbins+100, [(i+100)/1000 for i in xbins], fontsize=14)
            plt.yticks(list(sorted(set(y))), [i/1000 for i in list(sorted(set(y)))], fontsize=12)

            # show
            cbar = plt.colorbar()
            cbar.set_label(f"{name} Gluino Matching Efficiency [{dname}]", rotation=270, labelpad=30, fontsize=18)
            cbar.ax.tick_params(labelsize=14)

            plt.grid()

            # add selection text
            plt.text(xbins[1]+100, ybins[-4], r"Jet $\mathrm{p}_{\mathrm{T}}$ > 20 GeV", color="black", ha="center", va="center", fontsize=15)
            plt.text(xbins[1]+100, ybins[-4] - 100, r"10 < NJets $\leq$ 15", color="black", ha="center", va="center", fontsize=15)

            outFileName = os.path.join(outDir, f"eff_2x5_{dname}_{name}.pdf")
            plt.savefig(outFileName, bbox_inches='tight')
            plt.clf()

    if "/eos/home-a/abadea/" in outDir:
        print(f"Saved to https://cernbox.cern.ch/index.php/apps/files/?dir={outDir.split('/eos/home-a/abadea')[-1]}&")

def plot2x3(dsidList, effFile, outDir):
    print("Fill me in")

if __name__ == "__main__":
    main()