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
import scipy.stats

def main():

    ops = options()
    
    if not os.path.isdir(ops.outDir):
        os.makedirs(ops.outDir)

    if ops.plot:
        if "viaN1" in ops.dsidList:
            plot2x5(ops.dsidList, ops.inFile, ops.outDir)
            plot2x5masses(ops.dsidList, ops.inFile, ops.outDir, False, "masses")
            plot2x5masses(ops.dsidList, ops.inFile, ops.outDir, True, "masses")
            plot2x5masses(ops.dsidList, ops.inFile, ops.outDir, False, "neutralino_masses")
            plot2x5masses(ops.dsidList, ops.inFile, ops.outDir, True, "neutralino_masses")
        else:
            plot2x3(ops.dsidList, ops.inFile, ops.outDir)

        if "/eos/home-a/abadea/" in ops.outDir:
            print(f"Saved to https://cernbox.cern.ch/index.php/apps/files/?dir={ops.outDir.split('/eos/home-a/abadea')[-1]}&")

        return

    if ops.efficiencyTable:
        return efficiencyTable()
    
    # handle input files
    files = handleInput(ops.inFile)
    print(files)
    # pass in spanet predictions
    if ops.spanet:
        spanet = handleInput(ops.spanet)

    # compute efficiencies
    eff, normweight, m, mmask, neum = [], [], [], [], []
    for iF, file in enumerate(files): 
        # get dsid
        try:
            dsid = int(os.path.basename(file).split(".")[2])
        except: 
            dsid = 1
        # load file and gluinos
        x = h5py.File(file, "r")
        
        # determine quark list
        qs = [i for i in list(x['g1'].keys()) if 'q' in i]
        if ops.spanet:
            with h5py.File(spanet[iF], "r") as hf:
                g1 = np.stack([hf[f'g1/{i}'] for i in qs],-1)
                g2 = np.stack([hf[f'g2/{i}'] for i in qs],-1)
        else:
            g1 = np.stack([x[f'g1/{i}'] for i in qs],-1)
            g2 = np.stack([x[f'g2/{i}'] for i in qs],-1)
        
        # get 4-momentum
        p = np.stack([x[f'source/{i}'] for i in ['eta','mass','phi','pt']],-1)
        pz = p[:,:,3] * np.sinh(p[:,:,0])
        py = p[:,:,3] * np.sin(p[:,:,2])
        px = p[:,:,3] * np.cos(p[:,:,2])
        e = np.sqrt(p[:,:,1]**2 + px**2 + py**2 + pz**2)
        p = np.stack([e,px,py,pz],-1)
        
        # get gluino masses
        g1p = np.take_along_axis(p,np.expand_dims(g1,-1).repeat(4,-1),1)
        mask = (np.expand_dims(g1,-1).repeat(4,-1) == -1)
        g1p[mask] = 0
        g2p = np.take_along_axis(p,np.expand_dims(g2,-1).repeat(4,-1),1)
        mask = (np.expand_dims(g2,-1).repeat(4,-1) == -1)
        g2p[mask] = 0
        gp = np.stack([g1p,g2p],1).sum(2)
        gm = np.sqrt(gp[:,:,0]**2 - gp[:,:,1]**2 - gp[:,:,2]**2 - gp[:,:,3]**2)
        m.append(gm)
        mask = np.stack([(g1!=-1).sum(-1) == len(qs), (g2!=-1).sum(-1) == len(qs)],-1)
        mmask.append(mask)
        
        # if 2x5 get neutralino masses from q3,q4,q5
        if len(qs) == 5:
            neup = np.stack([g1p[:,[2,3,4]],g2p[:,[2,3,4]]],1).sum(2)
            neum.append(np.sqrt(neup[:,:,0]**2 - neup[:,:,1]**2 - neup[:,:,2]**2 - neup[:,:,3]**2))

        # -1 indicates a missing match
        g1 = (g1==-1).sum(-1)
        g2 = (g2==-1).sum(-1)
        # compute efficiencies
        full = ((g1+g2) == 0).sum() / g1.shape[0]
        partial = np.logical_or(np.logical_and(g1==0, g2!=0), np.logical_and(g1!=0, g2==0)).sum() / g1.shape[0]
        none = np.logical_and(g1!=0, g2!=0).sum() / g1.shape[0]
        print(f"{dsid}: Full ({full:.3f}), Partial ({partial:.3f}), None ({none:.3f})")
        eff.append([dsid, full, partial, none])
        
        # get normalization weight
        normweight.append(np.array(x['EventVars/normweight']))

        # close file
        x.close()

    # convert to matrix
    eff = np.array(eff)
    # save masses
    n = max([i.shape[0] for i in m])
    m = np.stack([np.pad(i, [(n-i.shape[0],0),(0,0)]) for i in m], 0)
    mmask = np.stack([np.pad(i, [(n-i.shape[0],0),(0,0)]) for i in mmask], 0)
    normweight = np.stack([np.pad(i, [(n-i.shape[0],0)]) for i in normweight], 0)
    
    # save to file
    outFileName = os.path.join(ops.outDir, f"eff_2x{len(qs)}_minJetPt{ops.minJetPt}_minNjets{ops.minNjets}_maxNjets{ops.maxNjets}.h5")
    print(f"Saving to {outFileName}")
    with h5py.File(outFileName, 'w') as hf:
        hf.create_dataset('eff', data=eff)
        hf.create_dataset('masses', data=m)
        hf.create_dataset('mmask', data=mmask)
        hf.create_dataset('normweight', data=normweight)
        if len(neum):
            neum = np.stack([np.pad(i, [(n-i.shape[0],0),(0,0)]) for i in neum], 0)
            hf.create_dataset('neutralino_masses', data=neum)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFile", default=None, help="Input file")
    parser.add_argument("-o", "--outDir", default="./", help="Output directory")
    parser.add_argument("-d", "--dsidList", default=None, help="List of files with dsid's")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("--efficiencyTable", action="store_true", help="run efficiencyTable function")
    parser.add_argument('--minJetPt', default=50, type=int, help="Minimum selected jet pt")
    parser.add_argument('--maxNjets', default=8, type=int, help="Maximum number of leading jets retained in h5 files")
    parser.add_argument('--minNjets', default=6, type=int, help="Minimum number of leading jets retained in h5 files")
    parser.add_argument('--signalModel', default='2x3', type=str, help="Signal model (2x3 or 2x5)")
    parser.add_argument("-s", "--spanet", default=None, help="Spanet prediction files")
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

    ops = options()

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
            plt.text(xbins[1]+100, ybins[-4], rf"Jet $\mathrm{{p}}_{{\mathrm{{T}}}}$ > {ops.minJetPt} GeV", color="black", ha="center", va="center", fontsize=15)
            plt.text(xbins[1]+100, ybins[-4] - 100, rf"{ops.minNjets} < NJets $\leq$ {ops.maxNjets}", color="black", ha="center", va="center", fontsize=15)

            outFileName = os.path.join(outDir, f"eff_2x5_{dname}_{name}.pdf")
            plt.savefig(outFileName, bbox_inches='tight')
            plt.clf()

def plot2x5masses(dsidList, effFile, outDir, useMask, masses="masses"):
    
    ops = options()

    x = h5py.File(effFile,"r")
    if ops.spanet:
        spanet_eff = h5py.File(ops.spanet, "r")

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
    gm = [i[1][0] for i in dsids.items()]
    mi = [int(gm.count(i)/2) for i in sorted(set(gm))]

    i = 0
    for dname in ["UDB", "UDS"]:
        nRow, nCol = 13, 8
        sRow, sCol = 24, 14
        fig, axs = plt.subplots(nRow, nCol, figsize=(sRow, sCol), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        for col, nRows in enumerate(mi):
            for j in range(len(axs)): #nRows):

                ax = axs[12-j][col]
                # handle xticks
                ax.set_xticks([1,2,3])
                # handle yticks
                ax.set_ylim(0,0.3)
                ax.set_yticks([0.1,0.2])
                ax.minorticks_on()
                ax.tick_params(axis='both', which='major', labelsize=14, direction="in")
                ax.tick_params(axis='both', which='minor', bottom=True, labelsize=8, direction="in")
                
                if j < nRows and i < x[masses].shape[0]:
                    X = x[masses][i][x[masses][i]!=0] / 1000 # in TeV
                    if useMask:
                        X = X[x['mmask'][i][x[masses][i]!=0]]
                    else:
                        X = X.flatten()
                    n, bins, patches = ax.hist( X, bins=np.linspace(0,3,50), histtype="step", density=False, weights=[1./X.shape[0]]*X.shape[0], color="blue")
                    mg, mchi = dsids[int(x['eff'][i][0])]
                    mg, mchi = mg/1000, mchi/1000 
                    eff = x['eff'][i][1] + x['eff'][i][2]
                    ax.text(0.01, 0.25, rf"$m_\tilde{{g}}$ = {mg}, $m_\chi$ = {mchi} TeV, $\epsilon$ = {eff:.3f}", color="black", ha="left", va="center", fontsize=8.5)
                    
                    # repeat for spanet
                    if ops.spanet:
                        X_spanet = spanet_eff[masses][i][spanet_eff[masses][i]!=0] / 1000 # in TeV
                        if useMask:
                            X_spanet = X_spanet[spanet_eff['mmask'][i][spanet_eff[masses][i]!=0]]
                        else:
                            X_spanet = X_spanet.flatten()
                        n, bins, patches = ax.hist( X_spanet, bins=np.linspace(0,3,50), histtype="step", density=False, weights=[1./X_spanet.shape[0]]*X_spanet.shape[0], color="red")
                    
                    # increment i
                    i+=1

        
        xtitle = "Gluino" if masses == "masses" else "Neutralino"
        fig.text(0.5, 0.06, f'Reconstructed {xtitle} Mass [TeV]', ha='center', fontsize=25)
        fig.text(0.09, 0.5, f'Fraction of {xtitle}s [{dname}]', va='center', rotation='vertical', fontsize=25)

        fig.text(0.175, 0.86,  rf"Jet $\mathrm{{p}}_{{\mathrm{{T}}}}$ > {ops.minJetPt} GeV", color="black", ha="center", va="center", fontsize=15)
        fig.text(0.175, 0.84, rf"{ops.minNjets} < NJets $\leq$ {ops.maxNjets}", color="black", ha="center", va="center", fontsize=15)

        outFileName = os.path.join(outDir, f"{masses}_2x5_{dname}_mask{int(useMask)}.pdf")
        plt.savefig(outFileName, bbox_inches='tight')
        plt.clf()


def plot2x3(dsidList, effFile, outDir):
    print("Fill me in")

def efficiencyTable():

    ops = options()
    fileList = handleInput(ops.inFile)

    # parse dsid lists
    with open(ops.dsidList, "r") as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        dsidMasses = {}
        for line in lines:
            split = line.split(".")
            dsid = int(split[1])
            gluino_mass = int(split[2].split("_")[5])
            if ops.signalModel == "2x5":
                neutralino_mass = int(split[2].split("_")[6])
                dsidMasses[dsid] = [gluino_mass, neutralino_mass]
            else:
                dsidMasses[dsid] = [gluino_mass]

    table, sels = [], []
    for iF, f in enumerate(fileList):
        print(f"File {iF} / {len(fileList)}")
        sels.append(os.path.basename(f).split(f"eff_{ops.signalModel}_")[1].split(".h5")[0])
        with h5py.File(f,"r") as hf:
            m = np.array(hf['masses'])
            mmask = np.array(hf['mmask'])
            eff = np.array(hf['eff'])

            # will be easier if you loop over the dsids
            stats = []
            for i in range(m.shape[0]):

                # apply the mask
                mt = m[i][mmask[i]]
                stats.append([
                    mt.mean(),
                    np.median(mt),
                    np.std(mt),
                    np.sqrt(np.mean(mt**2)), # rms
                    scipy.stats.iqr(mt)
                ])

            # stack 
            stats = np.stack(stats)
            dsids = eff[:,0]
            eff[:,1] = eff[:,1] + eff[:,2] # set the first value to full+partial for ease of use
            stats = np.concatenate([eff[:,1:], stats],1) # pre-pend efficiencies without dsid

            table.append(stats)

    # prepare data
    table = np.stack(table,1)
    dt = h5py.special_dtype(vlen=str) 
    dsids = dsids.astype(int)
    sels = np.array(sels, dtype=dt) 
    stats = np.array(['full_plus_partial_matching_eff', 'partial_matching_eff', 'no_matching_eff', 'mean','median','std','rms','iqr'], dtype=dt) 

    # pickup masses
    masses = np.array([dsidMasses[dsid] for dsid in dsids])

    # save to file
    outFileName = os.path.join(ops.outDir, f"eff_grid{ops.signalModel}_table.h5")
    print(f"Saving to {outFileName}")
    with h5py.File(outFileName, 'w') as hf:
        hf.create_dataset('table', data=table)
        hf.create_dataset('dim0_dsids', data=dsids)        
        hf.create_dataset('dim1_selections', data=sels)
        hf.create_dataset('dim2_stats', data=stats)
        hf.create_dataset('true_masses', data=masses)


if __name__ == "__main__":
    main()
    
