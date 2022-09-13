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

def main():

	ops = options()

	files = handleInput(ops.inFile)

	# compute efficiencies
	eff = []
	for file in files: 
		# get dsid
		dsid = int(os.path.basename(file).split(".")[2])
		# load file and gluinos
		x = h5py.File(file, "r")
		# determine quark list
		qs = [i for i in list(x['g1'].keys()) if 'q' in i]
		g1 = np.stack([x[f'g1/{i}'] for i in qs],-1)
		g2 = np.stack([x[f'g2/{i}'] for i in qs],-1)
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
	print(eff.shape)

	# save to file
	outFileName = os.path.join(ops.outDir, f"eff_2x{len(qs)}.h5")
	with h5py.File(outFileName, 'w') as hf:
		hf.create_dataset('eff', data=eff)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--inFile", default=None, help="Input file")
    parser.add_argument("-o", "--outDir", default="./", help="Output directory")
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

if __name__ == "__main__":
	main()