import subprocess
import multiprocessing as mp
import argparse


def main():

    ops = options()

    confs = []
    for minJetPt in range(20,50,5):
        for maxNjets in range(11,17):
            confs.append({"minJetPt":minJetPt,"maxNjets":maxNjets})

    # launch jobs
    if ops.ncpu == 1:
        for conf in confs:
            process(conf)
    else:
        results = mp.Pool(ops.ncpu).map(process, confs)

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--full', action="store_true", help="Run on full file list")
    parser.add_argument('--dry', action="store_true", help="Dry run.")
    parser.add_argument("-j", "--ncpu", default=1, help="Number of cores to use for multiprocessing.", type=int)
    parser.add_argument('--signalModel', default='2x3', type=str, help="Signal model (2x3 or 2x5)")
    return parser.parse_args()

def process(c):
    ops = options()

    minJetPt, maxNjets = c["minJetPt"], c["maxNjets"]
    inDir = f"Outputs/{ops.signalModel}/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}"
    cmd = f"python matchingEff.py -i '{inDir}/user.abadea.5*.h5' -o {inDir} --minJetPt {minJetPt} --minNjets 10 --maxNjets {maxNjets}"
    print(cmd)
    if not ops.dry:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print("Success" if process.returncode == 0 else "Failed")
    
    d = "mc16e_GG_rpv_viaN1.list" if ops.signalModel == "2x5" else "mc16e_GG_rpv.list"
    cmd = f"python matchingEff.py -i {inDir}/eff_{ops.signalModel}_minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}.h5 -d /afs/cern.ch/work/a/abadea/analysis/FactoryTools/src/FactoryTools/FactoryTools/data/RPVMultijet/samples/PHYS/{d} -o /eos/home-a/abadea/analysis/rpvmj/plots/matchingEfficiency/{ops.signalModel}/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets} --minJetPt {minJetPt} --minNjets 10 --maxNjets {maxNjets} -p"
    print(cmd)
    if not ops.dry:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print("Success" if process.returncode == 0 else "Failed")

if __name__ == "__main__":
    main()
