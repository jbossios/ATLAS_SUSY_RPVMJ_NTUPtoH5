import subprocess
import multiprocessing as mp

def main():
    confs = []
    for minJetPt in range(20,50,5):
        for maxNjets in range(11,17):
            confs.append({"minJetPt":minJetPt,"maxNjets":maxNjets})
            # inDir = f"Outputs/2x5/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}"
            # cmd = f"python matchingEff.py -i '{inDir}/user.abadea.512*.h5' -o {inDir} --minJetPt {minJetPt} --minNjets 10 --maxNjets {maxNjets}"
            # # print(cmd)
            # # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            # # process.wait()
            # # print("Success" if process.returncode == 0 else "Failed")
            
            # cmd = f"python matchingEff.py -i {inDir}/eff_2x5_minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}.h5 -d /afs/cern.ch/work/a/abadea/analysis/FactoryTools/src/FactoryTools/FactoryTools/data/RPVMultijet/samples/PHYS/mc16e_GG_rpv_viaN1.list -o /eos/home-a/abadea/analysis/rpvmj/plots/matchingEfficiency/2x5/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets} --minJetPt {minJetPt} --minNjets 10 --maxNjets {maxNjets} -p"
            # print(cmd)
            # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            # process.wait()
            # print("Success" if process.returncode == 0 else "Failed")

    # launch jobs
    ncpu = 10
    if ncpu == 1:
        for conf in confs:
            process(conf)
    else:
        results = mp.Pool(ncpu).map(process, confs)

def process(c):
    minJetPt, maxNjets = c["minJetPt"], c["maxNjets"]
    inDir = f"Outputs/2x5/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}"
    cmd = f"python matchingEff.py -i '{inDir}/user.abadea.512*.h5' -o {inDir} --minJetPt {minJetPt} --minNjets 10 --maxNjets {maxNjets}"
    # print(cmd)
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # process.wait()
    # print("Success" if process.returncode == 0 else "Failed")
    
    cmd = f"python matchingEff.py -i {inDir}/eff_2x5_minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}.h5 -d /afs/cern.ch/work/a/abadea/analysis/FactoryTools/src/FactoryTools/FactoryTools/data/RPVMultijet/samples/PHYS/mc16e_GG_rpv_viaN1.list -o /eos/home-a/abadea/analysis/rpvmj/plots/matchingEfficiency/2x5/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets} --minJetPt {minJetPt} --minNjets 10 --maxNjets {maxNjets} -p"
    print(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("Success" if process.returncode == 0 else "Failed")

if __name__ == "__main__":
    main()
