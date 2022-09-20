import subprocess
import argparse

parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--full', action="store_true", help="Run on full file list")
parser.add_argument('--dry', action="store_true", help="Dry run.")
parser.add_argument("-j", "--ncpu", default=1, help="Number of cores to use for multiprocessing.", type=int)
parser.add_argument('--signalModel', default='2x3', type=str, help="Signal model (2x3 or 2x5)")
ops = parser.parse_args()

inFile = ""
if ops.signalModel == "2x5":
    inFile = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/GG_rpv_viaN1/PROD1/user.abadea.mc16_13TeV.512858.MGPy8EG_A14NNPDF23LO_GG_rpv_UDB_2200_1400_viaN1.e8448_s3126_r10724_p5083.PROD1_trees.root/user.abadea.512858.e8448_e7400_s3126_r10724_r10726_p5083.29627300._000001.trees.root"
    if ops.full:
        inFile = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/GG_rpv_viaN1/PROD1/user.abadea.mc16_13TeV.512*/*.root"
elif ops.signalModel == "2x3":
    inFile = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/GG_rpv/PROD2/user.abadea.mc16_13TeV.504518.MGPy8EG_A14NNPDF23LO_GG_rpv_UDB_1400_squarks.e8258_s3126_r10724_p5083.PROD2_trees.root/user.abadea.504518.e8258_e7400_s3126_r10724_r10726_p5083.29777484._000001.trees.root"
    if ops.full:
        inFile = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/GG_rpv/PROD2/user.abadea.mc16_13TeV.5*/*.root"

for minJetPt in range(20,50,5):
    for maxNjets in range(11,17):
        cmd = f"python CreateH5files.py -i '{inFile}' -v 01 --minJetPt {minJetPt} --maxNjets {maxNjets} --minNjets 10 --signalModel {ops.signalModel} -o ./Outputs/{ops.signalModel}/v01/minJetPt{minJetPt}_minNjets10_maxNjets{maxNjets}/ --doOverwrite -j {ops.ncpu}"
        print(cmd)

        if not ops.dry:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()
            print("Success" if process.returncode == 0 else "Failed")
