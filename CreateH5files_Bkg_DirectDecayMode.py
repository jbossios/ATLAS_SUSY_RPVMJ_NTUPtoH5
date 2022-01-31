################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted for background samples                    #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch)                                      #
# Date:    6 Dec 2021                                                          #
#                                                                              #
################################################################################

# FIXME TODO
# Need to save normweight into the H5 files so I can make plots!

import ROOT,copy,resource

#############################
# Helpful classes/functions
#############################

# Pring memory usage
def printMemory(log):
  log.info("PERF: Memory usage = {} (MB)".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

# Just-in-time compile custom helper function to create vector of iJet objects
ROOT.gInterpreter.Declare("""
using namespace ROOT::VecOps;
using Vec  = std::vector<TLorentzVector>;
using VecF = const RVec<float>&;
using VecI = const RVec<int>&;
using Vecf = RVec<float>;
Vec SelectJets(VecF jetPt, VecF jetEta, VecF jetPhi, VecF jetE, VecI jetSig, VecI jetOR, float minPt, float maxNjets){
  Vec Jets;
  unsigned int nJets = jetPt.size();
  if(nJets > maxNjets) nJets = maxNjets;
  for(auto ijet = 0; ijet < nJets; ijet++) {
    if(jetPt[ijet] >= minPt && jetSig[ijet] && jetOR[ijet]) {
      TLorentzVector TLVjet = TLorentzVector();
      TLVjet.SetPtEtaPhiE(jetPt[ijet],jetEta[ijet],jetPhi[ijet],jetE[ijet]);
      Jets.push_back(TLVjet);
    }
  }
  int nMissingJets = maxNjets - Jets.size();
  if(nMissingJets > 0){
    for(unsigned int ijet = 0; ijet < nMissingJets; ijet++){
      TLorentzVector TLVjet = TLorentzVector();
      TLVjet.SetPtEtaPhiE(0, 0, 0, 0);
      Jets.push_back(TLVjet);
    }
  }
  return Jets;
}
Vecf GetPts(Vec Jets){
  Vecf JetPts;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetPts.push_back(Jets[ijet].Pt());
  }
  return JetPts;
}
Vecf GetEtas(Vec Jets){
  Vecf JetEtas;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetEtas.push_back(Jets[ijet].Eta());
  }
  return JetEtas;
}
Vecf GetPhis(Vec Jets){
  Vecf JetPhis;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetPhis.push_back(Jets[ijet].Phi());
  }
  return JetPhis;
}
Vecf GetMs(Vec Jets){
  Vecf JetMs;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetMs.push_back(Jets[ijet].M());
  }
  return JetMs;
}
Vecf GetMasks(Vec Jets){
  Vecf JetMasks;
  for(auto jet : Jets) {
    if(jet.Pt() != 0){
      JetMasks.push_back(true);
    } else {
      JetMasks.push_back(false);
    }
  }
  return JetMasks;
}
""")

def main(**kargs):

  # Settings
  #Sample = 'mc16e/dijets'
  
  ################################################################################
  # DO NOT MODIFY (below this line)
  ################################################################################

  # Global settings
  TreeName              = 'trees_SRRPV_'
  ApplyEventSelections  = True
  Debug                 = False
  MinNjets              = 6
  maxNjets              = 20
  minJetPt              = 20 # to be safe but there seems to be no jet below 20GeV

  # Create file with selected options
  Config = open('Options.txt','w')
  Config.write('ApplyEventSelections = {}\n'.format(ApplyEventSelections))
  Config.write('MinNjets             = {}\n'.format(MinNjets))
  Config.write('maxNjets             = {}\n'.format(maxNjets))
  Config.write('minJetPt             = {}\n'.format(minJetPt))
  Config.close()

  # Imports
  import ROOT
  import h5py,os,sys,argparse
  import numpy as np
  import logging
  logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
  log = logging.getLogger('')
  if Debug: log.setLevel("DEBUG")

  # Enable multithreading
  nthreads = 6
  ROOT.EnableImplicitMT(nthreads)

  ##############################################################################################
  # Find out how many events pass the event selections
  ##############################################################################################

  # Create TChain with input TTrees
  input_file = ROOT.TFile.Open(kargs['inputFile'])
  tree = input_file.Get(TreeName)
  
  log.info('{} events will be processed'.format(tree.GetEntries()))

  # Create RDataFrame
  DF = ROOT.RDataFrame(tree)
  if not DF:
    log.fatal('RDataFrame can not be created, exiting')
    sys.exit(1)
  
  # Select jets and discard events w/o enough passing jets
  DF = DF.Define('GoodJets', f'SelectJets(jet_pt, jet_eta, jet_phi, jet_e, jet_isSig, jet_passOR, {minJetPt}, {maxNjets})').Filter(f"GoodJets.size() >= {MinNjets}")
  DF = DF.Define('pt', f'GetPts(GoodJets)')
  DF = DF.Define('eta', f'GetEtas(GoodJets)')
  DF = DF.Define('phi', f'GetPhis(GoodJets)')
  DF = DF.Define('mass', f'GetMs(GoodJets)')
  DF = DF.Define('mask', f'GetMasks(GoodJets)')
  log.info('Get number of selected events')
  nPassingEvents = DF.Count().GetValue() # get number of selected events
  log.info(f'{nPassingEvents = }')

  ##############################################################################################
  # Create output H5 file(s)
  ##############################################################################################

  # Set structure of output H5 file
  Types      = { 'mask':bool, 'q1':int, 'q2':int, 'q3':int }
  Structure  = {
    'source' : { 'cases' : { # variable : shape
                   'eta'   : (nPassingEvents,maxNjets),
                   'mask'  : (nPassingEvents,maxNjets),
                   'mass'  : (nPassingEvents,maxNjets),
                   'phi'   : (nPassingEvents,maxNjets),
                   'pt'    : (nPassingEvents,maxNjets),
                 }},
    'g1'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingEvents,) },
  }

  # Create H5 file
  outFileName = '{}_{}.h5'.format(kargs['sample'], kargs['dsid'])
  log.info('Creating {}...'.format(outFileName))
  with h5py.File(outFileName, 'w') as hf:
    for key in Structure:
      Group = hf.create_group(key)
      for case in Structure[key]['cases']:
        log.info(f'Writing "{case}" data into the "{key}" group...')
        if key == 'source':
          data = DF.AsNumpy(columns = [case])[case]
          Group.create_dataset(case, data = np.stack(data) if case != 'mask' else np.stack(data).astype(bool))
          del data
        else: # g1 or g2
          Group.create_dataset(case, data = np.tile(np.array([False if case == 'mask' else -1]), nPassingEvents))
      # try with create_dataset(..., chunks=True)?

  # Close input file
  del tree
  log.info('>>> ALL DONE <<<')

if __name__ == '__main__':
  datasets = { # dsid : full_file_name
    '364702' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged/JZ2/364702.root',
    #'364703' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ3/364703.root',
    #'364704' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ4/364704.root',
    ##'364705' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ5/364705.root', # not available
    #'364706' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ6/364706.root',
    #'364707' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ7/364707.root',
    #'364708' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ8/364708.root',
    #'364709' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ9/364709.root',
    #'364710' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v4/JZ10/364710.root',
    #'364711' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v4/JZ11/364711.root',
    #'364712' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_merged_2022_v3/JZ12/364712.root',
  }
  for dsid, input_file in datasets.items():
    main(sample = 'mc16e_dijets', inputFile = input_file, dsid = dsid)
