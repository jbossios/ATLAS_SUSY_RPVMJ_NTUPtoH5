################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted for ttbar                                 #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch)                                      #
# Date:    29 June 2021                                                        #
#                                                                              #
################################################################################

# Settings
PATH                   = 'Inputs/'
TreeName               = 'trees_SRRPV_'
ApplyEventSelections   = True
shuffleJets            = False
SplitDataset4Training  = True # use 90% of total selected events for training
ProduceTrainingDataset = True
ProduceTestingDataset  = True
Debug                  = False

###################################
# DO NOT MODIFY (below this line)
###################################

# Global settings
dRcut       = 0.4
maxNjets    = 10
minJetPt    = 20 # to be safe but there seems to be no jet below 20GeV

# Imports
import ROOT
import h5py,os,sys,argparse
import numpy as np
import pandas as pd
import random
random.seed(4) # set the random seed for reproducibility
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
log = logging.getLogger('')
if Debug: log.setLevel("DEBUG")

# Enable multithreading
nthreads = 4
ROOT.EnableImplicitMT(nthreads)

###############
# Protections
###############
if not SplitDataset4Training and ProduceTrainingDataset and not ProduceTestingDataset:
  log.fatal('Asked to produce only training dataset but SplitDataset4Training is disabled, exiting')
  sys.exit(1)
if not SplitDataset4Training and ProduceTestingDataset and not ProduceTrainingDataset:
  log.fatal('Asked to produce only testing dataset but SplitDataset4Training is disabled, exiting')
  sys.exit(1)

# Choose datasets to be produced (if splitting is requested)
if SplitDataset4Training:
  Datasets2Produce = []
  if ProduceTrainingDataset: Datasets2Produce.append('training')
  if ProduceTestingDataset:  Datasets2Produce.append('testing')

##################
# Helpful classes
##################

class iJet(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.bTagged = False

class iParton(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.parentBarcode = -999
    self.parentID      = -999
    self.barcode       = -999
    self.pdgID         = -999

##############################################################################################
# Find out how many events pass the event selections
##############################################################################################

tree = ROOT.TChain(TreeName)
for Folder in os.listdir(PATH):
  for folder in os.listdir(PATH+Folder):
    if folder != 'data-trees': continue
    path = PATH+Folder+'/'+folder+'/'
    for File in os.listdir(path): # Loop over files
      tree.Add(path+File)

# Split selected events b/w training and testing (if requested)
EventNumbers4Training = []
EventNumbers4Testing  = []

# Loop over events
nPassingEvents = 0
for event in tree:
  AllPassJets   = [iJet().SetPtEtaPhiE(tree.jet_pt[i],tree.jet_eta[i],tree.jet_phi[i],tree.jet_e[i]) for i in range(len(tree.jet_pt)) if tree.jet_passOR[i] and tree.jet_pt[i] > minJetPt]
  SelectedJets  = [AllPassJets[i] for i in range(min(maxNjets,len(AllPassJets)))] # Select leading n jets with n == min(maxNjets,njets)
  #nJets         = len(tree.jet_pt)
  nJets         = len(SelectedJets)
  nQuarksFromWs = len(tree.truth_QuarkFromW_pt)
  nbsFromTops   = len(tree.truth_bFromTop_pt)
  nPartons      = nQuarksFromWs + nbsFromTops
  # Apply event selections
  passEventSelection = True
  if ApplyEventSelections:
    if nJets < 6:    passEventSelection = False
    if nPartons !=6: passEventSelection = False
  if not passEventSelection: continue # skip event
  if random.random() < 0.7: # use this event for training
    EventNumbers4Training.append(tree.eventNumber)
  else: # use this event for testing
    EventNumbers4Testing.append(tree.eventNumber)
  nPassingEvents += 1
log.info('{} events were selected'.format(nPassingEvents))

nPassingTrainingEvents = len(EventNumbers4Training)
nPassingTestingEvents  = len(EventNumbers4Testing)

if SplitDataset4Training:
  log.info('{} events were selected for training'.format(nPassingTrainingEvents))
  log.info('{} events were selected for testing'.format(nPassingTestingEvents))

# Protection
if nPassingTrainingEvents + nPassingTestingEvents != nPassingEvents:
  log.fatal('Number of training and selected events do not match total number of passing events, exiting')
  sys.exit(1)

##############################################################################################
# Create output H5 file(s)
##############################################################################################

# Structure of output H5 file
Types      = { 'mask':bool, 'b':int, 'q1':int, 'q2':int }
Structure  = {
  'all' : {
    'source' : { 'cases' : ['btag','eta','mask','mass','phi','pt'], 'shape' : (nPassingEvents,maxNjets) },
    't1'     : { 'cases' : ['b','mask','q1','q2'],                  'shape' : (nPassingEvents,) },
    't2'     : { 'cases' : ['b','mask','q1','q2'],                  'shape' : (nPassingEvents,) },
  },
  'training' : {
    'source' : { 'cases' : ['btag','eta','mask','mass','phi','pt'], 'shape' : (nPassingTrainingEvents,maxNjets) },
    't1'     : { 'cases' : ['b','mask','q1','q2'],                  'shape' : (nPassingTrainingEvents,) },
    't2'     : { 'cases' : ['b','mask','q1','q2'],                  'shape' : (nPassingTrainingEvents,) },
  },
  'testing' : {
    'source' : { 'cases' : ['btag','eta','mask','mass','phi','pt'], 'shape' : (nPassingTestingEvents,maxNjets) },
    't1'     : { 'cases' : ['b','mask','q1','q2'],                  'shape' : (nPassingTestingEvents,) },
    't2'     : { 'cases' : ['b','mask','q1','q2'],                  'shape' : (nPassingTestingEvents,) },
  },
}

# Create H5 file(s)
if not SplitDataset4Training:
  outFileName = 'AllData.h5'
  log.info('Creating {}...'.format(outFileName))
  HF          = h5py.File(outFileName, 'w')
  Groups      = dict()
  Datasets    = dict()
  for key in Structure['all']:
    Groups[key] = HF.create_group(key)
    for case in Structure['all'][key]['cases']:
      Datasets[key+'_'+case] = Groups[key].create_dataset(case,Structure['all'][key]['shape'],Types[case] if case in Types else float)
else: # split dataset into training and testing datasets
  Groups      = dict()
  Datasets    = dict()
  for datatype in Datasets2Produce:
    outFileName = 'AllData_{}.h5'.format(datatype)
    log.info('Creating {}...'.format(outFileName))
    HF                 = h5py.File(outFileName, 'w')
    Groups[datatype]   = dict()
    Datasets[datatype] = dict()
    for key in Structure[datatype]:
      Groups[datatype][key] = HF.create_group(key)
      for case in Structure[datatype][key]['cases']:
        Datasets[datatype][key+'_'+case] = Groups[datatype][key].create_dataset(case,Structure[datatype][key]['shape'],Types[case] if case in Types else float)
  
##############################################################################################
# Loop over events and fill the numpy arrays on each event
##############################################################################################

#nEntries = tree.GetEntries()

# Loop over events
allCounter      = -1
trainingCounter = -1
testingCounter  = -1
log.info('About to enter event loop')
ROOT.EnableImplicitMT(nthreads)
for event in tree:

  # Find number of particles/jets
  # Select reco jets
  AllPassJets = []
  for ijet in range(len(tree.jet_pt)):
    if tree.jet_passOR[ijet] and tree.jet_pt[ijet] > minJetPt:
      jet = iJet()
      jet.SetPtEtaPhiE(tree.jet_pt[ijet],tree.jet_eta[ijet],tree.jet_phi[ijet],tree.jet_e[ijet])
      jet.bTagged = int(ord(tree.jet_bTag[ijet])) # convert chart to int
      AllPassJets.append(jet)
  SelectedJets  = [AllPassJets[i] for i in range(min(maxNjets,len(AllPassJets)))] # Select leading n jets with n == min(maxNjets,njets)
  nJets         = len(SelectedJets)
  nQuarksFromWs = len(tree.truth_QuarkFromW_pt)
  nbsFromTops   = len(tree.truth_bFromTop_pt)
  nPartons      = nQuarksFromWs + nbsFromTops
  
  # Apply event selections
  passEventSelection = True
  if ApplyEventSelections:
    if nJets < 6:    passEventSelection = False
    if nPartons !=6: passEventSelection = False
  if not passEventSelection: continue # skip event

  allCounter += 1
  if (allCounter+1) % 10000 == 0:
    log.info('{} events processed (of {})'.format(allCounter+1,nPassingEvents))

  # Was this event assigned for training or testing?
  if SplitDataset4Training:
    ForTraining = True # if False then event to be used for testing
    if tree.eventNumber in EventNumbers4Testing: ForTraining = False
    if not ProduceTrainingDataset and ForTraining: continue    # skip event meant for training since asked not to produce training dataset
    if not ProduceTestingDataset and not ForTraining: continue # skip event meant for testing since asked not to produce testing dataset

  # Protection
  if nJets > maxNjets:
    log.fatal('More than {} jets were found ({}), update script!'.format(maxNjets,nJets))
    sys.exit(1)
 
  # Remove pt ordering in the jet array (if requested)
  if shuffleJets:
    random.shuffle(SelectedJets)

  # Select quarks from Ws
  QuarksFromWs = [iParton() for i in range(nQuarksFromWs)]
  for iquark in range(nQuarksFromWs):
    QuarksFromWs[iquark].SetPtEtaPhiE(tree.truth_QuarkFromW_pt[iquark],tree.truth_QuarkFromW_eta[iquark],tree.truth_QuarkFromW_phi[iquark],tree.truth_QuarkFromW_e[iquark])
    QuarksFromWs[iquark].parentBarcode = tree.truth_QuarkFromW_ParentBarcode[iquark]
    QuarksFromWs[iquark].parentID      = tree.truth_QuarkFromW_ParentID[iquark]
    QuarksFromWs[iquark].pdgID         = tree.truth_QuarkFromW_pdgID[iquark]

  # Select b quaks from Tops
  bsFromTops = [iParton() for i in range(nbsFromTops)]
  for b in range(nbsFromTops):
    bsFromTops[b].SetPtEtaPhiE(tree.truth_bFromTop_pt[b],tree.truth_bFromTop_eta[b],tree.truth_bFromTop_phi[b],tree.truth_bFromTop_e[b])
    bsFromTops[b].parentBarcode = tree.truth_bFromTop_ParentBarcode[b]
    bsFromTops[b].parentID      = tree.truth_bFromTop_ParentID[b]

  # Make array with partons to be matched to reco jets
  Partons = QuarksFromWs+bsFromTops

  # Match reco jets to closest parton
  Assigments = {
    # place holder with temporary values
    'source' : {'btag' : 0, 'eta': 0, 'mass': 0, 'phi': 0, 'pt' : 0, 'mask': True},
    # jet index for each particle b,q1,q2 (if no matching then -1) and mask set temporarily to True
    't1'     : {'b': -1, 'q1': -1, 'q2': -1, 'mask': True},
    't2'     : {'b': -1, 'q1': -1, 'q2': -1, 'mask': True},
  }
  jetIndex         = -1
  matchPartonIndex = -1
  for jet in SelectedJets: # loop over jets
    jetIndex   += 1
    dRmin       = 1E5
    partonIndex = -1
    for parton in Partons: # loop over partons
      partonIndex += 1
      dR = jet.DeltaR(parton)
      if dR < dRmin:
        dRmin = dR
        matchPartonIndex = partonIndex
    if dRmin < dRcut: # jet matches a parton (quark from W boson or b from top quark)
      # Find to which top it matches and determines if the parton is b, q1 or q2
      partonParentID = Partons[matchPartonIndex].parentID
      if partonParentID == 24 or partonParentID == -24: # quark from W
        Assigments['t1' if partonParentID == 24 else 't2']['q1' if Partons[matchPartonIndex].pdgID > 0 else 'q2'] = jetIndex
      elif partonParentID == 6 or partonParentID == -6: # b quark from top quark
        if jet.bTagged: # consider b-quark matched to reco jet only if reco jet is b-tagged
          Assigments['t1' if partonParentID == 6 else 't2']['b'] = jetIndex
      else:
        log.FATAL('partonParentID ({}) not recornized, exiting'.format(partonParentID))
        sys.exit(1)

  # Protection: make sure the same jet was not matched to two partons
  JetIndexes = [] # indexes of jets matched to partons
  for t in ['t1','t2']:
    for key in Assigments[t]:
      if key == 'mask': continue
      index = Assigments[t][key]
      if index != -1:
        if index not in JetIndexes:
          JetIndexes.append(index)
        else:
          log.warning('Jet index ({}) was assigned to more than one parton!'.format(index))

  # Create arrays with jet info (extend Assigments with jet reco info)
  for case in Structure['all']['source']['cases']:
     array = []
     for j in SelectedJets:
       if case == 'btag':
         array.append(j.bTagged)
       elif case == 'eta':
         array.append(j.Eta())
       elif case == 'mass':
         array.append(j.M())
       elif case == 'phi':
         array.append(j.Phi())
       elif case == 'pt':
         array.append(j.Pt())
       elif case == 'mask':
         array.append(True)
     if nJets < maxNjets: # add extra (padding) jets to keep the number of jets fixed
       for i in range(nJets,maxNjets):
         if case != 'mask':
           array.append(0.)
         else:
           array.append(False)
     Assigments['source'][case] = np.array(array)

  # See if top quarks were fully reconstructed (i.e. each decay particle matches a jet)
  for t in ['t1','t2']:
    TempMask = True
    for key in Assigments[t]:
      if Assigments[t][key] == -1: TempMask = False
    Assigments[t]['mask'] = TempMask

  # Split dataset b/w traning and testing (if requested)
  if SplitDataset4Training:
    if tree.eventNumber in EventNumbers4Training:
      trainingCounter += 1
      # Add data to the h5 training file
      for key in Structure['training']:
        for case in Structure['training'][key]['cases']:
          Datasets['training'][key+'_'+case][trainingCounter] = Assigments[key][case]
    elif tree.eventNumber in EventNumbers4Testing:
      testingCounter += 1
      # Add data to the h5 testing file
      for key in Structure['testing']:
        for case in Structure['testing'][key]['cases']:
          Datasets['testing'][key+'_'+case][testingCounter] = Assigments[key][case]
    else:
      log.error('Event is simultaneously not considered for training nor for testing, exiting')
      sys.exit(1)
  else: # Add data to a single h5 file
    for key in Structure:
      for case in Structure[key]['cases']:
        Datasets[key+'_'+case][allCounter] = Assigments[key][case]

# Close input file
del tree
HF.close()
  
log.info('>>> ALL DONE <<<')
