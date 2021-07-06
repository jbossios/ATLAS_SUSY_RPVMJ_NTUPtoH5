################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted for ttbar                                 #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch)                                      #
# Date:    29 June 2021                                                        #
#                                                                              #
################################################################################

# Settings
PATH                  = '/eos/user/j/jbossios/SUSY/NTUPs/ttbar/'
TreeName              = 'trees_SRRPV_'
ApplyEventSelections  = True
shuffleJets           = False
SplitDataset4Training = True # use 70% of total selected events for training

###################################
# DO NOT MODIFY (below this line)
###################################

# Global settings
dRcut       = 0.5
maxNjets    = 30

# Imports
from ROOT import *
from ROOT import RDataFrame
import h5py,os,sys,argparse
import numpy as np
import pandas as pd
import random

##################
# Helpful classes
##################

class iJet(TLorentzVector):
  def __init__(self):
    TLorentzVector.__init__(self)
    self.bTagged = False

class iParton(TLorentzVector):
  def __init__(self):
    TLorentzVector.__init__(self)
    self.parentBarcode = -999
    self.parentID      = -999
    self.barcode       = -999
    self.pdgID         = -999

##############################################################################################
# Find out how many events pass the event selections
##############################################################################################

tree = TChain(TreeName)
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
  nJets         = len(tree.jet_pt)
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
print('INFO: {} events were selected'.format(nPassingEvents))

nPassingTrainingEvents = len(EventNumbers4Training)
nPassingTestingEvents  = len(EventNumbers4Testing)

##############################################################################################
# Create output H5 file(s)
##############################################################################################

# Structure of output H5 file
Types      = { 'btag':int, 'mask':bool, 'b':int, 'q1':int, 'q2':int}
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
  print('Creating {}...'.format(outFileName))
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
  for datatype in ['training','testing']:
    outFileName = 'AllData_{}.h5'.format(datatype)
    print('Creating {}...'.format(outFileName))
    HF             = h5py.File(outFileName, 'w')
    Groups[datatype]   = dict()
    Datasets[datatype] = dict()
    for key in Structure[datatype]:
      Groups[datatype][key] = HF.create_group(key)
      for case in Structure[datatype][key]['cases']:
        Datasets[datatype][key+'_'+case] = Groups[datatype][key].create_dataset(case,Structure[datatype][key]['shape'],Types[case] if case in Types else float)
  
##############################################################################################
# Loop over events and fill the numpy arrays on each event
##############################################################################################

nEntries = tree.GetEntries()

# Loop over events
allCounter      = -1
trainingCounter = -1
testingCounter  = -1
#for event in tree:
for ientry in range(0, nEntries):
  
  tree.GetEntry(ientry)

  # Find number of particles
  nJets         = len(tree.jet_pt)
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
  if allCounter+1 % 10000 == 0:
    print('INFO: {} events processed (of {})'.format(allCounter+1,nPassingEvents))

  # Protection
  if nJets > maxNjets:
    print('ERROR: More than {} jets were found ({}), update script!'.format(maxNjets,nJets))
    sys.exit(1)

  # Select reco jets
  SelectedJets = [iJet() for i in range(nJets)]
  for ijet in range(nJets):
    SelectedJets[ijet].SetPtEtaPhiE(tree.jet_pt[ijet],tree.jet_eta[ijet],tree.jet_phi[ijet],tree.jet_e[ijet])
    SelectedJets[ijet].bTagged = int(ord(tree.jet_bTag[ijet])) # convert chart to int

  # Remove pt ordering in the jet array (if requested)
  if shuffleJets:
    seed = 4 # Set the random seed for reproducibility
    random.Random(seed).shuffle(SelectedJets)

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
        Assigments['t1' if partonParentID == 6 else 't2']['b'] = jetIndex
      else:
        print('ERROR: partonParentID ({}) not recornized, exiting'.format(partonParentID))
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
          print('WARNING: Jet index ({}) was assigned to more than one parton!'.format(index))

  # Create arrays with jet info (extend Assigments with jet reco info)
  for case in Structure['all']['source']:
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
        for case in Structure['training'][t]['cases']:
          Datasets['training'][t+'_'+case][trainingCounter] = Assigments[t][case]
    elif tree.eventNumber in EventNumbers4Testing:
      testingCounter += 1
      # Add data to the h5 testing file
      for key in Structure['testing']:
        for case in Structure['testing'][t]['cases']:
          Datasets['testing'][t+'_'+case][testingCounter] = Assigments[t][case]
    else:
      print('ERROR: Event is simultaneously not considered for training nor for testing, exiting')
      sys.exit(1)
  else: # Add data to a single h5 file
    for key in Structure:
      for case in Structure[t]['cases']:
        Datasets[t+'_'+case][allCounter] = Assigments[t][case]

# Close input file
del tree
  
print('>>> ALL DONE <<<')
