################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted for signal direct decay mode samples      #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch)                                      #
# Date:    22 July 2021                                                        #
#                                                                              #
################################################################################

from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVJet
from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVParton
from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVMatcher

# Settings
PATH                   = 'SignalInputs/MC16a_21_2_173_0_with_normweight/'
TreeName               = 'trees_SRRPV_'
ApplyEventSelections   = True
shuffleJets            = False
SplitDataset4Training  = False # use 90% of total selected events for training
ProduceTrainingDataset = True
ProduceTestingDataset  = True
Debug                  = False
MinNjets               = 6
maxNjets               = 8
FlavourType            = 'UDB+UDS' # options: All (ALL+UDB+UDS), UDB, UDS, UDB+UDS
MassPoints             = '1400' # Options: All, Low, Intermediate, IntermediateWo1400, High, 1400
UseAllFiles            = False  # Use only when running on Lea's files, meant to overrule FlavourType and MassPoints options
#MatchingCriteria       = 'UseFTDeltaRvalues' # Options: UseFTDeltaRvalues, RecomputeDeltaRvalues_ptPriority_ptPriority, RecomputeDeltaRvalues_drPriority # FIXME switch to which RecomputeDeltaRvalues?
#MatchingCriteria       = 'RecomputeDeltaRvalues_ptPriority' # Options: UseFTDeltaRvalues, RecomputeDeltaRvalues_ptPriority, RecomputeDeltaRvalues_drPriority # FIXME switch to RecomputeDeltaRvalues
MatchingCriteria       = 'RecomputeDeltaRvalues_drPriority' # Options: UseFTDeltaRvalues, RecomputeDeltaRvalues_ptPriority, RecomputeDeltaRvalues_drPriority # FIXME switch to RecomputeDeltaRvalues
ForceHalf              = False  # Force to use only half of input events (only works for 1400 samples, MassPoints==1400)

################################################################################
# DO NOT MODIFY (below this line)
################################################################################

###############################
# Conventions
###############################
# q1 is the first matched quark found for the corresponding gluino (f1 is its pdgID)
# q2 is the second matched quark found for the corresponding gluino (f2 is its pdgID)
# q3 is the third matched quark found for the corresponding gluino (f3 is its pdgID)
# g1 is the first parent gluino for first matched quark

# Global settings
dRcut       = 0.4
minJetPt    = 20 # to be safe but there seems to be no jet below 20GeV

# Create file with selected options
Config = open('Options.txt','w')
Config.write('ApplyEventSelections = {}\n'.format(ApplyEventSelections))
Config.write('ShuffleJets          = {}\n'.format(shuffleJets))
Config.write('MinNjets             = {}\n'.format(MinNjets))
Config.write('FlavourType          = {}\n'.format(FlavourType))
Config.write('MassPoints           = {}\n'.format(MassPoints))
Config.write('dRcut                = {}\n'.format(dRcut))
Config.write('maxNjets             = {}\n'.format(maxNjets))
Config.write('minJetPt             = {}\n'.format(minJetPt))
Config.close()

# Imports
import ROOT
import h5py,os,sys,argparse
import numpy as np
import pandas as pd
import random
random.seed(4) # set the random seed for reproducibility
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
log = logging.getLogger('CreateH4Files')
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

##############################################################################################
# Find out how many events pass the event selections
##############################################################################################

allDSIDs = {
  'All' : {
    "504513" : "GG_rpv_UDB_900",
    "504514" : "GG_rpv_UDB_1000",
    "504515" : "GG_rpv_UDB_1100",
    "504516" : "GG_rpv_UDB_1200",
    "504517" : "GG_rpv_UDB_1300",
    "504518" : "GG_rpv_UDB_1400",
    "504519" : "GG_rpv_UDB_1500",
    "504520" : "GG_rpv_UDB_1600",
    "504521" : "GG_rpv_UDB_1700",
    "504522" : "GG_rpv_UDB_1800",
    "504523" : "GG_rpv_UDB_1900",
    "504524" : "GG_rpv_UDB_2000",
    "504525" : "GG_rpv_UDB_2100",
    "504526" : "GG_rpv_UDB_2200",
    "504527" : "GG_rpv_UDB_2300",
    "504528" : "GG_rpv_UDB_2400",
    "504529" : "GG_rpv_UDB_2500",
    "504534" : "GG_rpv_UDS_900",
    "504535" : "GG_rpv_UDS_1000",
    "504536" : "GG_rpv_UDS_1100",
    "504537" : "GG_rpv_UDS_1200",
    "504538" : "GG_rpv_UDS_1300",
    "504539" : "GG_rpv_UDS_1400",
    "504540" : "GG_rpv_UDS_1500",
    "504541" : "GG_rpv_UDS_1600",
    "504542" : "GG_rpv_UDS_1700",
    "504543" : "GG_rpv_UDS_1800",
    "504544" : "GG_rpv_UDS_1900",
    "504545" : "GG_rpv_UDS_2000",
    "504546" : "GG_rpv_UDS_2100",
    "504547" : "GG_rpv_UDS_2200",
    "504548" : "GG_rpv_UDS_2300",
    "504549" : "GG_rpv_UDS_2400",
    "504550" : "GG_rpv_UDS_2500",
    "504551" : "GG_rpv_ALL_1800",
    "504552" : "GG_rpv_ALL_2200",
  },
  'Low' : {
    "504513" : "GG_rpv_UDB_900",
    "504514" : "GG_rpv_UDB_1000",
    "504515" : "GG_rpv_UDB_1100",
    "504516" : "GG_rpv_UDB_1200",
    "504517" : "GG_rpv_UDB_1300",
    "504534" : "GG_rpv_UDS_900",
    "504535" : "GG_rpv_UDS_1000",
    "504536" : "GG_rpv_UDS_1100",
    "504537" : "GG_rpv_UDS_1200",
    "504538" : "GG_rpv_UDS_1300",
  },
  'Intermediate' : {
    "504518" : "GG_rpv_UDB_1400",
    "504519" : "GG_rpv_UDB_1500",
    "504520" : "GG_rpv_UDB_1600",
    "504521" : "GG_rpv_UDB_1700",
    "504522" : "GG_rpv_UDB_1800",
    "504523" : "GG_rpv_UDB_1900",
    "504539" : "GG_rpv_UDS_1400",
    "504540" : "GG_rpv_UDS_1500",
    "504541" : "GG_rpv_UDS_1600",
    "504542" : "GG_rpv_UDS_1700",
    "504543" : "GG_rpv_UDS_1800",
    "504544" : "GG_rpv_UDS_1900",
    "504551" : "GG_rpv_ALL_1800",
  },
  'IntermediateWo1400' : {
    "504519" : "GG_rpv_UDB_1500",
    "504520" : "GG_rpv_UDB_1600",
    "504521" : "GG_rpv_UDB_1700",
    "504522" : "GG_rpv_UDB_1800",
    "504523" : "GG_rpv_UDB_1900",
    "504540" : "GG_rpv_UDS_1500",
    "504541" : "GG_rpv_UDS_1600",
    "504542" : "GG_rpv_UDS_1700",
    "504543" : "GG_rpv_UDS_1800",
    "504544" : "GG_rpv_UDS_1900",
    "504551" : "GG_rpv_ALL_1800",
  },
  'High' : {
    "504524" : "GG_rpv_UDB_2000",
    "504525" : "GG_rpv_UDB_2100",
    "504526" : "GG_rpv_UDB_2200",
    "504527" : "GG_rpv_UDB_2300",
    "504528" : "GG_rpv_UDB_2400",
    "504529" : "GG_rpv_UDB_2500",
    "504545" : "GG_rpv_UDS_2000",
    "504546" : "GG_rpv_UDS_2100",
    "504547" : "GG_rpv_UDS_2200",
    "504548" : "GG_rpv_UDS_2300",
    "504549" : "GG_rpv_UDS_2400",
    "504550" : "GG_rpv_UDS_2500",
    "504552" : "GG_rpv_ALL_2200",
  },
  '1400' : {
    "504518" : "GG_rpv_UDB_1400",
    "504539" : "GG_rpv_UDS_1400",
  },
}[MassPoints]

Flavours = []
if FlavourType == 'All':
  Flavours = ['ALL','UDS','UDB']
if 'UDB' in FlavourType:
  Flavours.append('UDB')
if 'UDS' in FlavourType:
  Flavours.append('UDS')
DSIDs = dict()
for dsid in allDSIDs:
  for flav in Flavours:
    if flav in allDSIDs[dsid]:
      DSIDs[dsid] = allDSIDs[dsid]
      break

tree = ROOT.TChain(TreeName)
for File in os.listdir(PATH):
  if '.root' not in File: continue # skip non-TFile files
  dsid = File.replace('.root','')
  if not UseAllFiles and dsid not in DSIDs: continue # skip undesired DSID
  tree.Add(PATH+File)

# Split selected events b/w training and testing (if requested)
EventNumbers4Training = []
EventNumbers4Testing  = []

# Collect info to know matching efficiency for each quark flavour
NquarksByFlavour        = {flav : 0 for flav in [1,2,3,4,5,6]}
NmatchedQuarksByFlavour = {flav : 0 for flav in [1,2,3,4,5,6]}

# Loop over events
nPassingEvents = 0
counter = -1
selectedEvents = 0
totalEvents = tree.GetEntries()
for event in tree:
  counter += 1
  AllPassJets = [RPVJet(tree.jet_pt[i], tree.jet_eta[i], tree.jet_phi[i], tree.jet_e[i]) for i in range(len(tree.jet_pt)) if tree.jet_passOR[i] and tree.jet_isSig[i] and tree.jet_pt[i] > minJetPt]
  SelectedJets  = [AllPassJets[i] for i in range(min(maxNjets,len(AllPassJets)))] # Select leading n jets with n == min(maxNjets,njets)
  nJets         = len(SelectedJets)
  nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt)
  # Apply event selections
  passEventSelection = True
  if ApplyEventSelections:
    if nJets < MinNjets:  passEventSelection = False
    if nQuarksFromGs !=6: passEventSelection = False
  if not passEventSelection: continue # skip event
  selectedEvents += 1
  #if ForceHalf and selectedEvents > 117941*0.5: continue
  if ForceHalf and selectedEvents > 16867: continue
  if random.random() < 0.9: # use this event for training
    EventNumbers4Training.append(counter)
  else: # use this event for testing
    EventNumbers4Testing.append(counter)
  nPassingEvents += 1
log.info('{} events out of {} were selected'.format(nPassingEvents, totalEvents))

nPassingTrainingEvents = len(EventNumbers4Training)
nPassingTestingEvents  = len(EventNumbers4Testing)

if SplitDataset4Training:
  log.info('{} events were selected for training'.format(nPassingTrainingEvents))
  log.info('{} events were selected for testing'.format(nPassingTestingEvents))

# Protection
if (nPassingTrainingEvents + nPassingTestingEvents) != nPassingEvents:
  log.fatal('Number of training and selected events do not match total number of passing events, exiting')
  sys.exit(1)

##############################################################################################
# Create output H5 file(s)
##############################################################################################

# Structure of output H5 file
Types      = { 'mask':bool, 'q1':int, 'q2':int, 'q3':int, 'f1':int, 'f2':int, 'f3':int }
Structure  = {
  'all' : {
    'source' : { 'cases' : { # variable : shape
                   'eta'   : (nPassingEvents,maxNjets),
                   'mask'  : (nPassingEvents,maxNjets),
                   'mass'  : (nPassingEvents,maxNjets),
                   'phi'   : (nPassingEvents,maxNjets),
                   'pt'    : (nPassingEvents,maxNjets),
                   'gmass' : (nPassingEvents,)
                 }},
    'g1'     : { 'cases' : ['mask','q1','q2','q3','f1','f2','f3'], 'shape' : (nPassingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3','f1','f2','f3'], 'shape' : (nPassingEvents,) },
    'normweight' : { 'cases' : ['normweight'], 'shape' : (nPassingEvents,) },
  },
  'training' : {
    'source' : { 'cases' : { # variable : shape
                   'eta'   : (nPassingTrainingEvents,maxNjets),
                   'mask'  : (nPassingTrainingEvents,maxNjets),
                   'mass'  : (nPassingTrainingEvents,maxNjets),
                   'phi'   : (nPassingTrainingEvents,maxNjets),
                   'pt'    : (nPassingTrainingEvents,maxNjets),
                   'gmass' : (nPassingTrainingEvents,)
                 }},
    'g1'     : { 'cases' : ['mask','q1','q2','q3','f1','f2','f3'], 'shape' : (nPassingTrainingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3','f1','f2','f3'], 'shape' : (nPassingTrainingEvents,) },
    'normweight' : { 'cases' : ['normweight'], 'shape' : (nPassingTrainingEvents,) },
  },
  'testing' : {
    'source' : { 'cases' : { # variable : shape
                   'eta'   : (nPassingTestingEvents,maxNjets),
                   'mask'  : (nPassingTestingEvents,maxNjets),
                   'mass'  : (nPassingTestingEvents,maxNjets),
                   'phi'   : (nPassingTestingEvents,maxNjets),
                   'pt'    : (nPassingTestingEvents,maxNjets),
                   'gmass' : (nPassingTestingEvents,)
                 }},
    'g1'     : { 'cases' : ['mask','q1','q2','q3','f1','f2','f3'], 'shape' : (nPassingTestingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3','f1','f2','f3'], 'shape' : (nPassingTestingEvents,) },
    'normweight' : { 'cases' : ['normweight'], 'shape' : (nPassingTestingEvents,) },
  },
}

########################################################
# Helpful functions
########################################################

def get_quark_flavour(pdgid, g, dictionary):
  """ Function that assigns quarks to q1, q2 or q3 """
  if dictionary[g]['q1'] == 0:
    dictionary[g]['q1'] = pdgid
    qFlavour = 'q1'
  elif dictionary[g]['q2'] == 0:
    dictionary[g]['q2'] = pdgid
    qFlavour = 'q2'
  else:
    qFlavour = 'q3'
  return dictionary, qFlavour

def get_parton_info(Partons, barcode):
  """ Get info of quark from gluino matched to a jet """
  # Loop over quarks from gluinos
  for partonIndex, parton in enumerate(Partons):
    if Partons[partonIndex].barcode == barcode:
      return partonIndex, Partons[partonIndex].pdgID, Partons[partonIndex].parentBarcode
  return -1,-1,-1

def get_fsr_info(FSRs, barcode):
  """ Get info of FSR quark matched to a jet """
  for fsr_index, FSR in enumerate(FSRs): # loop over FSRs
    if FSRs[fsr_index].barcode == barcode:
      return fsr_index, FSRs[fsr_index].pdgID, FSRs[fsr_index].parentBarcode, FSRs[fsr_index].originalBarcode
  return -1,-1,-1,-1

def make_assigments(assigments, g_barcodes, q_parent_barcode, pdgid, q_pdgid_dict, selected_jets, jet_index):
  """ Find to which gluino (g1 or g2) a jet was matched and determines if the parton is q1, q2 or q3 (see convention above) """
  if g_barcodes['g1'] == 0: # not assigned yet to any quark parent barcode
    g_barcodes['g1'] = q_parent_barcode
    q_pdgid_dict, q_flavour = get_quark_flavour(pdgid, 'g1', q_pdgid_dict)
    if assigments['g1'][q_flavour] == -1: # not assigned yet
      assigments['g1'][q_flavour] = jet_index
      assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
    else: # if assigned already, pick up jet with largets pT
      if selected_jets[jet_index].Pt() > selected_jets[assigments['g1'][q_flavour]].Pt():
        assigments['g1'][q_flavour] = jet_index
        assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
  else: # g1 was defined alredy (check if quark parent barcode agrees with it)
    if g_barcodes['g1'] == q_parent_barcode:
      q_pdgid_dict, q_flavour = get_quark_flavour(pdgid, 'g1', q_pdgid_dict)
      if assigments['g1'][q_flavour] == -1: # not assigned yet
        assigments['g1'][q_flavour] = jet_index
        assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
      else: # if assigned already, pick up jet with largets pT
        if selected_jets[jet_index].Pt() > selected_jets[assigments['g1'][q_flavour]].Pt():
          assigments['g1'][q_flavour] = jet_index
          assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
    else:
      q_pdgid_dict, q_flavour = get_quark_flavour(pdgid, 'g2', q_pdgid_dict)
      if assigments['g2'][q_flavour] == -1: # not assigned yet
        assigments['g2'][q_flavour] = jet_index
        assigments['g2'][q_flavour.replace('q', 'f')] = pdgid
      else: # if assigned already, pick up jet with largets pT
        if selected_jets[jet_index].Pt() > selected_jets[assigments['g2'][q_flavour]].Pt():
          assigments['g2'][q_flavour] = jet_index
          assigments['g2'][q_flavour.replace('q', 'f')] = pdgid
  return assigments

########################################################
# Create H5 file(s)
########################################################

if not SplitDataset4Training:
  outFileName = '{}SignalData_{}.h5'.format(FlavourType,MassPoints)
  log.info('Creating {}...'.format(outFileName))
  HF = h5py.File(outFileName, 'w')
  Groups, Datasets = dict(), dict()
  for key in Structure['all']:
    Groups[key] = HF.create_group(key)
    for case in Structure['all'][key]['cases']:
      if key == 'source':
        Datasets[key+'_'+case] = Groups[key].create_dataset(case,Structure['all'][key]['cases'][case],Types[case] if case in Types else float)
      else:
        Datasets[key+'_'+case] = Groups[key].create_dataset(case,Structure['all'][key]['shape'],Types[case] if case in Types else float)
else: # split dataset into training and testing datasets
  Groups      = dict()
  Datasets    = dict()
  for datatype in Datasets2Produce:
    outFileName = '{}SignalData_{}_{}.h5'.format(FlavourType,MassPoints,datatype)
    log.info('Creating {}...'.format(outFileName))
    HF = h5py.File(outFileName, 'w')
    Groups[datatype], Datasets[datatype] = dict(), dict()
    for key in Structure[datatype]:
      Groups[datatype][key] = HF.create_group(key)
      for case in Structure[datatype][key]['cases']:
        if key == 'source':
          Datasets[datatype][key+'_'+case] = Groups[datatype][key].create_dataset(case,Structure[datatype][key]['cases'][case],Types[case] if case in Types else float)
        else:
          Datasets[datatype][key+'_'+case] = Groups[datatype][key].create_dataset(case,Structure[datatype][key]['shape'],Types[case] if case in Types else float)
  
##############################################################################################
# Loop over events and fill the numpy arrays on each event
##############################################################################################

# Reconstructed mass by truth mass
Masses      = [900 + i*100 for i in range(0, 17)]
hRecoMasses = {mass : ROOT.TH1D('RecoMass_TruthMass{}'.format(mass), '', 300, 0, 3000) for mass in Masses}

# Histogram for reconstructed gluino mass - true gluino mass
hGluinoMassDiff = ROOT.TH1D('GluinoMassDiff', '', 10000, -10000, 10000)

# Loop over events
allCounter      = -1
trainingCounter = -1
testingCounter  = -1
log.info('About to enter event loop')
ROOT.EnableImplicitMT(nthreads)
matchedEvents               = 0
multipleQuarkMatchingEvents = 0
matchedEventNumbers         = []
for counter, event in enumerate(tree):
  # Select reco jets
  AllPassJets = []
  for ijet in range(len(tree.jet_pt)):
    if tree.jet_passOR[ijet] and tree.jet_isSig[ijet] and tree.jet_pt[ijet] > minJetPt:
      jet = RPVJet()
      jet.SetPtEtaPhiE(tree.jet_pt[ijet], tree.jet_eta[ijet], tree.jet_phi[ijet], tree.jet_e[ijet])
      if MatchingCriteria == 'UseFTDeltaRvalues':
        jet.set_matched_parton_barcode(int(tree.jet_deltaRcut_matched_truth_particle_barcode[ijet]))
        jet.set_matched_fsr_barcode(int(tree.jet_deltaRcut_FSRmatched_truth_particle_barcode[ijet]))
      AllPassJets.append(jet)
  SelectedJets  = [AllPassJets[i] for i in range(min(maxNjets,len(AllPassJets)))] # Select leading n jets with n == min(maxNjets,njets)
  nJets         = len(SelectedJets)
  nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt)
  nFSRsFromGs   = len(tree.truth_FSRFromGluinoQuark_pt)

  # Apply event selections
  passEventSelection = True
  if ApplyEventSelections:
    if nJets < MinNjets:  passEventSelection = False
    if nQuarksFromGs !=6: passEventSelection = False
  if not passEventSelection: continue # skip event

  allCounter += 1
  if (allCounter+1) % 10000 == 0:
    log.info('{} events processed (of {})'.format(allCounter+1,nPassingEvents))

  # Was this event assigned for training or testing?
  if SplitDataset4Training:
    ForTraining = True # if False then event will be used for testing
    if counter in EventNumbers4Testing: ForTraining = False
    if not ProduceTrainingDataset and ForTraining: continue    # skip event meant for training since asked not to produce training dataset
    if not ProduceTestingDataset and not ForTraining: continue # skip event meant for testing since asked not to produce testing dataset

  # Protection
  if nJets > maxNjets:
    log.fatal('More than {} jets were found ({}), fix me!'.format(maxNjets,nJets))
    sys.exit(1)
 
  # Remove pt ordering in the jet array (if requested)
  if shuffleJets:
    random.shuffle(SelectedJets)

  # Extract gluino mass
  for ipart in range(len(tree.truth_parent_m)): # loop over truth particles
    if tree.truth_parent_pdgId[ipart] == 1000021: # it's a gluino
      gmass = tree.truth_parent_m[ipart]
      break

  # Collect gluino barcodes
  gBarcodes = {'g1': 0, 'g2': 0} # fill temporary values

  # Collect quark -> gluino associations
  qPDGIDs = { g : {'q1': 0, 'q2': 0, 'q3': 0} for g in ['g1', 'g2']} # fill temporary values

  # Select quarks from gluinos
  QuarksFromGluinos = [RPVParton() for i in range(nQuarksFromGs)]
  for iquark in range(nQuarksFromGs):
    QuarksFromGluinos[iquark].SetPtEtaPhiE(tree.truth_QuarkFromGluino_pt[iquark],tree.truth_QuarkFromGluino_eta[iquark],tree.truth_QuarkFromGluino_phi[iquark],tree.truth_QuarkFromGluino_e[iquark])
    QuarksFromGluinos[iquark].set_gluino_barcode(tree.truth_QuarkFromGluino_ParentBarcode[iquark])
    QuarksFromGluinos[iquark].set_barcode(tree.truth_QuarkFromGluino_barcode[iquark])
    QuarksFromGluinos[iquark].set_pdgid(tree.truth_QuarkFromGluino_pdgID[iquark])

  # Select FSR quarks from gluinos
  FSRsFromGluinos = [RPVParton() for i in range(nFSRsFromGs)]
  for iFSR in range(nFSRsFromGs):
    FSRsFromGluinos[iFSR].SetPtEtaPhiE(tree.truth_FSRFromGluinoQuark_pt[iFSR],tree.truth_FSRFromGluinoQuark_eta[iFSR],tree.truth_FSRFromGluinoQuark_phi[iFSR],tree.truth_FSRFromGluinoQuark_e[iFSR])
    # Find quark which emitted this FSR and get its parentBarcode
    for parton in QuarksFromGluinos:
      if parton.barcode == tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR]:
        FSRsFromGluinos[iFSR].set_gluino_barcode(parton.get_gluino_barcode())
        FSRsFromGluinos[iFSR].set_pdgid(parton.get_pdgid())
    FSRsFromGluinos[iFSR].set_barcode(tree.truth_FSRFromGluinoQuark_barcode[iFSR])
    FSRsFromGluinos[iFSR].set_quark_barcode(tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR])

  # Rename array with partons to be matched to reco jets
  Partons = QuarksFromGluinos
  FSRs    = FSRsFromGluinos

  #####################################################
  # Match reco jets to closest parton
  #####################################################

  Assigments = {
    # place holder with temporary values
    'source' : {'gmass' : 0, 'eta': 0, 'mass': 0, 'phi': 0, 'pt' : 0, 'mask': True},
    #'source' : {'eta': 0, 'mass': 0, 'phi': 0, 'pt' : 0, 'mask': True},
    # jet index for each particle b,q1,q2 (if no matching then -1) and mask set temporarily to True
    'g1'     : {'q1':-1, 'q2':-1, 'q3':-1, 'f1':0, 'f2':0, 'f3':0, 'mask': True},
    'g2'     : {'q1':-1, 'q2':-1, 'q3':-1, 'f1':0, 'f2':0, 'f3':0, 'mask': True},
    'normweight' : {'normweight':tree.normweight},
  }

  matcher = RPVMatcher(Jets = SelectedJets, Partons = QuarksFromGluinos, FSRs = FSRsFromGluinos)
  if Debug:
    matcher.set_property('Debug', True) # Temporary
  matcher.set_property('MatchingCriteria', MatchingCriteria)
  matched_jets = matcher.match()

  # Fill Assigments (info for matched jets)
  for jet_index, jet in enumerate(matched_jets):
    if jet.is_matched():
      Assigments = make_assigments(Assigments, gBarcodes, jet.get_match_gluino_barcode(), jet.get_match_pdgid(), qPDGIDs, matched_jets, jet_index)
  else:
    print('ERROR: Matching criteria not recognized, exiting')
    sys.exit(1)
 
  n_matched_jets = sum([1 if jet.is_matched() else 0 for jet in matched_jets])
  if n_matched_jets == 6:
    matchedEventNumbers.append(tree.eventNumber)
    matchedEvents +=1

  # Protection: make sure the same jet was not matched to two partons (if appropriate)
  JetIndexes = [] # indexes of jets matched to partons
  for g in ['g1','g2']:
    for key in Assigments[g]:
      if key == 'mask' or 'f' in key: continue
      index = Assigments[g][key]
      if index != -1:
        if index not in JetIndexes:
          JetIndexes.append(index)
        else:
          log.warning('Jet index ({}) was assigned to more than one parton!'.format(index))

  # Create arrays with jet info (extend Assigments with jet reco info)
  for case in Structure['all']['source']['cases']:
     array = []
     if case == 'gmass':
       array.append(gmass)
     else:
       ## Get sum_energy (sum energy from every real jet)
       #totalEnergy = 0
       #for j in SelectedJets:
       #  totalEnergy += j.E()
       ## Divide each four-vector by total jet energy
       #SelectedJets = [jet*(1/totalEnergy) for jet in SelectedJets]
       for j in SelectedJets:
         if case == 'eta':
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

  # See if gluinos were fully reconstructed (i.e. each decay particle matches a jet)
  for g in ['g1', 'g2']:
    TempMask = True
    for key in Assigments[g]:
      if key == 'mask' or 'f' in key: continue
      if Assigments[g][key] == -1: TempMask = False
    Assigments[g]['mask'] = TempMask

  # Compare reconstructed gluino mass with true gluino mass
  MultipleJetsMatchingAQuark = False
  AllMatchedJetsIndexes      = []
  for ig in ['g1', 'g2']: # loop over gluinos
    if Assigments[ig]['mask']: # fully reconstructable gluino (every quark matches a jet)
      Jets2sum   = []
      JetIndexes = []
      for key in ['q1', 'q2', 'q3']:
        jIndex = Assigments[ig][key]
        if jIndex not in JetIndexes:
          JetIndexes.append(jIndex)
          Jets2sum.append(SelectedJets[jIndex])
        if jIndex not in AllMatchedJetsIndexes:
          AllMatchedJetsIndexes.append(jIndex)
        else:
          MultipleJetsMatchingAQuark = True
      # Sum assigned jets
      if len(Jets2sum) == 3:
        JetsSum = Jets2sum[0] + Jets2sum[1] + Jets2sum[2]
      elif len(Jets2sum) == 2:
        JetsSum = Jets2sum[0] + Jets2sum[1]
      else:
        JetsSum = Jets2sum[0]
      gReco   = JetsSum.M()
      gTruth  = gmass
      if gTruth not in hRecoMasses:
        print('MC channel number: {}'.format(tree.mcChannelNumber))
        print('Event number: {}'.format(tree.eventNumber))
      hRecoMasses[gTruth].Fill(gReco)
      hGluinoMassDiff.Fill(gReco-gTruth)

  if MultipleJetsMatchingAQuark:
    multipleQuarkMatchingEvents += 1

  # Split dataset b/w traning and testing (if requested)
  if SplitDataset4Training:
    if counter in EventNumbers4Training:
      if ProduceTrainingDataset:
        trainingCounter += 1
        # Add data to the h5 training file
        for key in Structure['training']:
          for case in Structure['training'][key]['cases']:
            Datasets['training'][key+'_'+case][trainingCounter] = Assigments[key][case]
    elif counter in EventNumbers4Testing:
      if ProduceTestingDataset:
        testingCounter += 1
        # Add data to the h5 testing file
        for key in Structure['testing']:
          for case in Structure['testing'][key]['cases']:
            Datasets['testing'][key+'_'+case][testingCounter] = Assigments[key][case]
    else:
      if not ForceHalf:
        log.error('Event is simultaneously not considered for training nor for testing, exiting')
        sys.exit(1)
  else: # Add data to a single h5 file
    for key in Structure['all']:
      for case in Structure['all'][key]['cases']:
        Datasets[key+'_'+case][allCounter] = Assigments[key][case]

# Close input file
del tree
HF.close()

# Save histogram
outName = '{}GluinoMassDiff_{}.root'.format('LeaV3/' if 'Lea' in PATH else '', MatchingCriteria)
outFile = ROOT.TFile(outName,'RECREATE')
hGluinoMassDiff.Write()
outFile.Close()

# Reco gluino mass distributions
outName = 'ReconstructedGluinoMasses.root'.format()
outFile = ROOT.TFile(outName,'RECREATE')
for key,hist in hRecoMasses.items():
  hist.Write()
outFile.Close()
  
log.info('>>> ALL DONE <<<')
print('matching efficiency (percentage of events where 6 quarks are matched): {}'.format(matchedEvents/totalEvents))
print('Number of events where 6 quarks are matched: {}'.format(matchedEvents))
print('percentage of events having a quark matching several jets: {}'.format(multipleQuarkMatchingEvents/totalEvents))
for flav in [1,2,3,4,5,6]:
  if NquarksByFlavour[flav]!=0: print('Matching efficiency for quarks w/ abs(pdgID)=={}: {}'.format(flav,NmatchedQuarksByFlavour[flav]/NquarksByFlavour[flav]))
outFile = open('matchedEvents_{}.txt'.format(MatchingCriteria),'w')
for event in matchedEventNumbers:
  outFile.write(str(event)+'\n')
outFile.close()
