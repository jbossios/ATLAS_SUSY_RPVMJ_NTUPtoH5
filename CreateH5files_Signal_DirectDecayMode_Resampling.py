################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted for signal direct decay mode samples      #
#          Every mass point will have roughly the same stats                   #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch)                                      #
# Date:    September 2021                                                      #
#                                                                              #
################################################################################

# Settings
PATH                   = 'SignalInputs/MC16a/'
TreeName               = 'trees_SRRPV_'
ApplyEventSelections   = True
shuffleJets            = False
Debug                  = False
MinNjets               = 6
FlavourType            = 'All' # options: All (ALL+UDB+UDS), UDB, UDS

################################################################################
# DO NOT MODIFY (below this line)
################################################################################

# Global settings
dRcut       = 0.4
maxNjets    = 10
minJetPt    = 20 # to be safe but there seems to be no jet below 20GeV

# Create file with selected options
Config = open('Options.txt','w')
Config.write('ApplyEventSelections = {}\n'.format(ApplyEventSelections))
Config.write('ShuffleJets          = {}\n'.format(shuffleJets))
Config.write('MinNjets             = {}\n'.format(MinNjets))
Config.write('FlavourType          = {}\n'.format(FlavourType))
Config.write('dRcut                = {}\n'.format(dRcut))
Config.write('maxNjets             = {}\n'.format(maxNjets))
Config.write('minJetPt             = {}\n'.format(minJetPt))
Config.close()

# Dictorionary with number of events for each mass point
Nevents = {
  '900'  : 16867,
  '1000' : 17405,
  '1100' : 17772,
  '1200' : 17929,
  '1300' : 18225,
  '1400' : 110047,
  '1500' : 18523,
  '1600' : 18546,
  '1700' : 18633,
  '1800' : 28210,
  '1900' : 18745,
  '2000' : 18788,
  '2100' : 18886,
  '2200' : 28466,
  '2300' : 18926,
  '2400' : 17981,
  '2500' : 18986,
}
maxNevents4Training = Nevents[min(Nevents, key=Nevents.get)]

###############################
# Conventions
###############################
# q1 is the first matched quark found for the corresponding gluion
# q2 is the second matched quark found for the corresponding gluion
# q3 is the third matched quark found for the corresponding gluion
# g1 is the first parent gluino for first matched quark

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

# Choose datasets to be produced (if splitting is requested)
Datasets2Produce = ['training','testing']

##################
# Helpful classes
##################

class iJet(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)

class iParton(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.parentBarcode = -999
    self.barcode       = -999
    self.pdgID         = -999

##############################################################################################
# Find out how many events pass the event selections
##############################################################################################

allDSIDs = {
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
}

Flavours = [FlavourType] if FlavourType == 'UDB' or FlavourType == 'UDS' else ['ALL','UDS','UDB']
DSIDs = dict()
for dsid in allDSIDs:
  for flav in Flavours:
    if flav in allDSIDs[dsid]:
      DSIDs[dsid] = allDSIDs[dsid]
      break

tree = ROOT.TChain(TreeName)
for File in os.listdir(PATH):
  dsid = File.replace('.root','')
  if dsid not in DSIDs: continue # skip undesired DSID
  tree.Add(PATH+File)

# Split selected events b/w training and testing (if requested)
EventNumbers4Training = []
EventNumbers4Testing  = []

# Loop over events
nPassingEvents = 0
counter        = -1
for event in tree:
  counter += 1
  AllPassJets   = [iJet().SetPtEtaPhiE(tree.jet_pt[i],tree.jet_eta[i],tree.jet_phi[i],tree.jet_e[i]) for i in range(len(tree.jet_pt)) if tree.jet_passOR[i] and tree.jet_pt[i] > minJetPt]
  SelectedJets  = [AllPassJets[i] for i in range(min(maxNjets,len(AllPassJets)))] # Select leading n jets with n == min(maxNjets,njets)
  nJets         = len(SelectedJets)
  nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt)
  gmass         = tree.truth_parent_m[0]
  # Apply event selections
  passEventSelection = True
  if ApplyEventSelections:
    if nJets < MinNjets:  passEventSelection = False
    if nQuarksFromGs !=6: passEventSelection = False
  if not passEventSelection: continue # skip event
  prob = maxNevents4Training/Nevents[str(int(gmass))]
  if random.random() < prob: # use this event for training
    EventNumbers4Training.append(counter)
  else: # use this event for testing
    EventNumbers4Testing.append(counter)
  nPassingEvents += 1
log.info('{} events were selected'.format(nPassingEvents))

nPassingTrainingEvents = len(EventNumbers4Training)
nPassingTestingEvents  = len(EventNumbers4Testing)

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
Types      = { 'mask':bool, 'q1':int, 'q2':int, 'q3':int }
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
    'g1'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingEvents,) },
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
    'g1'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingTrainingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingTrainingEvents,) },
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
    'g1'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingTestingEvents,) },
    'g2'     : { 'cases' : ['mask','q1','q2','q3'], 'shape' : (nPassingTestingEvents,) },
  },
}

# Function that assigns quarks to q1, q2 or q3
def getQuarkFlavour(pdgid,g,dictionary):
  if dictionary[g]['q1'] == 0:
    dictionary[g]['q1'] = pdgid
    qFlavour = 'q1'
  elif dictionary[g]['q2'] == 0:
    dictionary[g]['q2'] = pdgid
    qFlavour = 'q2'
  else:
    qFlavour = 'q3'
  return dictionary,qFlavour

# Create H5 file(s)
# split dataset into training and testing datasets
Groups      = dict()
Datasets    = dict()
for datatype in Datasets2Produce:
  outFileName = '{}SignalData_{}.h5'.format(FlavourType,datatype)
  log.info('Creating {}...'.format(outFileName))
  HF                 = h5py.File(outFileName, 'w')
  Groups[datatype]   = dict()
  Datasets[datatype] = dict()
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

# Loop over events
counter         = -1
allCounter      = -1
trainingCounter = -1
testingCounter  = -1
log.info('About to enter event loop')
ROOT.EnableImplicitMT(nthreads)
for event in tree:
  counter += 1
  # Find number of particles/jets
  # Select reco jets
  AllPassJets = []
  for ijet in range(len(tree.jet_pt)):
    if tree.jet_passOR[ijet] and tree.jet_pt[ijet] > minJetPt:
      jet = iJet()
      jet.SetPtEtaPhiE(tree.jet_pt[ijet],tree.jet_eta[ijet],tree.jet_phi[ijet],tree.jet_e[ijet])
      AllPassJets.append(jet)
  SelectedJets  = [AllPassJets[i] for i in range(min(maxNjets,len(AllPassJets)))] # Select leading n jets with n == min(maxNjets,njets)
  nJets         = len(SelectedJets)
  nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt)
  
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
  ForTraining = True # if False then event will be used for testing
  if counter in EventNumbers4Testing: ForTraining = False

  # Protection
  if nJets > maxNjets:
    log.fatal('More than {} jets were found ({}), fix me!'.format(maxNjets,nJets))
    sys.exit(1)
 
  # Remove pt ordering in the jet array (if requested)
  if shuffleJets:
    random.shuffle(SelectedJets)

  # Extract gluino mass
  gmass = tree.truth_parent_m[0]

  # Collect gluino barcodes
  gBarcodes = {'g1':0,'g2':0} # fill temporary values

  # Collect quark -> gluino associations
  qPDGIDs = { g : {'q1':0,'q2':0,'q3':0} for g in ['g1','g2']} # fill temporary values

  # Select quarks from gluinos
  QuarksFromGluinos = [iParton() for i in range(nQuarksFromGs)]
  for iquark in range(nQuarksFromGs):
    QuarksFromGluinos[iquark].SetPtEtaPhiE(tree.truth_QuarkFromGluino_pt[iquark],tree.truth_QuarkFromGluino_eta[iquark],tree.truth_QuarkFromGluino_phi[iquark],tree.truth_QuarkFromGluino_e[iquark])
    QuarksFromGluinos[iquark].parentBarcode = tree.truth_QuarkFromGluino_ParentBarcode[iquark]
    QuarksFromGluinos[iquark].pdgID         = tree.truth_QuarkFromGluino_pdgID[iquark]

  # Rename array with partons to be matched to reco jets
  Partons = QuarksFromGluinos

  # Match reco jets to closest parton
  Assigments = {
    # place holder with temporary values
    'source' : {'gmass' : 0, 'eta': 0, 'mass': 0, 'phi': 0, 'pt' : 0, 'mask': True},
    #'source' : {'eta': 0, 'mass': 0, 'phi': 0, 'pt' : 0, 'mask': True},
    # jet index for each particle b,q1,q2 (if no matching then -1) and mask set temporarily to True
    'g1'     : {'q1': -1, 'q2': -1, 'q3': -1, 'mask': True},
    'g2'     : {'q1': -1, 'q2': -1, 'q3': -1, 'mask': True},
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
    if dRmin < dRcut: # jet matches a parton from gluino
      # Find to which gluino (g1 or g2) it matches and determines if the parton is q1, q2 or q3 (see convention above)
      pdgid          = Partons[matchPartonIndex].pdgID
      # Assign to g1 or g2 and fill
      qParentBarcode = Partons[matchPartonIndex].parentBarcode
      if gBarcodes['g1'] == 0:
        gBarcodes['g1']            = qParentBarcode
        qPDGIDs,qFlavour           = getQuarkFlavour(pdgid,'g1',qPDGIDs)
        Assigments['g1'][qFlavour] = jetIndex
      else:
        if gBarcodes['g1'] == qParentBarcode:
          qPDGIDs,qFlavour           = getQuarkFlavour(pdgid,'g1',qPDGIDs)
          Assigments['g1'][qFlavour] = jetIndex
        else:
          qPDGIDs,qFlavour           = getQuarkFlavour(pdgid,'g2',qPDGIDs)
          Assigments['g2'][qFlavour] = jetIndex

  # Protection: make sure the same jet was not matched to two partons
  JetIndexes = [] # indexes of jets matched to partons
  for g in ['g1','g2']:
    for key in Assigments[g]:
      if key == 'mask': continue
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
  for g in ['g1','g2']:
    TempMask = True
    for key in Assigments[g]:
      if Assigments[g][key] == -1: TempMask = False
    Assigments[g]['mask'] = TempMask

  # Split dataset b/w traning and testing (if requested)
  if counter in EventNumbers4Training:
    trainingCounter += 1
    # Add data to the h5 training file
    for key in Structure['training']:
      for case in Structure['training'][key]['cases']:
        Datasets['training'][key+'_'+case][trainingCounter] = Assigments[key][case]
  elif counter in EventNumbers4Testing:
    testingCounter += 1
    # Add data to the h5 testing file
    for key in Structure['testing']:
      for case in Structure['testing'][key]['cases']:
        Datasets['testing'][key+'_'+case][testingCounter] = Assigments[key][case]
  else:
    log.error('Event is simultaneously not considered for training nor for testing, exiting')
    sys.exit(1)

# Close input file
del tree
HF.close()
  
log.info('>>> ALL DONE <<<')
