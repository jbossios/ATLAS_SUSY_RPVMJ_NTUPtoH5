################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted                                           #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch)                                      #
# Date:    10 May 2022                                                         #
#                                                                              #
################################################################################

# Imports
import ROOT
import h5py
import os
import numpy as np
import random
random.seed(4) # set the random seed for reproducibility
import logging

# Global settings
TTREE_NAME = 'trees_SRRPV_'
PATH_SIGNALS = 'SignalInputs/MC16a_21_2_173_0_with_fixed_normweight/'
PATH_DIJETS = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets/PROD1/'


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


def process_files(input_files, settings):

    from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVJet
    from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVParton
    from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVMatcher
    
    # User settings
    Version = settings['Version']
    MinNjets = settings['MinNjets']
    maxNjets = settings['maxNjets']
    minJetPt = settings['minJetPt']
    shuffleJets = settings['shuffleJets']
    Debug = settings['Debug']
    sample = settings['sample']
    log = settings['Logger']
    outDir = settings['outDir']
    do_matching = False
    if sample == 'Signal':
      MatchingCriteria = settings['MatchingCriteria']
      dRcut = settings['dRcut']
      useFSRs = settings['useFSRs']
      MassPoints = settings['MassPoints']
      FlavourType = settings['FlavourType']
      do_matching = True
   
    # Create TChain using all input ROOT files 
    tree = ROOT.TChain(TTREE_NAME)
    for input_file in input_files:
        tree.Add(input_file)
    
    # Collect info to know matching efficiency for each quark flavour
    if do_matching:
        quark_flavours = [1, 2, 3, 4, 5, 6]
        NquarksByFlavour = {flav: 0 for flav in quark_flavours}
        NmatchedQuarksByFlavour = {flav: 0 for flav in quark_flavours}
    
    # Set structure of output H5 file
    Structure = {
        'source': ['eta', 'mask', 'mass', 'phi', 'pt', 'QGTaggerBDT'],
        'normweight': ['normweight'],
        'EventVars': ['HT', 'deta', 'djmass', 'minAvgMass'],
    }
    if do_matching:
        # conventions:
        # q1 is the first matched quark found for the corresponding gluino (f1 is its pdgID)
        # q2 is the second matched quark found for the corresponding gluino (f2 is its pdgID)
        # q3 is the third matched quark found for the corresponding gluino (f3 is its pdgID)
        # g1 is the first parent gluino for first matched quark
        Structure['g1'] = ['mask', 'q1', 'q2', 'q3', 'f1', 'f2', 'f3']
        Structure['g2'] = ['mask', 'q1', 'q2', 'q3', 'f1', 'f2', 'f3']
        Structure['EventVars'].append('gmass')
    
    # Book histograms
    if do_matching:
        # Reconstructed mass by truth mass
        Masses = [900 + i*100 for i in range(0, 17)]
        hRecoMasses = {mass: ROOT.TH1D(f'RecoMass_TruthMass{mass}', '', 300, 0, 3000) for mass in Masses}
        
        # Reconstructed gluino mass - true gluino mass
        hGluinoMassDiff = ROOT.TH1D('GluinoMassDiff', '', 10000, -10000, 10000)
        matchedEvents = 0
        multipleQuarkMatchingEvents = 0
        matchedEventNumbers = []
    
    # Initialize lists for each variable to be saved
    assigments_list = {key: {case: [] for case in cases} for key, cases in Structure.items()}
   
    ##############################################################################################
    # Loop over events and fill the numpy arrays on each event
    ##############################################################################################

    log.info('About to enter event loop')
    event_counter = 0
    for counter, event in enumerate(tree):
        log.debug('Processing eventNumber = {}'.format(tree.eventNumber))

        # Skip events with any number of electrons/muons
        #if tree.nBaselineElectrons or tree.nBaselineMuons:  # Temporary (uncomment once I have new samples)
        #    continue

        # Skip events not passing event-level jet cleaning cut
        #if not tree.DFCommonJets_eventClean_LooseBad:  # Temporary (uncomment once I have new samples)
        #    continue

        # Select reco jets
        AllPassJets = []
        for ijet in range(len(tree.jet_pt)):
            #if tree.jet_passOR[ijet] and tree.jet_isSig[ijet] and tree.jet_pt[ijet] > minJetPt:
            if tree.jet_pt[ijet] > minJetPt:
                jet = RPVJet()
                jet.SetPtEtaPhiE(tree.jet_pt[ijet], tree.jet_eta[ijet], tree.jet_phi[ijet], tree.jet_e[ijet])
                jet.set_qgtagger_bdt(tree.jet_QGTagger_bdt[ijet])
                if do_matching and MatchingCriteria == 'UseFTDeltaRvalues':
                    jet.set_matched_parton_barcode(int(tree.jet_deltaRcut_matched_truth_particle_barcode[ijet]))
                    jet.set_matched_fsr_barcode(int(tree.jet_deltaRcut_FSRmatched_truth_particle_barcode[ijet]))
                AllPassJets.append(jet)
        SelectedJets = [AllPassJets[i] for i in range(min(maxNjets, len(AllPassJets)))]  # select leading n jets with n == min(maxNjets, njets)
        nJets = len(SelectedJets)
        if do_matching:
            nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt)
            nFSRsFromGs = len(tree.truth_FSRFromGluinoQuark_pt)
    
        # Apply event selections
        passEventSelection = True
        if nJets < MinNjets:
            passEventSelection = False
        if do_matching and nQuarksFromGs !=6:
            passEventSelection = False
        if not passEventSelection:
            continue  # skip event
    
        event_counter += 1
    
        # Protection
        if nJets > maxNjets:
            log.fatal(f'More than {maxNjets} jets were found ({nJets}), fix me!')
            sys.exit(1)
     
        # Remove pt ordering in the jet array (if requested)
        if shuffleJets:
            random.shuffle(SelectedJets)
    
        # Extract gluino mass
        if sample == 'Signal':
            for ipart in range(len(tree.truth_parent_m)): # loop over truth particles
                if tree.truth_parent_pdgId[ipart] == 1000021: # it's a gluino
                    gmass = tree.truth_parent_m[ipart]
                    break
   
        if do_matching: 
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
                    if parton.get_barcode() == tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR]:
                        FSRsFromGluinos[iFSR].set_gluino_barcode(parton.get_gluino_barcode())
                        FSRsFromGluinos[iFSR].set_pdgid(parton.get_pdgid())
                FSRsFromGluinos[iFSR].set_barcode(tree.truth_FSRFromGluinoQuark_barcode[iFSR])
                FSRsFromGluinos[iFSR].set_quark_barcode(tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR])
    
            # Rename array with partons to be matched to reco jets
            Partons = QuarksFromGluinos
            FSRs    = FSRsFromGluinos
    
        # Put place-holder values for each variable
        # set jet index for each particle q1,q2,q3 to -1 (i.e. no matching) and mask to True
        def init_value(case):
            if case == 'mask':
                return True
            if 'q' in case:
                return -1
            return 0
        Assigments = {key: {case: init_value(case) for case in cases} for key, cases in Structure.items()}

        # Match reco jets to closest parton
        if do_matching:
            matcher = RPVMatcher(Jets = SelectedJets, Partons = QuarksFromGluinos)
            if useFSRs:
                matcher.add_fsrs(FSRsFromGluinos)
            if Debug:
                matcher.set_property('Debug', True)
            matcher.set_property('MatchingCriteria', MatchingCriteria)
            matched_jets = matcher.match()
    
            # Fill Assigments (info for matched jets)
            for jet_index, jet in enumerate(matched_jets):
                if jet.is_matched():
                    Assigments = make_assigments(Assigments, gBarcodes, jet.get_match_gluino_barcode(), jet.get_match_pdgid(), qPDGIDs, matched_jets, jet_index)
    
            # Check if fully matched 
            n_matched_jets = sum([1 if jet.is_matched() else 0 for jet in matched_jets])
            if n_matched_jets == 6:
                matchedEventNumbers.append(tree.eventNumber)
                matchedEvents +=1
    
        # Create arrays with jet info (extend Assigments with jet reco info)
        ht_calculated = False
        ht = 0
        for case in Structure['source']:
             array = []
             for j in SelectedJets:
                 if not ht_calculated:
                     ht += j.Pt()
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
                 elif case == 'QGTaggerBDT':
                     array.append(j.get_qgtagger_bdt())
             ht_calculated = True
             if nJets < maxNjets: # add extra (padding) jets to keep the number of jets fixed
                 for i in range(nJets, maxNjets):
                     if case != 'mask':
                         array.append(0.)
                     else:
                         array.append(False)
             Assigments['source'][case] = np.array(array)
        
        # Save event-level variables
        Assigments['EventVars']['HT'] = ht
        Assigments['EventVars']['deta'] = SelectedJets[0].Eta()-SelectedJets[1].Eta()
        Assigments['EventVars']['djmass'] = (SelectedJets[0]+SelectedJets[1]).M()
        if sample == 'Signal':
            Assigments['EventVars']['gmass'] = gmass
        Assigments['EventVars']['minAvgMass'] = 0 #tree.minAvgMass  # FIXME: need to update branch name!
        Assigments['normweight']['normweight'] = tree.normweight

        if do_matching: 
            # See if gluinos were fully reconstructed (i.e. each decay particle matches a jet)
            for g in ['g1', 'g2']:
                TempMask = True
                for key in Assigments[g]:
                    if key == 'mask' or 'f' in key: continue
                    if Assigments[g][key] == -1: TempMask = False
                Assigments[g]['mask'] = TempMask
    
            # Compare reconstructed gluino mass with true gluino mass
            MultipleJetsMatchingAQuark = False
            AllMatchedJetsIndexes = []
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
    
        # Add data to assigments_list
        for key in Structure:
            for case in Structure[key]:
                assigments_list[key][case].append(Assigments[key][case])
  
    ########################################################
    # Create H5 file
    ########################################################
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    if sample == 'Signal':
        outFileName = os.path.join(outDir,'Signal_{}_{}_full_{}.h5'.format(MassPoints, '_'.join(FlavourType.split('+')), Version))
    else:  # Dijets
        input_file_name = input_files[0].split('/')[-1]
        input_file_name = input_file_name.replace('.trees.root', '')
        outFileName = os.path.join(outDir,'Dijets_{}_{}.h5'.format(Version, input_file_name))
    log.info('Creating {}...'.format(outFileName))
    HF = h5py.File(outFileName, 'w')
    Groups, Datasets = dict(), dict()
    for key in Structure:
        Groups[key] = HF.create_group(key)
        for case in Structure[key]:
            if key == 'source':
                Datasets[key+'_'+case] = Groups[key].create_dataset(case, data=assigments_list[key][case])
            else:
                Datasets[key+'_'+case] = Groups[key].create_dataset(case, data=assigments_list[key][case])

    # Close input file
    del tree
    HF.close()

    if do_matching:    
        # Save histogram
        outName = 'GluinoMassDiff_{}_{}_{}_{}.root'.format(MassPoints, MatchingCriteria, Version, '_'.join(FlavourType.split('+')))
        outFile = ROOT.TFile(outName,'RECREATE')
        hGluinoMassDiff.Write()
        outFile.Close()
        
        # Reco gluino mass distributions
        outName = 'ReconstructedGluinoMasses_{}_{}_{}_{}.root'.format(MassPoints, MatchingCriteria, Version, '_'.join(FlavourType.split('+')))
        outFile = ROOT.TFile(outName,'RECREATE')
        for key,hist in hRecoMasses.items():
          hist.Write()
        outFile.Close()
      
    log.info('>>> ALL DONE <<<')
    if do_matching:
        print('matching efficiency (percentage of events where 6 quarks are matched): {}'.format(matchedEvents/event_counter))
        print('Number of events where 6 quarks are matched: {}'.format(matchedEvents))
        print('percentage of events having a quark matching several jets: {}'.format(multipleQuarkMatchingEvents/event_counter))
        for flav in quark_flavours:
          if NquarksByFlavour[flav]!=0: print('Matching efficiency for quarks w/ abs(pdgID)=={}: {}'.format(flav,NmatchedQuarksByFlavour[flav]/NquarksByFlavour[flav]))
        outFile = open('matchedEvents_{}_{}_{}_{}.txt'.format(MassPoints, MatchingCriteria, Version, '_'.join(FlavourType.split('+'))),'w')
        for event in matchedEventNumbers:
          outFile.write(str(event)+'\n')
        outFile.close()


def get_dijet_files(settings):
    input_files = []
    for folder in os.listdir(settings['PATH']):
        path = os.path.join(settings['PATH'], folder) #f'{settings["PATH"]}{folder}/'
        if os.path.isdir(path):
            for input_file in os.listdir(path):
                if os.path.basename(input_file).endswith('.root') and "expanded" in input_file: 
                    input_files.append(os.path.join(path,input_file))
        else:
            if "expanded" in path:
                input_files.append(path)
    return input_files


def get_signal_files(settings):
    dsids = {  # all available DSIDs
        "504513": "GG_rpv_UDB_900",
        "504514": "GG_rpv_UDB_1000",
        "504515": "GG_rpv_UDB_1100",
        "504516": "GG_rpv_UDB_1200",
        "504517": "GG_rpv_UDB_1300",
        "504518": "GG_rpv_UDB_1400",
        "504519": "GG_rpv_UDB_1500",
        "504520": "GG_rpv_UDB_1600",
        "504521": "GG_rpv_UDB_1700",
        "504522": "GG_rpv_UDB_1800",
        "504523": "GG_rpv_UDB_1900",
        "504524": "GG_rpv_UDB_2000",
        "504525": "GG_rpv_UDB_2100",
        "504526": "GG_rpv_UDB_2200",
        "504527": "GG_rpv_UDB_2300",
        "504528": "GG_rpv_UDB_2400",
        "504529": "GG_rpv_UDB_2500",
        "504534": "GG_rpv_UDS_900",
        "504535": "GG_rpv_UDS_1000",
        "504536": "GG_rpv_UDS_1100",
        "504537": "GG_rpv_UDS_1200",
        "504538": "GG_rpv_UDS_1300",
        "504539": "GG_rpv_UDS_1400",
        "504540": "GG_rpv_UDS_1500",
        "504541": "GG_rpv_UDS_1600",
        "504542": "GG_rpv_UDS_1700",
        "504543": "GG_rpv_UDS_1800",
        "504544": "GG_rpv_UDS_1900",
        "504545": "GG_rpv_UDS_2000",
        "504546": "GG_rpv_UDS_2100",
        "504547": "GG_rpv_UDS_2200",
        "504548": "GG_rpv_UDS_2300",
        "504549": "GG_rpv_UDS_2400",
        "504550": "GG_rpv_UDS_2500",
        "504551": "GG_rpv_ALL_1800",
        "504552": "GG_rpv_ALL_2200",
    }

    # Use only the above samples of the requested flavour (UDS, UDS+UDB, UDB, ALL)
    Flavours = []
    if settings['FlavourType'] == 'All':
        Flavours = ['ALL','UDS','UDB']
    if settings['FlavourType'] == 'ALL':
        Flavours = ['ALL']
    if 'UDB' in settings['FlavourType']:
        Flavours.append('UDB')
    if 'UDS' in settings['FlavourType']:
        Flavours.append('UDS')

    # Set samples to use based on the requested mass point(s)
    if settings['MassPoints'] == 'All':
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if sample.split('_')[2] in Flavours}
    elif 'AllExcept' in settings['MassPoints']:
        exclude = MassPoints.split('AllExcept')[1]
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if exclude not in sample and sample.split('_')[2] in Flavours}
    elif settings['MassPoints'] == 'Low':
        masses = ['900', '1000', '1100', '1200', '1300']
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if sample.split('_')[3] in masses and sample.split('_')[2] in Flavours}
    elif settings['MassPoints'] == 'Intermediate':
        masses = ['1400', '1500', '1600', '1700', '1800', '1900']
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if sample.split('_')[3] in masses and sample.split('_')[2] in Flavours}
    elif settings['MassPoints'] == 'IntermediateWo1400':
        masses = ['1500', '1600', '1700', '1800', '1900']
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if sample.split('_')[3] in masses and sample.split('_')[2] in Flavours}
    elif settings['MassPoints'] == 'High':
        masses = ['2000', '2100', '2200', '2300', '2400', '2500']
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if sample.split('_')[3] in masses and sample.split('_')[2] in Flavours}
    else:  # individual mass
        dsids_to_use = {dsid: sample for dsid, sample in dsids.items() if settings['MassPoints'] in sample and sample.split('_')[2] in Flavours}

    # Prepare list of input files
    input_files = []
    for root_file in os.listdir(settings['PATH']):
        if '.root' not in root_file: continue  # skip non-TFile files
        dsid = root_file.replace('.root', '')
        if dsid not in dsids_to_use: continue  # skip undesired DSID
        input_file = f'{settings["PATH"]}{root_file}'
        input_files.append(input_file)
    return input_files


def set_settings(args):
    # User settings
    settings = {
        'useFSRs': not args.doNotUseFSRs,
        'Version': args.version,
        'maxNjets': int(args.maxNjets),
        'minJetPt': int(args.minJetPt),
        'FlavourType': args.flavour,
        'MassPoints': args.masses,
        'MatchingCriteria': args.matchingCriteria,
        'MinNjets': int(args.minNjets),
        'shuffleJets': args.shuffleJets,
        'Debug': args.debug,
        'PATH': args.path,
        'sample': args.sample,
        'dRcut': 0.4,
        'Logger': args.logger,
        'outDir' : args.outDir,
    }
    
    # Create file with selected options
    Config = open('Options_{}_{}_{}.txt'.format(settings['Version'], settings['MassPoints'], '_'.join(settings['FlavourType'].split('+'))),'w')
    for key, value in settings.items():
        if settings['sample'] == 'Dijets':
            skip_on_dijets = ['MatchingCriteria', 'MassPoints', 'dRcut', 'useFSRs', 'FlavourType']
            if key in skip_on_dijets:
                continue  # skip settings that only make sense on signals
        Config.write(f'{key} = {value}\n')
    Config.close()

    return settings


if __name__ == '__main__':

    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '--v', action='store', dest='version', default='')
    parser.add_argument('--sample', action='store', dest='sample', default='Signal')
    parser.add_argument('--pTcut', action='store', dest='minJetPt', default='')
    parser.add_argument('--maxNjets', action='store', dest='maxNjets', default='')
    parser.add_argument('--minNjets', action='store', dest='minNjets', default='6')
    parser.add_argument('--nQuarksPerGluino', action='store', dest='nQuarksPerGluino', default='6')
    parser.add_argument('--shuffleJets', action='store_true', dest='shuffleJets', default=False)
    parser.add_argument('--matchingCriteria', action='store', dest='matchingCriteria', default='RecomputeDeltaRvalues_drPriority',
        help='Choose matching criteria from: UseFTDeltaRvalues, RecomputeDeltaRvalues_ptPriority, RecomputeDeltaRvalues_drPriority')
    parser.add_argument('--doNotUseFSRs', action='store_true', dest='doNotUseFSRs', default=False)
    parser.add_argument('--masses', action='store', dest='masses', default='1400',
        help="Choose set of masses from: All, AllExceptX (with X any mass), X (with X any mass), Low, Intermediate, IntermediateWo1400, High")
    parser.add_argument('--flavour', action='store', dest='flavour', default='UDB+UDS', help='UDB+UDS, or UDB, or UDS, or All (ALL+UDB+UDS)')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False)
    parser.add_argument('--outDir', default='./', help="Output directory for files.")
    parser.add_argument('--ncpu', default=1, type=int, help="Number of cores to use in multiprocessing pool.")
    args = parser.parse_args()
    
    # Protections
    import sys
    if not args.version:
        print('ERROR: version (--version OR -v) not provided, exiting')
        parser.print_help()
        sys.exit(1)
    if not args.maxNjets:
        print('ERROR: maxNjets (--maxNjets) not provided, exiting')
        parser.print_help()
        sys.exit(1)
    if not args.minJetPt:
        print('ERROR: minimum jet pT (--pTcut) not provided, exiting')
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
    log = logging.getLogger('CreateH4Files')
    if args.debug: log.setLevel("DEBUG")
    args.logger = log
    
    # Find input files
    # signal: will create a TChain using all input files
    # dijets: will run on each input file separately
    if args.sample == 'Signal':
        args.path = PATH_SIGNALS
        settings = set_settings(args)
        input_files = get_signal_files(settings)
        process_files(input_files, settings)
    elif args.sample == 'Dijets':
        args.path = PATH_DIJETS
        settings = set_settings(args)
        input_files = get_dijet_files(settings)
        from multiprocessing import Pool
        from functools import partial
        input_files_listed = [[input_file] for input_file in input_files]
        with Pool(args.ncpu) as p:
            process_files_partial = partial(process_files, settings = settings)
            p.map(process_files_partial, input_files_listed)
    else:
        print('ERROR: sample=={settings["sample"]} not supported yet')
