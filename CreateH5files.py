################################################################################
#                                                                              #
# Purpose: ROOT file -> H5 converted                                           #
#                                                                              #
# Authour: Jona Bossio (jbossios@cern.ch), Anthony Badea (abadea@cern.ch)      #
# Date:    10 May 2022                                                         #
#                                                                              #
################################################################################

# Imports
from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVMatcher
from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVParton
from ATLAS_SUSY_RPVMJ_JetPartonMatcher.rpv_matcher.rpv_matcher import RPVJet
import logging
import ROOT
import h5py
import os
import numpy as np
import random
import multiprocessing as mp
from glob import glob
import argparse
import sys
import json
random.seed(4)  # set the random seed for reproducibility

# global variables
logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
log = logging.getLogger('CreateH4Files')

def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inDir', required=True, help="Input directory of files")
    parser.add_argument('-o', '--outDir', default='./', help="Output directory for files")
    parser.add_argument('-v', '--version', default="0", help="Production version")
    parser.add_argument('-j', '--ncpu', default=1, type=int, help="Number of cores to use in multiprocessing pool.")
    parser.add_argument('--minJetPt', default=50, type=int, help="Minimum selected jet pt")
    parser.add_argument('--maxNjets', default=8, type=int, help="Maximum number of leading jets retained in h5 files")
    parser.add_argument('--minNjets', default=6, type=int, help="Minimum number of leading jets retained in h5 files")
    parser.add_argument('--nQuarksPerGluino', default=6, type=int, help="Number of quarks per gluino from signal model")
    parser.add_argument('--shuffleJets', action='store_true', help="Shuffle jets before saving")
    parser.add_argument('--matchingCriteria', default='RecomputeDeltaRvalues_drPriority', help='Choose matching criteria from: UseFTDeltaRvalues, RecomputeDeltaRvalues_ptPriority, RecomputeDeltaRvalues_drPriority')
    parser.add_argument('--doNotUseFSRs', action='store_true', help="Do not consider final state radiation (FSR) in jet-parton matching")
    parser.add_argument('--debug', action='store_true', help="Enable debug print statemetents")
    parser.add_argument('--combine', action='store_true', help="Only combine the list of h5 files. File name automatically handled.")
    parser.add_argument('--combineExcludedDSIDs',  nargs="+", help="List of DSIDs to exclude when combining h5 files")
    args = parser.parse_args()

    if args.debug:
        log.setLevel("DEBUG")

    # get input files and sum of weights
    input_files = handleInput(args.inDir)

    # make sure output directory exists
    if not os.path.isdir(args.outDir):
        os.makedirs(args.outDir)

    # if just combine
    if args.combine:
        # remove unwanted dsids
        input_files = [i for i in input_files if not any([j for j in args.combineExcludedDSIDs if j in i])]
        return combine_h5(input_files, args.outDir, args.version)

    # get sum of weights
    sum_of_weights = get_sum_of_weights(input_files)
    log.info('Sum of weights: {}'.format(sum_of_weights))

    # get list of ttrees using the first input file
    f = ROOT.TFile(input_files[0])
    treeNames = [key.GetName() for key in list(f.GetListOfKeys()) if "trees" in key.GetName()]
    f.Close()

    # prepare outdir
    if not os.path.isdir(args.outDir):
        os.makedirs(args.outDir)

    # make job configurations
    confs = []
    for treeName in treeNames:
        print(f"Including tree {treeName}")
        for inFileName in input_files:
            
            # understand file type
            sample = "Signal"
            if "dijet" in inFileName or "WithSW" in inFileName:
                sample = "Dijet"
            elif "data" in inFileName:
                sample = "Data"
            do_matching = sample == 'Signal'
            dsid = int(inFileName.split('user.')[1].split('.')[2])

            # create outfile tag
            tag = f"{treeName.strip('trees_')}_minJetPt{args.minJetPt}_minNjets{args.minNjets}_maxNjets{args.maxNjets}"
            if do_matching:
                criteriaTag = {'UseFTDeltaRvalues':'FTDR', 'RecomputeDeltaRvalues_ptPriority': 'RDR_pt', 'RecomputeDeltaRvalues_drPriority' : 'RDR_dr'}
                tag += f"_{criteriaTag[args.matchingCriteria]}"
            tag += f"_v{args.version}"

            confs.append({
                # input settings
                'inFileName': inFileName,
                'sum_of_weights': sum_of_weights[dsid],
                'sample' : sample,
                'treeName' : treeName,
                # output settings
                'outDir': args.outDir,
                'tag': tag,
                # jet settings
                'minJetPt': args.minJetPt,
                'maxNjets': args.maxNjets,
                'MinNjets': args.minNjets,
                'shuffleJets': args.shuffleJets,
                # matching settings
                'do_matching' : do_matching,
                'MatchingCriteria': args.matchingCriteria,
                'useFSRs': not args.doNotUseFSRs,
                'dRcut': 0.4,
                # global settings
                'Debug': args.debug
            })
    print(f"Number of jobs launching: {len(confs)}")        
    
    # save confs to json
    with open(os.path.join(args.outDir,'CreateH5files_confs.json'), 'w') as f:
        json.dump(confs, f, sort_keys=False, indent=4)
    
    # launch jobs
    if args.ncpu == 1:
        for conf in confs:
            process_files(conf)
    else:
        results = mp.Pool(args.ncpu).map(process_files, confs)


def handleInput(data):
    if os.path.isfile(data) and ".root" in os.path.basename(data):
        return [data]
    elif os.path.isfile(data) and ".txt" in os.path.basename(data):
        return sorted([line.strip() for line in open(data, "r")])
    elif os.path.isdir(data):
        return [os.path.join(data, i) for i in sorted(os.listdir(data))]
    elif "*" in data:
        return sorted(glob(data))
    return []


def get_sum_of_weights(file_list):
    # Get sum of weights from all input files (by DSID)
    sum_of_weights = {}  # sum of weights for each dsid
    for file_name in file_list:
        # Make sure it's a ROOT file
        if not file_name.endswith('.root'):
            continue
        # Get metadata from all files, even those w/ empty TTrees
        try:
            tfile = ROOT.TFile.Open(file_name)
        except OSError:
            raise OSError('{} can not be opened'.format(file_name))
        # Identify DSID for this file
        dsid = int(file_name.split('user.')[1].split('.')[2])
        # Get sum of weights from metadata
        metadata_hist = tfile.Get('MetaData_EventCount')
        if dsid not in sum_of_weights:
            # initial sum of weights
            sum_of_weights[dsid] = metadata_hist.GetBinContent(3)
        else:
            # initial sum of weights
            sum_of_weights[dsid] += metadata_hist.GetBinContent(3)
    return sum_of_weights


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
    if g_barcodes['g1'] == 0:  # not assigned yet to any quark parent barcode
        g_barcodes['g1'] = q_parent_barcode
        q_pdgid_dict, q_flavour = get_quark_flavour(pdgid, 'g1', q_pdgid_dict)
        if assigments['g1'][q_flavour] == -1:  # not assigned yet
            assigments['g1'][q_flavour] = jet_index
            assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
        else:  # if assigned already, pick up jet with largets pT
            if selected_jets[jet_index].Pt() > selected_jets[assigments['g1'][q_flavour]].Pt():
                assigments['g1'][q_flavour] = jet_index
                assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
    else:  # g1 was defined alredy (check if quark parent barcode agrees with it)
        if g_barcodes['g1'] == q_parent_barcode:
            q_pdgid_dict, q_flavour = get_quark_flavour(
                pdgid, 'g1', q_pdgid_dict)
            if assigments['g1'][q_flavour] == -1:  # not assigned yet
                assigments['g1'][q_flavour] = jet_index
                assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
            else:  # if assigned already, pick up jet with largets pT
                if selected_jets[jet_index].Pt() > selected_jets[assigments['g1'][q_flavour]].Pt():
                    assigments['g1'][q_flavour] = jet_index
                    assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
        else:
            q_pdgid_dict, q_flavour = get_quark_flavour(pdgid, 'g2', q_pdgid_dict)
            if assigments['g2'][q_flavour] == -1:  # not assigned yet
                assigments['g2'][q_flavour] = jet_index
                assigments['g2'][q_flavour.replace('q', 'f')] = pdgid
            else:  # if assigned already, pick up jet with largets pT
                if selected_jets[jet_index].Pt() > selected_jets[assigments['g2'][q_flavour]].Pt():
                    assigments['g2'][q_flavour] = jet_index
                    assigments['g2'][q_flavour.replace('q', 'f')] = pdgid
    return assigments


def process_files(settings):

    # Set structure of output H5 file
    Structure = {
        'source': ['eta', 'mask', 'mass', 'phi', 'pt', 'QGTaggerBDT'],
        'EventVars': ['HT', 'deta', 'djmass', 'minAvgMass', 'rowNo', 'normweight'],
    }

    if settings['do_matching']:

        # Collect info to know matching efficiency for each quark flavour
        quark_flavours = [1, 2, 3, 4, 5, 6]
        NquarksByFlavour = {flav: 0 for flav in quark_flavours}
        NmatchedQuarksByFlavour = {flav: 0 for flav in quark_flavours}

        # conventions:
        # q1 is the first matched quark found for the corresponding gluino (f1 is its pdgID)
        # q2 is the second matched quark found for the corresponding gluino (f2 is its pdgID)
        # q3 is the third matched quark found for the corresponding gluino (f3 is its pdgID)
        # g1 is the first parent gluino for first matched quark
        Structure['g1'] = ['mask', 'q1', 'q2', 'q3', 'f1', 'f2', 'f3']
        Structure['g2'] = ['mask', 'q1', 'q2', 'q3', 'f1', 'f2', 'f3']
        Structure['EventVars'].append('gmass')

        # Reconstructed mass by truth mass
        Masses = [100, 200, 300, 400] + [900 + i*100 for i in range(0, 17)]
        hRecoMasses = {mass: ROOT.TH1D(f'RecoMass_TruthMass{mass}', '', 300, 0, 3000) for mass in Masses}

        # Reconstructed gluino mass - true gluino mass
        hGluinoMassDiff = ROOT.TH1D('GluinoMassDiff', '', 10000, -10000, 10000)
        matchedEvents = 0
        multipleQuarkMatchingEvents = 0
        matchedEventNumbers = []

    # Initialize lists for each variable to be saved
    assigments_list = {key: {case: [] for case in cases}
                       for key, cases in Structure.items()}

    ##############################################################################################
    # Loop over events and fill the numpy arrays on each event
    ##############################################################################################

    # Create TChain using all input ROOT files
    tree = ROOT.TChain(settings["treeName"])
    tree.Add(settings["inFileName"])
    log.info(f'About to enter event loop for {settings["inFileName"]}')
    event_counter = 0
    for counter, event in enumerate(tree):
        log.debug('Processing eventNumber = {}'.format(tree.eventNumber))
        
        # Skip events with any number of electrons/muons
        if tree.nBaselineElectrons or tree.nBaselineMuons:
           continue

        # Skip events not passing event-level jet cleaning cut
        if not tree.DFCommonJets_eventClean_LooseBad:
           continue
        
        # Skip dijet events without truth branches because likely low pt cut designed to remove pileup removed these events
        if settings["sample"] == "Dijet" and not tree.GetBranchStatus("truth_jet_pt"):
            continue

        # Skip dijet events with truth cleaning for pileup
        if settings["sample"] == "Dijet" and len(tree.truth_jet_pt) >= 0 and len(tree.jet_pt) >= 2:
            try:
                if not (1.4*tree.truth_jet_pt[0] > (tree.jet_pt[0]+tree.jet_pt[1])/2):
                    continue
            except:
                print("Pileup cleaning based on truth and reco jet pts failed. Skipping event.")
                continue
            
        # Select reco jets
        AllPassJets = []
        for ijet in range(len(tree.jet_pt)):
            if tree.jet_pt[ijet] > settings['minJetPt']:
                jet = RPVJet()
                jet.SetPtEtaPhiE(tree.jet_pt[ijet], tree.jet_eta[ijet], tree.jet_phi[ijet], tree.jet_e[ijet])
                jet.set_qgtagger_bdt(tree.jet_JetQGTaggerBDT_score[ijet]) #tree.jet_QGTagger_bdt[ijet])
                if settings['do_matching'] and settings['MatchingCriteria'] == 'UseFTDeltaRvalues':
                    jet.set_matched_parton_barcode(int(tree.jet_deltaRcut_matched_truth_particle_barcode[ijet]))
                    jet.set_matched_fsr_barcode(int(tree.jet_deltaRcut_FSRmatched_truth_particle_barcode[ijet]))
                AllPassJets.append(jet)
        # select leading n jets with n == min(maxNjets, njets)
        SelectedJets = [AllPassJets[i] for i in range(min(settings['maxNjets'], len(AllPassJets)))]
        nJets = len(SelectedJets)
        if settings['do_matching']:
            nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt) if tree.GetBranchStatus("truth_QuarkFromGluino_pt") else 0
            nFSRsFromGs = len(tree.truth_FSRFromGluinoQuark_pt) if tree.GetBranchStatus("truth_FSRFromGluinoQuark_pt") else 0

        # Apply event selections
        passEventSelection = True
        if nJets < settings['MinNjets']:
            passEventSelection = False
        if settings['do_matching'] and nQuarksFromGs != 6:
            passEventSelection = False
        if not passEventSelection:
            continue  # skip event

        event_counter += 1

        # Protection
        if nJets > settings['maxNjets']:
            log.fatal(f'More than {settings["maxNjets"]} jets were found ({nJets}), fix me!')
            sys.exit(1)

        # Remove pt ordering in the jet array (if requested)
        if settings['shuffleJets']:
            random.shuffle(SelectedJets)

        # Extract gluino mass
        if settings['sample'] == 'Signal':
            for ipart in range(len(tree.truth_parent_m)):  # loop over truth particles
                if tree.truth_parent_pdgId[ipart] == 1000021:  # it's a gluino
                    gmass = tree.truth_parent_m[ipart]
                    break

        # Put place-holder values for each variable
        # set jet index for each particle q1,q2,q3 to -1 (i.e. no matching) and mask to True
        def init_value(case):
            if case == 'mask':
                return True
            if 'q' in case:
                return -1
            return 0
        Assigments = {key: {case: init_value(
            case) for case in cases} for key, cases in Structure.items()}

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
            # add extra (padding) jets to keep the number of jets fixed
            if nJets < settings['maxNjets']:
                for i in range(nJets, settings['maxNjets']):
                    if case != 'mask':
                        array.append(0.)
                    else:
                        array.append(False)
            Assigments['source'][case] = np.array(array)

        # Save event-level variables
        Assigments['EventVars']['rowNo'] = counter # row number of event
        Assigments['EventVars']['HT'] = ht
        Assigments['EventVars']['deta'] = SelectedJets[0].Eta() - SelectedJets[1].Eta()
        Assigments['EventVars']['djmass'] = (SelectedJets[0]+SelectedJets[1]).M()
        if settings['sample'] == 'Signal':
            Assigments['EventVars']['gmass'] = gmass
        Assigments['EventVars']['minAvgMass'] = tree.minAvgMass_jetdiff10_btagdiff10
        Assigments['EventVars']['normweight'] = tree.mcEventWeight * tree.pileupWeight * tree.weight_filtEff * tree.weight_kFactor * tree.weight_xs / settings['sum_of_weights']

        if settings['do_matching']:

            # Collect gluino barcodes
            gBarcodes = {'g1': 0, 'g2': 0}  # fill temporary values

            # Collect quark -> gluino associations
            qPDGIDs = {g: {'q1': 0, 'q2': 0, 'q3': 0}
                       for g in ['g1', 'g2']}  # fill temporary values

            # Select quarks from gluinos
            QuarksFromGluinos = [RPVParton() for i in range(nQuarksFromGs)]
            for iquark in range(nQuarksFromGs):
                QuarksFromGluinos[iquark].SetPtEtaPhiE(tree.truth_QuarkFromGluino_pt[iquark], tree.truth_QuarkFromGluino_eta[iquark],
                                                       tree.truth_QuarkFromGluino_phi[iquark], tree.truth_QuarkFromGluino_e[iquark])
                QuarksFromGluinos[iquark].set_gluino_barcode(
                    tree.truth_QuarkFromGluino_ParentBarcode[iquark])
                QuarksFromGluinos[iquark].set_barcode(
                    tree.truth_QuarkFromGluino_barcode[iquark])
                QuarksFromGluinos[iquark].set_pdgid(
                    tree.truth_QuarkFromGluino_pdgID[iquark])

            # Select FSR quarks from gluinos
            FSRsFromGluinos = [RPVParton() for i in range(nFSRsFromGs)]
            for iFSR in range(nFSRsFromGs):
                FSRsFromGluinos[iFSR].SetPtEtaPhiE(tree.truth_FSRFromGluinoQuark_pt[iFSR], tree.truth_FSRFromGluinoQuark_eta[iFSR],
                                                   tree.truth_FSRFromGluinoQuark_phi[iFSR], tree.truth_FSRFromGluinoQuark_e[iFSR])
                # Find quark which emitted this FSR and get its parentBarcode
                for parton in QuarksFromGluinos:
                    if parton.get_barcode() == tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR]:
                        FSRsFromGluinos[iFSR].set_gluino_barcode(
                            parton.get_gluino_barcode())
                        FSRsFromGluinos[iFSR].set_pdgid(parton.get_pdgid())
                FSRsFromGluinos[iFSR].set_barcode(
                    tree.truth_FSRFromGluinoQuark_barcode[iFSR])
                FSRsFromGluinos[iFSR].set_quark_barcode(
                    tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR])

            # Rename array with partons to be matched to reco jets
            Partons = QuarksFromGluinos
            FSRs = FSRsFromGluinos

            # Match reco jets to closest parton
            matcher = RPVMatcher(Jets=SelectedJets, Partons=QuarksFromGluinos)
            if settings['useFSRs']:
                matcher.add_fsrs(FSRsFromGluinos)
            if settings['Debug']:
                matcher.set_property('Debug', True)
            matcher.set_property('MatchingCriteria',
                                 settings['MatchingCriteria'])
            if settings['MatchingCriteria'] != "UseFTDeltaRvalues":
                matcher.set_property('DeltaRcut', settings['dRcut'])
            matched_jets = matcher.match()

            # Fill Assigments (info for matched jets)
            for jet_index, jet in enumerate(matched_jets):
                if jet.is_matched():
                    Assigments = make_assigments(Assigments, gBarcodes, jet.get_match_gluino_barcode(
                    ), jet.get_match_pdgid(), qPDGIDs, matched_jets, jet_index)

            # Check if fully matched
            n_matched_jets = sum(
                [1 if jet.is_matched() else 0 for jet in matched_jets])
            if n_matched_jets == 6:
                matchedEventNumbers.append(tree.eventNumber)
                matchedEvents += 1

            # See if gluinos were fully reconstructed (i.e. each decay particle matches a jet)
            for g in ['g1', 'g2']:
                TempMask = True
                for key in Assigments[g]:
                    if key == 'mask' or 'f' in key:
                        continue
                    if Assigments[g][key] == -1:
                        TempMask = False
                Assigments[g]['mask'] = TempMask

            # Compare reconstructed gluino mass with true gluino mass
            MultipleJetsMatchingAQuark = False
            AllMatchedJetsIndexes = []
            for ig in ['g1', 'g2']:  # loop over gluinos
                # fully reconstructable gluino (every quark matches a jet)
                if Assigments[ig]['mask']:
                    Jets2sum = []
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
                    gReco = JetsSum.M()
                    gTruth = gmass
                    if gTruth not in hRecoMasses:
                        print('MC channel number: {}'.format(
                            tree.mcChannelNumber))
                        print('Event number: {}'.format(tree.eventNumber))
                    hRecoMasses[gTruth].Fill(gReco)
                    hGluinoMassDiff.Fill(gReco-gTruth)

            if MultipleJetsMatchingAQuark:
                multipleQuarkMatchingEvents += 1

        # Add data to assigments_list
        for key in Structure:
            for case in Structure[key]:
                assigments_list[key][case].append(Assigments[key][case])

    # Close input file
    del tree

    # Create H5 file
    outFileName = os.path.join(settings["outDir"], os.path.basename(settings["inFileName"]).replace(".root", f"_{settings['tag']}.h5"))
    log.info('Creating {}'.format(outFileName))
    with h5py.File(outFileName, 'w') as HF:
        Groups, Datasets = dict(), dict()
        for key in Structure:
            Groups[key] = HF.create_group(key)
            for case in Structure[key]:
                Datasets[key+'_'+case] = Groups[key].create_dataset(case, data=assigments_list[key][case])

    if settings['do_matching']:

        # Save histogram
        outFile = ROOT.TFile(os.path.join(settings["outDir"], f"GluinoMassDiff_{settings['tag']}.root"), 'RECREATE')
        hGluinoMassDiff.Write()
        outFile.Close()

        # Reco gluino mass distributions
        outFile = ROOT.TFile(os.path.join(settings["outDir"], f"ReconstructedGluinoMasses_{settings['tag']}.root"), 'RECREATE')
        for key, hist in hRecoMasses.items():
            hist.Write()
        outFile.Close()

        # print matching efficiency
        log.info(f'matching efficiency (percentage of events where 6 quarks are matched): {matchedEvents/event_counter}')
        log.info(f'Number of events where 6 quarks are matched: {matchedEvents}')
        log.info(f'percentage of events having a quark matching several jets: {multipleQuarkMatchingEvents/event_counter}')
        for flav in quark_flavours:
            if NquarksByFlavour[flav] != 0:
                log.info(f'Matching efficiency for quarks w/ abs(pdgID)=={flav}: {NmatchedQuarksByFlavour[flav]/NquarksByFlavour[flav]}')

        # saving matching settings
        with open(os.path.join(settings["outDir"], f"matchedEvents_{settings['tag']}.root"),"w") as outFile:
            for event in matchedEventNumbers:
                outFile.write(str(event)+'\n')

    log.info('>>> ALL DONE <<<')


def combine_h5(inFileList, outDir, version):

    # inherit structure from first file
    with h5py.File(inFileList[0],"r") as f:
        Structure = {key:list(f[key].keys()) for key in f.keys()}

    # make and populate assignments_list
    assigments_list = {key: {case: [] for case in cases} for key, cases in Structure.items()}
    tags = []
    for iF, inFile in enumerate(inFileList):
        log.info(f"File {iF}/{len(inFileList)}")
        tags.append(inFile.split("trees_")[-1].strip(".h5"))
        with h5py.File(inFile,"r") as f:
            for key in Structure:
                for case in Structure[key]:
                    if len(f[key][case]):
                        assigments_list[key][case].append(np.array(f[key][case]))
                    else:
                        print(f"{inFile} has zero entries in {case}")
    
    # create outfilename
    outFileName = ""
    if any([i for i in list(range(504513,504552+1)) if str(i) in inFileList[0]]):
        outFileName = "gg_rpv"
    elif any([i for i in list(range(364700,364712+1)) if str(i) in inFileList[0]]):
        outFileName = "jetjet_JZWithSW"
    elif "data" in inFileList[0]:
        outFileName = "data"
    # check the tag and update output name
    tags = list(set(tags))
    if len(tags) > 1:
        log.error(f"You are combining files with more than one tag: {tags}")
        return
    else:
        outFileName += f"_{tags[0]}"
    # add combine tag
    outFileName += f"_c{version}"
    # add h5 tag
    outFileName += ".h5"
    # add output directory
    outFileName = os.path.join(outDir, outFileName)
    
    # create combined file
    log.info(f"Combining into {outFileName}")
    with h5py.File(outFileName,"w") as HF:
        Groups, Datasets = dict(), dict()
        for key in Structure:
            Groups[key] = HF.create_group(key)
            for case in Structure[key]:
                Datasets[key+'_'+case] = Groups[key].create_dataset(case, data=np.concatenate(assigments_list[key][case]))

    # save list of files to txt file
    txtFileName = outFileName.replace(".h5",".txt")
    log.info(f"Documenting used log files in {txtFileName}")
    with open(txtFileName,"w") as f:
        f.write('\n'.join(inFileList))

    log.info("Done!")

if __name__ == '__main__':
    main()
