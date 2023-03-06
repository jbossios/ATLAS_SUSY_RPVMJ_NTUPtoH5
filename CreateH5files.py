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
import matplotlib.pyplot as plt

# global variables
logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
log = logging.getLogger('CreateH4Files')

def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inDir', required=True, help="Input directory of files")
    parser.add_argument('-o', '--outDir', default='./', help="Output directory for files")
    parser.add_argument('-v', '--version', default="0", help="Production version")
    parser.add_argument('-j', '--ncpu', default=1, type=int, help="Number of cores to use in multiprocessing pool")
    parser.add_argument('-n', '--normalization_denominator', default=None, help='json library for norm weights.')
    parser.add_argument('--minJetPt', default=50, type=int, help="Minimum selected jet pt")
    parser.add_argument('--maxNjets', default=8, type=int, help="Maximum number of leading jets retained in h5 files")
    parser.add_argument('--minNjets', default=6, type=int, help="Minimum number of leading jets retained in h5 files")
    parser.add_argument('--signalModel', default='2x3', type=str, help="Signal model (2x3 or 2x5)")
    parser.add_argument('--nQuarks', default=6, type=int, help="Number of quarks per gluino. Only used when --allowQuarkReMatches is use. Example: 2 quarks could be the same but matched to two jets)")
    parser.add_argument('--allowQuarkReMatches', action='store_true', help="Let quarks to be matched to multiple jets")
    parser.add_argument('--shuffleJets', action='store_true', help="Shuffle jets before saving")
    parser.add_argument('--matchingCriteria', default='RecomputeDeltaRvalues_drPriority', help='Choose matching criteria from: UseFTDeltaRvalues, RecomputeDeltaRvalues_ptPriority, RecomputeDeltaRvalues_drPriority')
    parser.add_argument('--doNotUseFSRs', action='store_true', help="Do not consider final state radiation (FSR) in jet-parton matching")
    parser.add_argument('--debug', action='store_true', help="Enable debug print statemetents")
    parser.add_argument('--doOverwrite', action="store_true", help="Overwrite already existing files")
    parser.add_argument('--doEventDisplays', action="store_true", default=False, help="Create event displays (only done if matching is performed)")
    parser.add_argument('--doSystematics', action="store_true", default=False, help="Process systematic TTrees (off by default)")
    args = parser.parse_args()

    if args.debug:
        log.setLevel("DEBUG")

    # Protection
    if args.signalModel == '2x5' and args.allowQuarkReMatches:
        log.fatal('2x5 model is not supported with the "allowQuarkReMatches" option, exiting')
        sys.exit(1)

    # get input files and sum of weights
    input_files = handleInput(args.inDir)

    # make sure output directory exists
    if not os.path.isdir(args.outDir):
        os.makedirs(args.outDir)

    # get sum of weights
    if args.normalization_denominator:
        with open(args.normalization_denominator) as f:
            sum_of_weights = json.load(f)
    else:
        sum_of_weights = get_sum_of_weights(input_files)
    log.info('Sum of weights: {}'.format(sum_of_weights))

    # get list of ttrees using the first input file
    f = ROOT.TFile(input_files[0])
    if args.doSystematics:
        treeNames = [key.GetName() for key in list(f.GetListOfKeys()) if "trees" in key.GetName()]
    else:
        treeNames = ['trees_SRRPV_']
    f.Close()

    # prepare outdir
    if not os.path.isdir(args.outDir):
        os.makedirs(args.outDir)

    # make job configurations
    confs = []
    for treeName in treeNames:
        print(f"Including tree {treeName}")
        for inFileName in input_files:
            
            dsid = str(inFileName.split('user.')[1].split('.')[2])

            confs.append({
                # input settings
                'inFileName': inFileName,
                'sum_of_weights': sum_of_weights[dsid],
                'treeName' : treeName,
                'doOverwrite' : args.doOverwrite,
                'signalModel': args.signalModel,
                'nQuarks': args.nQuarks,
                'allowQuarkReMatches': args.allowQuarkReMatches,
                # output settings
                'outDir': args.outDir,
                # jet settings
                'minJetPt': args.minJetPt,
                'maxNjets': args.maxNjets,
                'MinNjets': args.minNjets,
                # matching settings
                'MatchingCriteria': args.matchingCriteria,
                'useFSRs': not args.doNotUseFSRs,
                'dRcut': 0.4,
                # global settings
                'Debug': args.debug,
                'doEventDisplays': args.doEventDisplays
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
        dsid = str(file_name.split('user.')[1].split('.')[2])
        # Get sum of weights from metadata
        metadata_hist = tfile.Get('MetaData_EventCount')
        if dsid not in sum_of_weights:
            # initial sum of weights
            sum_of_weights[dsid] = metadata_hist.GetBinContent(3)
        else:
            # initial sum of weights
            sum_of_weights[dsid] += metadata_hist.GetBinContent(3)
    return sum_of_weights


def get_quark_flavour(quark_labels, pdgid, g, dictionary, match_neutralino, model, log):
    """ Function that assigns quarks to
    q1, q2 or q3 (q1, q2, q3, q4 or q5) for 2x3 (2x5) model """
    n_quark_labels = len(quark_labels)
    if model == '2x3':
        log.debug(f'DEBUG: g = {g}')
        log.debug(f'DEBUG: jet_matched_to_neutralino = {match_neutralino}')
        for qi, qlabel in enumerate(quark_labels, 1):
            if qi == n_quark_labels:
                return dictionary, qlabel
            if dictionary[g][qlabel] == 0:
                dictionary[g][qlabel] = pdgid
                return dictionary, qlabel
    else:  # 2x5 model
        n_quark_labels = 2 if not match_neutralino else 3
        new_quark_labels = {
          False: ['q1', 'q2'],
          True: ['q3', 'q4', 'q5'],
        }[match_neutralino]
        log.debug(f'DEBUG: g = {g}')
        log.debug(f'DEBUG: jet_matched_to_neutralino = {match_neutralino}')
        for qi, qlabel in enumerate(new_quark_labels, 1):
            if qi == n_quark_labels:
                return dictionary, qlabel
            if dictionary[g][qlabel] == 0:
                dictionary[g][qlabel] = pdgid
                return dictionary, qlabel


def make_assigments(quark_labels, assigments, g_barcodes, q_parent_barcode, pdgid, q_pdgid_dict, selected_jets, jet_index, match_neutralino, allow_quark_rematches, model, log):
    """ Find to which gluino (g1 or g2) a jet was matched and determine to which parton qX
    with X=[1, 2, 3] (X=[1, 2, 3, 4, 5]) for the 2x3 (2x5) model
    (see convention on process_files())
    """
    if g_barcodes['g1'] == 0:  # not assigned yet to any quark parent barcode
        g_barcodes['g1'] = q_parent_barcode
        q_pdgid_dict, q_flavour = get_quark_flavour(quark_labels, pdgid, 'g1', q_pdgid_dict, match_neutralino, model, log)
        if assigments['g1'][q_flavour] == -1:  # not assigned yet
            assigments['g1'][q_flavour] = jet_index
            assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
        else:  # if assigned already, pick up jet with largets pT
            if selected_jets[jet_index].Pt() > selected_jets[assigments['g1'][q_flavour]].Pt():
                assigments['g1'][q_flavour] = jet_index
                assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
    else:  # g1 was defined already (check if quark parent barcode agrees with it)
        if g_barcodes['g1'] == q_parent_barcode:
            q_pdgid_dict, q_flavour = get_quark_flavour(
                quark_labels, pdgid, 'g1', q_pdgid_dict, match_neutralino, model, log)
            if assigments['g1'][q_flavour] == -1:  # not assigned yet
                assigments['g1'][q_flavour] = jet_index
                assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
            else:  # if assigned already, pick up jet with largets pT
                if selected_jets[jet_index].Pt() > selected_jets[assigments['g1'][q_flavour]].Pt():
                    assigments['g1'][q_flavour] = jet_index
                    assigments['g1'][q_flavour.replace('q', 'f')] = pdgid
        else:
            q_pdgid_dict, q_flavour = get_quark_flavour(quark_labels, pdgid, 'g2', q_pdgid_dict, match_neutralino, model, log)
            if assigments['g2'][q_flavour] == -1:  # not assigned yet
                assigments['g2'][q_flavour] = jet_index
                assigments['g2'][q_flavour.replace('q', 'f')] = pdgid
            else:  # if assigned already, pick up jet with largets pT
                if selected_jets[jet_index].Pt() > selected_jets[assigments['g2'][q_flavour]].Pt():
                    assigments['g2'][q_flavour] = jet_index
                    assigments['g2'][q_flavour.replace('q', 'f')] = pdgid
    return assigments


def make_event_display_pt_ranked(event_number, jets_dict, quarks, fsrs = []):
    # unpack jet dict
    jets = jets_dict['jets']
    jets_level = jets_dict['level']

    # output file name
    out_name = f'Plots/event_display_{event_number}{"" if jets_level == "reco" else "_truth"}.pdf'

    # prepare data
    jet_pts = np.array([jet.Pt() for jet in jets])
    jet_etas = np.array([jet.Eta() for jet in jets])
    jet_phis = np.array([jet.Phi() for jet in jets])
    quark_pts = np.array([quark.Pt() for quark in quarks])
    quark_etas = np.array([quark.Eta() for quark in quarks])
    quark_phis = np.array([quark.Phi() for quark in quarks])
    if fsrs:
        fsr_pts = np.array([fsr.Pt() for fsr in fsrs])
        fsr_etas = np.array([fsr.Eta() for fsr in fsrs])
        fsr_phis = np.array([fsr.Phi() for fsr in fsrs])

    # make figure
    fig, ax = plt.subplots()

    plt.scatter(quark_etas, quark_phis, c=quark_pts, s=50,  label = "quarks", cmap='viridis_r')
    if fsrs:
        plt.scatter(fsr_etas, fsr_phis, c=fsr_pts, marker = 'X', s=30,  label = "FSRs", cmap='viridis_r')
    plt.scatter(jet_etas, jet_phis, c=jet_pts, alpha=0.5, s=400,  label = "jets" if jets_level == 'reco' else 'truth jets', cmap='viridis_r')

    for ijet in range(len(jets)):
        if not ijet:
            ax.add_patch(plt.Circle((jet_etas[ijet], jet_phis[ijet]), 0.4, color='k', fill = False, linestyle='dashed', alpha=0.5, label='R = 0.4'))
        else:
            ax.add_patch(plt.Circle((jet_etas[ijet], jet_phis[ijet]), 0.4, color='k', fill = False, linestyle='dashed', alpha=0.5))

    plt.xlim([-4, 4])
    plt.ylim([-np.pi, np.pi])

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_title('eventNumber = {}'.format(event_number))
    ax.legend()

    cbar = plt.colorbar()
    cbar.set_label('pT [GeV]')
    plt.legend()
    fig.tight_layout()

    plt.savefig(out_name)


def make_event_display_grouped(event_number, jets, quarks):
    # output file name
    out_name = f'Plots/event_display_{event_number}_grouped.pdf'

    # prepare jet data
    jet_pts = np.array([jet.Pt() for jet in jets])
    jet_etas = np.array([jet.Eta() for jet in jets])
    jet_phis = np.array([jet.Phi() for jet in jets])

    gluino_barcodes = list(set([quark.get_gluino_barcode() for quark in quarks]))
    neutralino_barcodes = list(set([quark.get_neutralino_barcode() for quark in quarks if quark.get_neutralino_barcode() != -999]))

    # get quarks from each gluinos
    quarks_from_gluinos_pts = {barcode: [] for barcode in gluino_barcodes}
    quarks_from_gluinos_etas = {barcode: [] for barcode in gluino_barcodes}
    quarks_from_gluinos_phis = {barcode: [] for barcode in gluino_barcodes}
    for quark in quarks:
        neutralino_barcode = quark.get_neutralino_barcode()
        if neutralino_barcode == -999:
            gluino_barcode = quark.get_gluino_barcode()
            quarks_from_gluinos_pts[gluino_barcode].append(quark.Pt())
            quarks_from_gluinos_etas[gluino_barcode].append(quark.Eta())
            quarks_from_gluinos_phis[gluino_barcode].append(quark.Phi())
    for barcode in gluino_barcodes:
        quarks_from_gluinos_pts[barcode] = np.array(quarks_from_gluinos_pts[barcode])
        quarks_from_gluinos_etas[barcode] = np.array(quarks_from_gluinos_etas[barcode])
        quarks_from_gluinos_phis[barcode] = np.array(quarks_from_gluinos_phis[barcode])

    # get quarks from each neutralino
    quarks_from_neutralinos_pts = {barcode: [] for barcode in neutralino_barcodes}
    quarks_from_neutralinos_etas = {barcode: [] for barcode in neutralino_barcodes}
    quarks_from_neutralinos_phis = {barcode: [] for barcode in neutralino_barcodes}
    for quark in quarks:
        neutralino_barcode = quark.get_neutralino_barcode()
        if neutralino_barcode != -999:
            quarks_from_neutralinos_pts[neutralino_barcode].append(quark.Pt())
            quarks_from_neutralinos_etas[neutralino_barcode].append(quark.Eta())
            quarks_from_neutralinos_phis[neutralino_barcode].append(quark.Phi())
    for barcode in neutralino_barcodes:
        quarks_from_neutralinos_pts[barcode] = np.array(quarks_from_neutralinos_pts[barcode])
        quarks_from_neutralinos_etas[barcode] = np.array(quarks_from_neutralinos_etas[barcode])
        quarks_from_neutralinos_phis[barcode] = np.array(quarks_from_neutralinos_phis[barcode])

    colors = ['red', 'green', 'orange', 'cyan']

    # make figure
    fig, ax = plt.subplots()

    color_counter = -1
    for barcode in gluino_barcodes:
        color_counter += 1
        plt.scatter(quarks_from_gluinos_etas[barcode], quarks_from_gluinos_phis[barcode], c = colors[color_counter], s=50,  label = "quarks from g", cmap='viridis_r')
    for barcode in neutralino_barcodes:
        color_counter += 1
        plt.scatter(quarks_from_neutralinos_etas[barcode], quarks_from_neutralinos_phis[barcode], c = colors[color_counter], marker = 'P', s=50,  label = "quarks from n", cmap='viridis_r')
    plt.scatter(jet_etas, jet_phis, c='blue', alpha=0.5, s=400,  label = "jets", cmap='viridis_r')

    for ijet in range(len(jets)):
        if not ijet:
            ax.add_patch(plt.Circle((jet_etas[ijet], jet_phis[ijet]), 0.4, color='k', fill = False, linestyle='dashed', alpha=0.5, label='R = 0.4'))
        else:
            ax.add_patch(plt.Circle((jet_etas[ijet], jet_phis[ijet]), 0.4, color='k', fill = False, linestyle='dashed', alpha=0.5))

    plt.xlim([-4, 4])
    plt.ylim([-np.pi, np.pi])

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_title('eventNumber = {}'.format(event_number))
    ax.legend()

    plt.legend()
    fig.tight_layout()

    plt.savefig(out_name)


def process_files(settings):

    # check if output file already exists
    outFileName = os.path.join(settings["outDir"], os.path.basename(settings["inFileName"]).replace(".root", ".h5"))
    if os.path.isfile(outFileName) and not settings['doOverwrite']:
        log.info(f"Output file already exists so skipping: {outFileName}")
        return

    signal_model = settings['signalModel']
    n_quarks = settings['nQuarks']
    allow_quark_rematches = settings['allowQuarkReMatches']

    # Set structure of output H5 file
    Structure = {
        'source': ['eta', 'mask', 'phi', 'pt', 'e'],
        'EventVars': ['gmass', 'normweight', 'jet_SphericityTensor_eigen21', 'jet_SphericityTensor_eigen22','jet_SphericityTensor_eigen31', 'jet_SphericityTensor_eigen32', 'jet_SphericityTensor_eigen33'],
    }

    # Collect info to know matching efficiency for each quark flavour
    quark_flavours = [1, 2, 3, 4, 5, 6]
    NquarksByFlavour = {flav: 0 for flav in quark_flavours}
    NmatchedQuarksByFlavour = {flav: 0 for flav in quark_flavours}
    
    # conventions:
    # 2x3 signals:
    # q1 is the first matched quark found for the corresponding gluino (f1 is its pdgID)
    # q2 is the second matched quark found for the corresponding gluino (f2 is its pdgID)
    # q3 is the third matched quark found for the corresponding gluino (f3 is its pdgID)
    # 2x5 signals:
    # q1 is the first matched quark found for the corresponding gluino (f1 is its pdgID)
    # q2 is the second matched quark found for the corresponding gluino (f2 is its pdgID)
    # q3 is the first matched quark found for the corresponding neutralino (f3 is its pdgID)
    # q4 is the second matched quark found for the corresponding neutralino (f4 is its pdgID)
    # q5 is the third matched quark found for the corresponding neutralino (f5 is its pdgID)
    # g1 is the first parent gluino for first matched quark
    if not allow_quark_rematches:
        quark_labels = ['q1', 'q2', 'q3']
        if signal_model == '2x5':
            quark_labels += ['q4', 'q5']
    else:
        quark_labels = [f'q{x}' for x in range(1, n_quarks+1)]
    for gcase in ['g1', 'g2']:
        Structure[gcase] = ['mask']
        Structure[gcase] += quark_labels
        Structure[gcase] += [label.replace('q', 'f') for label in quark_labels]
                
    # Reconstructed mass by truth mass
    Masses = [100, 200, 300, 400] + [900 + i*100 for i in range(0, 17)]
    hRecoMasses = {mass: ROOT.TH1D(f'RecoMass_TruthMass{mass}', '', 300, 0, 3000) for mass in Masses}
    
    # Reconstructed gluino mass - true gluino mass
    hGluinoMassDiff = ROOT.TH1D('GluinoMassDiff', '', 10000, -10000, 10000)
    matchedEvents = 0
    partial_events = 0  # keep track of partially reconstructed events (gluinos)
    neutralinos_fully_matched = 0
    neutralinos_partially_matched = 0
    multipleQuarkMatchingEvents = 0
    matchedEventNumbers = []

    # Initialize lists for each variable to be saved
    assigments_list = {key: {case: [] for case in cases} for key, cases in Structure.items()}

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
        
        # Select reco jets
        AllPassJets = []
        for ijet in range(len(tree.jet_pt)):
            if tree.jet_pt[ijet] > settings['minJetPt']:
                jet = RPVJet()
                jet.SetPtEtaPhiE(tree.jet_pt[ijet], tree.jet_eta[ijet], tree.jet_phi[ijet], tree.jet_e[ijet])
                if settings['MatchingCriteria'] == 'UseFTDeltaRvalues':
                    jet.set_matched_parton_barcode(int(tree.jet_deltaRcut_matched_truth_particle_barcode[ijet]))
                    jet.set_matched_fsr_barcode(int(tree.jet_deltaRcut_FSRmatched_truth_particle_barcode[ijet]))
                AllPassJets.append(jet)
        # select leading n jets with n == min(maxNjets, njets)
        SelectedJets = [AllPassJets[i] for i in range(min(settings['maxNjets'], len(AllPassJets)))]
        nJets = len(SelectedJets)
        nQuarksFromGs = len(tree.truth_QuarkFromGluino_pt) if tree.GetBranchStatus("truth_QuarkFromGluino_pt") else 0
        nQuarks = nQuarksFromGs
        nFSRsFromGs = len(tree.truth_FSRFromGluinoQuark_pt) if tree.GetBranchStatus("truth_FSRFromGluinoQuark_pt") else 0
        if signal_model == '2x5':
            if tree.GetBranchStatus("truth_QuarkFromNeutralino_pt"):
                nQuarks += len(tree.truth_QuarkFromNeutralino_pt)
            if tree.GetBranchStatus("truth_FSRFromNeutralinoQuark_pt"):
                nFSRsFromNeutralinos = len(tree.truth_FSRFromNeutralinoQuark_pt)

        # Apply event selections
        passEventSelection = True
        if nJets < settings['MinNjets']:
            passEventSelection = False
        if not allow_quark_rematches and nQuarks != len(quark_labels) * 2:
            passEventSelection = False
        if not passEventSelection:
            continue  # skip event

        event_counter += 1

        # Protection
        if nJets > settings['maxNjets']:
            log.fatal(f'More than {settings["maxNjets"]} jets were found ({nJets}), fix me!')
            sys.exit(1)

        # Extract gluino mass
        for ipart in range(len(tree.truth_parent_m)):  # loop over truth particles
            if tree.truth_parent_pdgId[ipart] == 1000021:  # it's a gluino
                gmass = tree.truth_parent_m[ipart]
                break

        # Put place-holder values for each variable
        # set jet index for each particle q1,q2,q3 (q4,q5 for 2x5) to -1 (i.e. no matching) and mask to True
        def init_value(case):
            if case == 'mask':
                return True
            if 'q' in case:
                return -1
            return 0
        Assigments = {key: {case: init_value(case) for case in cases} for key, cases in Structure.items()}

        # Create arrays with jet info (extend Assigments with jet reco info)
        for case in Structure['source']:
            array = []
            for j in SelectedJets:
                if case == 'eta':
                    array.append(j.Eta())
                elif case == 'e':
                    array.append(j.E())
                elif case == 'phi':
                    array.append(j.Phi())
                elif case == 'pt':
                    array.append(j.Pt())
                elif case == 'mask':
                    array.append(True)
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
        Assigments['EventVars']['jet_Cparam'] = 3*(tree.jet_SphericityTensor_eigen31[0]*tree.jet_SphericityTensor_eigen32[0] + tree.jet_SphericityTensor_eigen31[0]*tree.jet_SphericityTensor_eigen33[0] + tree.jet_SphericityTensor_eigen32[0]*tree.jet_SphericityTensor_eigen33[0])
        Assigments['EventVars']['gmass'] = gmass
        #Assigments['EventVars']['minAvgMass'] = tree.minAvgMass_jetdiff10_btagdiff10
        Assigments['EventVars']['normweight'] = tree.mcEventWeightsVector[0] * tree.pileupWeight * tree.weight_filtEff * tree.weight_kFactor * tree.weight_xs / settings['sum_of_weights']

        # Collect gluino barcodes
        gBarcodes = {'g1': 0, 'g2': 0}  # fill temporary values

        # Collect quark -> gluino associations
        qPDGIDs = {g: {label: 0 for label in quark_labels} for g in ['g1', 'g2']}  # fill temporary values

        # Select quarks from gluinos
        log.debug('Get quarks from gluinos')
        Quarks = [RPVParton() for i in range(nQuarksFromGs)]
        for iquark in range(nQuarksFromGs):
            quark_pt = tree.truth_QuarkFromGluino_pt[iquark]
            quark_eta = tree.truth_QuarkFromGluino_eta[iquark]
            quark_phi = tree.truth_QuarkFromGluino_phi[iquark]
            quark_e = tree.truth_QuarkFromGluino_e[iquark]
            quark_parent_barcode = tree.truth_QuarkFromGluino_ParentBarcode[iquark]
            quark_barcode = tree.truth_QuarkFromGluino_barcode[iquark]
            quark_pdgid = tree.truth_QuarkFromGluino_pdgId[iquark]
            Quarks[iquark].SetPtEtaPhiE(quark_pt, quark_eta, quark_phi, quark_e)
            Quarks[iquark].set_gluino_barcode(quark_parent_barcode)
            Quarks[iquark].set_barcode(quark_barcode)
            Quarks[iquark].set_pdgid(quark_pdgid)

        if settings['useFSRs']:  # select FSR quarks from gluinos
            log.debug('Get FSRs from gluinos')
            FSRs = [RPVParton() for i in range(nFSRsFromGs)]
            for iFSR in range(nFSRsFromGs):
                FSRs[iFSR].SetPtEtaPhiE(tree.truth_FSRFromGluinoQuark_pt[iFSR], tree.truth_FSRFromGluinoQuark_eta[iFSR],
                                        tree.truth_FSRFromGluinoQuark_phi[iFSR], tree.truth_FSRFromGluinoQuark_e[iFSR])
                # Find quark which emitted this FSR and get its parentBarcode
                quark_found = False
                for parton in Quarks:
                    if parton.get_barcode() == tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR]:
                        quark_found = True
                        FSRs[iFSR].set_gluino_barcode(
                            parton.get_gluino_barcode())
                        FSRs[iFSR].set_pdgid(parton.get_pdgid())
                if not quark_found:
                    log.fatal('Quark from gluino that emitted FSR (iFSR = {}) not found, exiting'.format(iFSR))
                    sys.exit(1)
                FSRs[iFSR].set_barcode(
                    tree.truth_FSRFromGluinoQuark_barcode[iFSR])
                FSRs[iFSR].set_quark_barcode(
                    tree.truth_FSRFromGluinoQuark_LastQuarkInChain_barcode[iFSR])

        # Get nuetralinos
        if signal_model == '2x5':
            nNeutralinos = len(tree.truth_NeutralinoFromGluino_pt)
            neutralinos = [RPVParton() for i in range(nNeutralinos)]
            for iNeutralino in range(nNeutralinos):
                neutralinos[iNeutralino].SetPtEtaPhiE(tree.truth_NeutralinoFromGluino_pt[iNeutralino], tree.truth_NeutralinoFromGluino_eta[iNeutralino],
                                                      tree.truth_NeutralinoFromGluino_phi[iNeutralino], tree.truth_NeutralinoFromGluino_e[iNeutralino])
                neutralinos[iNeutralino].set_barcode(tree.truth_NeutralinoFromGluino_barcode[iNeutralino])
                neutralinos[iNeutralino].set_gluino_barcode(tree.truth_NeutralinoFromGluino_ParentBarcode[iNeutralino])

        # Add quarks from neutralinos
        log.debug('Adding quarks from neutralinos')
        for iquark in range(nQuarksFromGs, nQuarks):
            index = iquark - nQuarksFromGs
            quark_pt = tree.truth_QuarkFromNeutralino_pt[index]
            quark_eta = tree.truth_QuarkFromNeutralino_eta[index]
            quark_phi = tree.truth_QuarkFromNeutralino_phi[index]
            quark_e = tree.truth_QuarkFromNeutralino_e[index]
            quark_neutralino_barcode = tree.truth_QuarkFromNeutralino_ParentBarcode[index]
            # Find parent gluino barcode
            neutralino_found = False
            for neutralino in neutralinos:
                if neutralino.get_barcode() == quark_neutralino_barcode:
                    neutralino_found = True
                    quark_gluino_barcode = neutralino.get_gluino_barcode()
            if not neutralino_found:
                log.fatal(f'Corresponding neutralino not found for quark {iquark} not found, exiting')
                sys.exit(1)
            quark_barcode = tree.truth_QuarkFromNeutralino_barcode[index]
            quark_pdgid = tree.truth_QuarkFromNeutralino_pdgId[index]
            Quarks += [RPVParton()]
            Quarks[iquark].SetPtEtaPhiE(quark_pt, quark_eta, quark_phi, quark_e)
            Quarks[iquark].set_gluino_barcode(quark_gluino_barcode)
            Quarks[iquark].set_neutralino_barcode(quark_neutralino_barcode)
            Quarks[iquark].set_barcode(quark_barcode)
            Quarks[iquark].set_pdgid(quark_pdgid)
            Quarks[iquark].set_is_coming_from_neutralino()

        if signal_model == '2x5' and settings['useFSRs']:  # select FSR quarks from neutralinos
            log.debug('Get FSRs from neutralinos')
            FSRs += [RPVParton() for i in range(nFSRsFromNeutralinos)]
            for iFSR in range(nFSRsFromGs, nFSRsFromNeutralinos + nFSRsFromGs):
                index = iFSR - nFSRsFromGs
                FSRs[iFSR].SetPtEtaPhiE(tree.truth_FSRFromNeutralinoQuark_pt[index], tree.truth_FSRFromNeutralinoQuark_eta[index],
                                        tree.truth_FSRFromNeutralinoQuark_phi[index], tree.truth_FSRFromNeutralinoQuark_e[index])
                # Find quark which emitted this FSR and get its parentBarcode
                quark_found = False
                for iparton in range(nQuarksFromGs, len(Quarks)):
                    parton = Quarks[iparton]
                    if parton.get_barcode() == tree.truth_FSRFromNeutralinoQuark_LastNeutralinoInChain_barcode[index]:
                        quark_found = True
                        FSRs[iFSR].set_gluino_barcode(parton.get_gluino_barcode())
                        FSRs[iFSR].set_pdgid(parton.get_pdgid())
                if not quark_found:
                    log.fatal('Quark from neutralino that emitted FSR (iFSR = {}) not found, exiting'.format(iFSR))
                    sys.exit(1)
                FSRs[iFSR].set_barcode(tree.truth_FSRFromNeutralinoQuark_barcode[index])
                FSRs[iFSR].set_quark_barcode(tree.truth_FSRFromNeutralinoQuark_LastNeutralinoInChain_barcode[index])
                FSRs[iFSR].set_is_coming_from_neutralino()

        # Create event display
        if settings['doEventDisplays']:
            make_event_display_pt_ranked(
                tree.eventNumber,
                {'jets': SelectedJets, 'level': 'reco'},
                Quarks,
                [] if not settings['useFSRs'] else FSRs)
            make_event_display_grouped(tree.eventNumber, SelectedJets, Quarks)
            # get truth jets
            truth_jets = []
            for ijet in range(len(tree.truth_jet_pt)):
                if tree.truth_jet_pt[ijet] > settings['minJetPt']:
                    jet = RPVJet()
                    jet.SetPtEtaPhiE(tree.truth_jet_pt[ijet], tree.truth_jet_eta[ijet], tree.truth_jet_phi[ijet], tree.truth_jet_e[ijet])
                    truth_jets.append(jet)
            make_event_display_pt_ranked(tree.eventNumber, {'jets': truth_jets, 'level': 'truth'}, Quarks)

        # Match reco jets to closest parton
        matcher = RPVMatcher(Jets = SelectedJets, Partons = Quarks)
        if settings['useFSRs']:
            matcher.add_fsrs(FSRs)
        if settings['Debug']:
            matcher.set_property('Debug', True)
        matcher.set_property('maxNmatchedJets', len(quark_labels) * 2)
        if allow_quark_rematches:
            matcher.set_property('MatchJetsToMatchedQuarks', True)
        matcher.set_property('MatchingCriteria',settings['MatchingCriteria'])
        if settings['MatchingCriteria'] != "UseFTDeltaRvalues":
            matcher.set_property('DeltaRcut', settings['dRcut'])
        matched_jets = matcher.match()

        # Fill Assigments (info for matched jets)
        for jet_index, jet in enumerate(matched_jets):
            if jet.is_matched():
                Assigments = make_assigments(quark_labels, Assigments, gBarcodes, jet.get_match_gluino_barcode(), jet.get_match_pdgid(), qPDGIDs, matched_jets, jet_index, jet.is_matched_to_neutralino(), allow_quark_rematches, signal_model, log)

        # Check if fully matched
        n_matched_jets = sum([1 if jet.is_matched() else 0 for jet in matched_jets])
        log.debug(f'number of matched jets = {n_matched_jets}')
        if n_matched_jets == len(quark_labels) * 2:
            matchedEventNumbers.append(tree.eventNumber)
            matchedEvents += 1

        # See if gluinos were fully reconstructed (i.e. each decay particle matches a jet)
        for g in ['g1', 'g2']:
            TempMask = True
            if not allow_quark_rematches:
                for key in Assigments[g]:
                    if key == 'mask' or 'f' in key:
                        continue
                    if Assigments[g][key] == -1:
                        TempMask = False
            else:  # check only first 3 (5) quarks for the 2x3 (2x5) model
                quark_labels_to_check = ['q1', 'q2', 'q3']
                if signal_model == '2x5':
                    quark_labels_to_check += ['q4', 'q5']
                for key in quark_labels_to_check:
                    if Assigments[g][key] == -1:
                        TempMask = False
            Assigments[g]['mask'] = TempMask

        # Count number of at least partially reconstructed events
        if Assigments['g1']['mask'] or Assigments['g2']['mask']:
            partial_events += 1
            log.debug('Event is at least partially reconstructed!')

        # Check if neutralinos were matched
        if signal_model == '2x5':
            neutralino_barcodes = list(set([quark.get_neutralino_barcode() for quark in Quarks if quark.get_neutralino_barcode() != -999]))
            matched_neutralinos = {barcode: 0 for barcode in neutralino_barcodes}
            for jet in matched_jets:
                match_neutralino_barcode = jet.get_match_neutralino_barcode()
                if match_neutralino_barcode in matched_neutralinos:
                    matched_neutralinos[match_neutralino_barcode] += 1
            fully_matched_neutralinos = sum([1 if n == 3 else 0 for barcode, n in matched_neutralinos.items()])
            if fully_matched_neutralinos == 1:
                neutralinos_partially_matched += 1
            elif fully_matched_neutralinos == 2:
                neutralinos_partially_matched += 1
                neutralinos_fully_matched += 1

        # Compare reconstructed gluino mass with true gluino mass
        MultipleJetsMatchingAQuark = False
        AllMatchedJetsIndexes = []
        for ig in ['g1', 'g2']:  # loop over gluinos
            # fully reconstructable gluino (every quark matches a jet)
            if Assigments[ig]['mask']:
                Jets2sum = []
                JetIndexes = []
                for key in quark_labels:
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

    # Close input file
    del tree

    # Create H5 file
    log.info('Creating {}'.format(outFileName))
    with h5py.File(outFileName, 'w') as HF:
        Groups, Datasets = dict(), dict()
        for key in Structure:
            Groups[key] = HF.create_group(key)
            for case in Structure[key]:
                Datasets[key+'_'+case] = Groups[key].create_dataset(case, data=assigments_list[key][case])

    # Save histogram
    outFile = ROOT.TFile(os.path.join(settings["outDir"], "GluinoMassDiff.root"), 'RECREATE')
    hGluinoMassDiff.Write()
    outFile.Close()
    
    # Reco gluino mass distributions
    outFile = ROOT.TFile(os.path.join(settings["outDir"], "ReconstructedGluinoMasses.root"), 'RECREATE')
    for key, hist in hRecoMasses.items():
        hist.Write()
    outFile.Close()

    # print matching efficiency
    log.info(f'matching efficiency (percentage of events where {len(quark_labels) * 2} quarks are matched): {matchedEvents/event_counter}')
    log.info(f'partial matching efficiency (percentage of events where at least one gluino is matched): {partial_events/event_counter}')
    if signal_model == '2x5':
        log.info(f'matching efficiency for reconstructing both neutralinos: {neutralinos_fully_matched/event_counter}')
        log.info(f'matching efficiency for reconstructing at least one neutralino: {neutralinos_partially_matched/event_counter}')
    log.info(f'Number of events where {len(quark_labels) * 2} quarks are matched: {matchedEvents}')
    log.info(f'percentage of events having a quark matching several jets: {multipleQuarkMatchingEvents/event_counter}')
    for flav in quark_flavours:
        if NquarksByFlavour[flav] != 0:
            log.info(f'Matching efficiency for quarks w/ abs(pdgID)=={flav}: {NmatchedQuarksByFlavour[flav]/NquarksByFlavour[flav]}')

    # saving matching settings
    with open(os.path.join(settings["outDir"], "matchedEvents.root"),"w") as outFile:
        for event in matchedEventNumbers:
            outFile.write(str(event)+'\n')

    log.info('>>> ALL DONE <<<')

if __name__ == '__main__':
    main()
