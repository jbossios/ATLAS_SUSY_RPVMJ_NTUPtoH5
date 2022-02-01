import h5py
import os
import ROOT
import numpy as np

SAMPLES = {
  'Signal' : {
    'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Git/Outputs/Signal/v24/UDB+UDSSignalData_All_testing.h5',
    'Pred' : '/home/jbossios/cern/SUSY/RPVMJ/SPANet_outputs/H5predictions/v60/signal_testing_v60_output.h5', # v60 spanet predictions for v24 h5 file
  },
  'Dijets' : {
    'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Git/Outputs/dijets/dijets_364712_spanet_test.h5',
    'Pred' : '/home/jbossios/cern/SUSY/RPVMJ/SPANet_outputs/H5predictions/dijets_test/signal_testing_v60_output_364712_test.h5', # v60 spanet predictions for JZ12
  } 
}

def get_reco_gluino_masses(case: str, case_dict: dict) -> [float]:
  # Save reconstructed masses using true matched jets for '2g' events
  RecoMasses2g = dict()

  # Open H5DF files and get data
  Files = {level: h5py.File(file_name, 'r') for level, file_name in case_dict.items()}
  Data = {level: {case: Files[level].get(case) for case in ['source', 'g1', 'g2']} for level in case_dict}
  
  # Get gluino info
  gluinoInfo = {level: {gCase: {info: np.array(Data[level][gCase].get(info)) for info in ['mask', 'q1', 'q2', 'q3']} for gCase in ['g1', 'g2']} for level in case_dict}

  # Get jet info
  jetMaskInfo = {level: np.array(Data[level]['source'].get('mask')) for level in case_dict}
  jetPtInfo   = np.array(Data['True']['source'].get('pt'))
  jetEtaInfo  = np.array(Data['True']['source'].get('eta'))
  jetPhiInfo  = np.array(Data['True']['source'].get('phi'))
  jetMassInfo = np.array(Data['True']['source'].get('mass'))

  for level, full_file_name in case_dict.items():
    if case == 'Dijets' and level == 'True': continue
    RecoMasses2g[level] = []
    # Event loop
    for ievent in range(jetMaskInfo[level].shape[0]):
      ReconstructableGluinos = 0 if case != 'Dijets' else 2 # number of reconstructable gluinos in this event
      if case != 'Dijets':
        for gCase in ['g1', 'g2']:
          if gluinoInfo['True'][gCase]['mask'][ievent]:
            ReconstructableGluinos += 1
      if ReconstructableGluinos == 2:
        masses = dict()
        for gcase in ['g1', 'g2']:
          Jets = []
          for qcase in ['q1', 'q2', 'q3']:
            jetIndex = gluinoInfo[level][gcase][qcase][ievent]
            jetPt    = jetPtInfo[ievent][jetIndex]
            jetEta   = jetEtaInfo[ievent][jetIndex]
            jetPhi   = jetPhiInfo[ievent][jetIndex]
            jetM     = jetMassInfo[ievent][jetIndex]
            Jet      = ROOT.TLorentzVector()
            Jet.SetPtEtaPhiM(jetPt,jetEta,jetPhi,jetM)
            Jets.append(Jet)
          masses[gcase] = (Jets[0]+Jets[1]+Jets[2]).M()
          #RecoMasses2g.append((Jets[0]+Jets[1]+Jets[2]).M())
        RecoMasses2g[level].append(0.5 * (masses['g1'] + masses['g2']))
  #if case == 'Dijets': print(RecoMasses2g)
  return RecoMasses2g

def make_hist(case: str, masses_dict: dict) -> ROOT.TH1D:
  hists = dict()
  for level, masses in masses_dict.items():
    if case == 'Dijets' and level == 'True': continue
    hist = ROOT.TH1D(case, '', 500, 0, 5000)
    for value in masses:
      hist.Fill(value)
    hists[level] = hist
  return hists

def compare_hists(hists: dict()):
  if not os.path.exists('Plots'):
    os.makedirs('Plots')

  colors = [ROOT.kBlack, ROOT.kRed, ROOT.kMagenta]
  
  # TCanvas
  Canvas = ROOT.TCanvas()
  comparison = '_vs_'.join(hists.keys())
  outName = f"Plots/RecoMass_{comparison}_2g.pdf"
  Canvas.Print(outName+"[")
  Canvas.SetLogy()
  Stack = ROOT.THStack()
  Legends = ROOT.TLegend(0.7,0.7,0.92,0.9)
  Legends.SetTextFont(42)
  counter = 0
  for case, case_dict in hists.items():
    for level, hist in case_dict.items():
      hist.SetLineColor(colors[counter])
      hist.SetMarkerColor(colors[counter])
      Stack.Add(hist, 'HIST][')
      Legends.AddEntry(hist, f'{case}_{level}')
      counter += 1
  Stack.Draw('nostack')
  #Stack.GetXaxis().SetTitle('Reconstructed gluino Mass [GeV]')
  Stack.GetXaxis().SetTitle('Averaged reconstructed gluino Mass [GeV]')
  Stack.GetYaxis().SetTitle('Number of (2g) events')
  Legends.Draw("same")
  Canvas.Update()
  Canvas.Modified()
  Canvas.Print(outName)
  Canvas.Print(outName+']')

if __name__ == '__main__':
  hists = {case: make_hist(case, get_reco_gluino_masses(case, case_dict)) for case, case_dict in SAMPLES.items()}
  compare_hists(hists)
