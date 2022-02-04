import h5py
import os
import ROOT
import numpy as np

def get_true_reco_gluino_masses(file_name: str, use_avg: bool = True) -> [float]:
  # Open H5DF file and get data
  hf = h5py.File(file_name, 'r')
  Data = {case: hf.get(case) for case in ['source', 'g1', 'g2']}

  # Get jet info
  jetMaskInfo = np.array(Data['source'].get('mask'))
  jetPtInfo   = np.array(Data['source'].get('pt'))
  jetEtaInfo  = np.array(Data['source'].get('eta'))
  jetPhiInfo  = np.array(Data['source'].get('phi'))
  jetMassInfo = np.array(Data['source'].get('mass'))

  # Get info for each particle
  gluinoInfo = {gCase: {info: np.array(Data[gCase].get(info)) for info in ['mask', 'q1', 'q2', 'q3']} for gCase in ['g1', 'g2']}

  # Save reconstructed masses using true matched jets for '2g' events
  TrueRecoMasses2g = []

  # Event loop
  TotalNevents = jetMaskInfo.shape[0]
  for ievent in range(TotalNevents):
    ReconstructableGluinos = 0 # number of reconstructable gluinos in this event
    for gCase in ['g1', 'g2']:
      if gluinoInfo[gCase]['mask'][ievent]:
        ReconstructableGluinos += 1
    if ReconstructableGluinos == 2:
      if use_avg: masses = dict()
      for gcase in ['g1', 'g2']:
        Jets = []
        for qcase in ['q1', 'q2', 'q3']:
          jetIndex = gluinoInfo[gcase][qcase][ievent]
          jetPt    = jetPtInfo[ievent][jetIndex]
          jetEta   = jetEtaInfo[ievent][jetIndex]
          jetPhi   = jetPhiInfo[ievent][jetIndex]
          jetM     = jetMassInfo[ievent][jetIndex]
          Jet      = ROOT.TLorentzVector()
          Jet.SetPtEtaPhiM(jetPt,jetEta,jetPhi,jetM)
          Jets.append(Jet)
        if use_avg: masses[gcase] = (Jets[0]+Jets[1]+Jets[2]).M()
        else: TrueRecoMasses2g.append((Jets[0]+Jets[1]+Jets[2]).M())
      if use_avg: TrueRecoMasses2g.append(0.5 * (masses['g1'] + masses['g2']))
  hf.close()
  return TrueRecoMasses2g

def make_figure(masses: [float], use_avg: bool = True):
  TrueHist = ROOT.TH1D('TrueHists', '', 300, 0, 3000)
  for value in masses:
    TrueHist.Fill(value)

  if not os.path.exists('Plots'):
    os.makedirs('Plots')
  
  # TCanvas
  Canvas  = ROOT.TCanvas()
  outName = "Plots/RecoMass_true_matching_2g.pdf"
  Canvas.Print(outName+"[")
  Canvas.SetLogy()
  Stack = ROOT.THStack()
  Stack.Add(TrueHist,'HIST][')
  Stack.Draw('nostack')
  if use_avg:
    Stack.GetXaxis().SetTitle('Averaged reconstructed gluino Mass [GeV]')
  else:
    Stack.GetXaxis().SetTitle('Reconstructed gluino Mass [GeV]')
  Stack.GetYaxis().SetTitle('Number of (2g) events')
  #Canvas.Update()
  #Canvas.Modified()
  Canvas.Print(outName)
  Canvas.Print(outName+']')

if __name__ == '__main__':
  values = get_true_reco_gluino_masses('UDB+UDSSignalData_1400_training.h5')
  make_figure(values)
