import ROOT,os,sys

# Styles
Colors = [ROOT.kBlack,ROOT.kRed,ROOT.kBlue]

# AtlasStyle
#ROOT.gROOT.LoadMacro("/afs/cern.ch/user/j/jbossios/work/public/xAOD/Results/AtlasStyle/AtlasStyle.C")
#ROOT.SetAtlasStyle()
ROOT.gROOT.SetBatch(True)

# Collect histograms
Hists = dict()

# Loop over matching criteria
for case in ['JetsFirst_rmMQs','QuarksFirst']:
  # Open input file
  FileName = 'LeaV3/GluinoMassDiff_{}.root'.format(case)
  iFile    = ROOT.TFile.Open(FileName)
  if not iFile:
    print('ERROR: {} not found, exiting'.format(FileName))
    sys.exit(1)
  hist = iFile.Get('GluinoMassDiff')
  hist.SetDirectory(0)
  Hist = hist.Clone('GluinoMassDiff_{}'.format(case))
  Hists[case] = hist
  iFile.Close()

# Compare distributions
Canvas  = ROOT.TCanvas()
outName = 'GluinoMassDiff_LastQuarkInChain.pdf'
Canvas.SetLogy()
Canvas.Print(outName+'[')
Stack = ROOT.THStack()
Legends = ROOT.TLegend(0.7,0.43,0.92,0.9)
Legends.SetTextFont(42)
counter = 0
for case,hist in Hists.items():
  hist.SetLineColor(Colors[counter])
  hist.SetMarkerColor(Colors[counter])
  print('Integral for {}: {}'.format(case,hist.Integral()))
  hist.Scale(1./hist.Integral())
  Stack.Add(hist,'HIST][')
  Legends.AddEntry(hist,case.replace('_rmMQs','') if '_rmMQs' in case else case,'l')
  counter += 1
Stack.Draw('nostack')
Stack.GetXaxis().SetTitle('Reconstructed gluino mass - True gluino mass [GeV]')
Stack.GetXaxis().SetRangeUser(-1500,1000)
Legends.Draw('same')
Canvas.Print(outName)
Canvas.Print(outName+']')

########################################################################
# Check if usage of last quark in chain improves gluino mass difference
########################################################################

# Collect histograms
Hists = dict()

case = 'JetsFirst_rmMQs'

# Open input file
FileName = 'LeaV3/GluinoMassDiff_{}.root'.format(case)
iFile    = ROOT.TFile.Open(FileName)
if not iFile:
  print('ERROR: {} not found, exiting'.format(FileName))
  sys.exit(1)
hist = iFile.Get('GluinoMassDiff')
hist.SetDirectory(0)
Hist = hist.Clone('GluinoMassDiff_{}'.format(case))
Hists['LastQuarkInChain'] = hist
iFile.Close()

# Open input file
FileName = 'GluinoMassDiff_{}.root'.format(case)
iFile    = ROOT.TFile.Open(FileName)
if not iFile:
  print('ERROR: {} not found, exiting'.format(FileName))
  sys.exit(1)
hist = iFile.Get('GluinoMassDiff')
hist.SetDirectory(0)
Hist = hist.Clone('GluinoMassDiff_{}'.format(case))
Hists['FistQuarkInChain'] = hist
iFile.Close()

# Compare distributions
Canvas  = ROOT.TCanvas()
outName = 'GluinoMassDiff_FirstQuark_vs_LastQuark.pdf'
Canvas.Print(outName+'[')
Stack = ROOT.THStack()
Legends = ROOT.TLegend(0.7,0.43,0.92,0.9)
Legends.SetTextFont(42)
counter = 0
for case,hist in Hists.items():
  hist.SetLineColor(Colors[counter])
  hist.SetMarkerColor(Colors[counter])
  print('Integral for {}: {}'.format(case,hist.Integral()))
  hist.Scale(1./hist.Integral())
  Stack.Add(hist,'HIST][')
  Legends.AddEntry(hist,case,'l')
  counter += 1
Stack.Draw('nostack')
Stack.GetXaxis().SetTitle('Reconstructed gluino mass - True gluino mass [GeV]')
Stack.GetXaxis().SetRangeUser(-1500,1000)
Legends.Draw('same')
Canvas.Print(outName)
Canvas.Print(outName+']')

