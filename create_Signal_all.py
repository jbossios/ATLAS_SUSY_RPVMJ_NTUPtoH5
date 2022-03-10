import os

def main(versions):
  # Loop over versions
  for version, vdict in versions.items():
    # Loop over full and training/test productions
    for case in ['', ' --doNotSplit4Training']:
      fsr = ''
      if 'noFSRs' in vdict and vdict['noFSRs']:
        fsr = ' --doNotUseFSRs'
      os.system(f'python3 CreateH5files_Signal_DirectDecayMode.py --v {version} --maxNjets {vdict["maxNjets"]} --pTcut {vdict["pTcut"]}{case} > Log_{version}{"_full" if case!="" else ""}{fsr} 2>&1 &')
  print('>>> ALL JOBS SUBMITTED <<<')

if __name__ == '__main__':
  versions = {
  ##  'v41': {'maxNjets': 6, 'pTcut': 20},
  ##  'v42': {'maxNjets': 6, 'pTcut': 30},
  ##  'v43': {'maxNjets': 6, 'pTcut': 40},
  ##  'v44': {'maxNjets': 6, 'pTcut': 50},
  ##  'v45': {'maxNjets': 6, 'pTcut': 60},
  ##  'v46': {'maxNjets': 7, 'pTcut': 20},
  ##  'v47': {'maxNjets': 7, 'pTcut': 30},
  ##  'v48': {'maxNjets': 7, 'pTcut': 40},
  ##  'v49': {'maxNjets': 7, 'pTcut': 50},
  ##  'v50': {'maxNjets': 7, 'pTcut': 60},
  ##  'v51': {'maxNjets': 8, 'pTcut': 20},
  ##  'v52': {'maxNjets': 8, 'pTcut': 30},
  ##  'v53': {'maxNjets': 8, 'pTcut': 40},
  ##  'v54': {'maxNjets': 8, 'pTcut': 60},
  ##  'v55': {'maxNjets': 9, 'pTcut': 20},
  ##  'v56': {'maxNjets': 9, 'pTcut': 30},
  ##  'v57': {'maxNjets': 9, 'pTcut': 40},
  ##  'v58': {'maxNjets': 9, 'pTcut': 50},
  ##  'v59': {'maxNjets': 9, 'pTcut': 60},
    'v60': {'maxNjets': 8, 'pTcut': 50, 'noFSRs': True},
  }
  main(versions)
