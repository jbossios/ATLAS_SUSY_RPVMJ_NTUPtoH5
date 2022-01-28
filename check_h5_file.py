import h5py,os,sys
import pandas as pd
import numpy as np

def check_h5_file(filename):
  iFile = h5py.File(filename,"r")
  keys  = list(iFile.keys())
  print('Groups: ')
  print(keys)
  for key in keys:
    print('Subgroups inside {}:'.format(key))
    data      = iFile.get(key) # get group
    subgroups = list(data.items()) # get list of subgroups
    subGroups = [x[0] for x in subgroups]
    print(subGroups)
    for item in subGroups:
      print('Data on {}/{}:'.format(key,item))
      print(np.array(data.get(item)))
      print('size: {}'.format(np.array(data.get(item).size)))

if __name__ == '__main__':
  check_h5_file('/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_inputs/signal_UDB_UDS_training_v22.h5')
