
# TODO FIXME
# Support the addition of jets/partons/fsrs by passing values
# Create Matcher class??

import ROOT
import sys
from typing import Union

class RPVJet(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.BarcodeLeaMatch = -1 # FIXME
    self.isMatched = False

class RPVParton(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.parentBarcode = -999
    self.barcode       = -999
    self.pdgID         = -999

class RPVMatcher():
  __properties = {
    'RecomputeDeltaRvalues' : False, # Temporary FIXME
    'DeltaRcut' : 0.4,
    'ReturnOnlyMatched' : False,
  }
  def __init__(self, **kargs):
    # Protection
    for key in kargs:
      if key not in self.__properties and key != 'Jets' and key != 'Partons' and key != 'FSRs':
        print(f'ERROR: {opt} was not recognized, exiting')
        sys.exit(1)
    self.properties = dict()
    # Set default properties
    for opt, default_value in self.__properties.items():
      self.properties[opt] = default_value
    # Use provided settings
    if 'Jets' in kargs:
      self.jets = kargs['Jets']
    else:
      self.jets = None
    if 'Partons' in kargs:
      self.partons = kargs['Partons']
    else:
      self.partons = None
    if 'FSRs' in kargs:
      self.fsrs = kargs['FSRs']
    else:
      self.fsrs = None
    for key in kargs:
      if key in self.__properties:
        self.set_property(key, kargs[key])

  def add_jets(self, jets: [RPVJet]):
    self.jets = jets

  def add_partons(self, partons: [RPVParton]):
    self.partons = partons

  def add_fsrs(self, fsrs: [RPVParton]):
    self.fsrs = fsrs

  def set_property(self, opt: str, value: Union[bool, float]):
    if opt not in self.__properties:
      print(f'ERROR: {opt} was not recognized, exiting')
      sys.exit(1)
    self.properties[opt] = value

  def __match_use_deltar_values_from_ft(self) -> [RPVJet]:
    return 'NotImplemented' # FIXME
  
  def __match_recompute_deltar_values(self) -> [RPVJet]:
    return 'NotImplemented' # FIXME

  def match(self) -> [RPVJet]:
    # Protections
    if self.properties['DeltaRcut'] != self.__properties['DeltaRcut'] and self.properties['RecomputeDeltaRvalues']:
      print('ERROR: DeltaRcut was set but RecomputeDeltaRvalues is False, exiting')
      sys.exit(1)
    if not self.jets:
      print('ERROR: No jets were provided, exiting')
      sys.exit(1)
    if not self.partons:
      print('ERROR: No partons were provided, exiting')
      sys.exit(1)
    # Run appropriate matching
    if self.properties['RecomputeDeltaRvalues']:
      self.__match_recompute_deltar_values()
    else:
      self.__match_use_deltar_values_from_ft()
