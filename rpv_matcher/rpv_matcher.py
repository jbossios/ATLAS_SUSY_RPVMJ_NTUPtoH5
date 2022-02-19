
# TODO FIXME
# Support the addition of jets/partons/fsrs by passing values
# Create Matcher class??

import ROOT
import sys
from typing import Union

class RPVJet(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.is_matched = False
    self.match_type = 'None' # options: 'None', 'Parton', 'FSR'
    self.match_parton_index = -1
    self.match_pdgID = -1
    self.match_barcode = -1
    self.match_parent_barcode = -1

class RPVParton(ROOT.TLorentzVector):
  def __init__(self):
    ROOT.TLorentzVector.__init__(self)
    self.parent_barcode = -999
    self.barcode = -999
    self.pdgID = -999

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
    print('INFO: Jets will be matched to partons{} using DeltaR values from FactoryTools'.format(' and FSRs' if self.fsrs else ''))
    print('INFO: Will return {} jets'.format('only matched' if self.__properties['ReturnOnlyMatched'] else 'all'))
    print('NotImplemented') # FIXME
    return self.jets # FIXME
  
  def __match_recompute_deltar_values(self) -> [RPVJet]:
    print('INFO: Jets will be matched to partons{} computing DeltaR values using a maximum DeltaR value of {}'.format(' and FSRs' if self.fsrs else '', self.__properties['DeltaRcut']))
    print('INFO: Will return {} jets'.format('only matched' if self.__properties['ReturnOnlyMatched'] else 'all'))
    print('NotImplemented') # FIXME
    return self.jets # FIXME

  def match(self) -> [RPVJet]:
    # Protections
    if self.properties['DeltaRcut'] != self.__properties['DeltaRcut'] and not self.properties['RecomputeDeltaRvalues']:
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
      return self.__match_recompute_deltar_values()
    else:
      return self.__match_use_deltar_values_from_ft()
