#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information
  and do jet-finding, and save response histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
from array import *

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjtools

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.mputils import CEventSubtractor

################################################################
class ProcessMC_subjet_z(process_mc_base.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_subjet_z, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects_R(self, jetR):
      
    if not self.is_pp:
    
      for R_max in self.max_distance:

        for subjetR in self.obs_settings[self.observable]:
      
          name = 'hDeltaR_combined_ppdet_{}_R{}_{}_Rmax{}'.format('subjet_z', jetR, subjetR, R_max)
          h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
          setattr(self, name, h)
          
          name = 'hDeltaR_ppdet_pptrue_{}_R{}_{}_Rmax{}'.format('subjet_z', jetR, subjetR, R_max)
          h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
          setattr(self, name, h)
          
      #for subjetR in self.obs_settings[self.observable]:
      
        # Create prong matching histograms
        #self.create_prong_matching_histograms(jetR, subjetR)

    for subjetR in self.obs_settings[observable]:
      
      name = 'hDeltaR_All_{}_R{}_{}'.format(observable, jetR, subjetR)
      h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
      setattr(self, name, h)
    
      name = 'hResidual_JetPt_{}_R{}_{}'.format(observable, jetR, subjetR)
      h = ROOT.TH2F(name, name, 300, 0, 300, 200, -1.0, 1.0)
      h.GetXaxis().SetTitle('p_{T,truth}')
      h.GetYaxis().SetTitle('#frac{z_{det}-z_{truth}}{z_{truth}}')
      setattr(self, name, h)
      
      # Create THn of response for subjet z
      dim = 4;
      title = ['p_{T,det}', 'p_{T,truth}', 'z_{det}', 'z_{truth}']
      nbins = [30, 30, 100, 50]
      min = [0., 0., 0., 0.]
      max = [150., 300., 1., 1.]
      name = 'hResponse_JetPt_{}_R{}_{}'.format(observable, jetR, subjetR)
      self.create_thn(name, title, dim, nbins, min, max)

  #---------------------------------------------------------------
  # Create theta_g response histograms
  #---------------------------------------------------------------
  def create_prong_matching_histograms(self, jetR, subjetR):
  
    prong_list = ['leading', 'subleading']
    match_list = ['leading', 'subleading', 'ungroomed', 'outside']

    for R_max in self.max_distance:
      for prong in prong_list:
        for match in match_list:

          name = 'hProngMatching_{}_{}_JetPt_R{}_{}_Rmax{}'.format(prong, match, jetR, grooming_label, R_max)
          h = ROOT.TH3F(name, name, 20, 0, 200, 15, -0.4, 1.1, 20, 0., 2*jetR)
          h.GetXaxis().SetTitle('p_{T,truth}')
          h.GetYaxis().SetTitle('Prong matching fraction')
          h.GetZaxis().SetTitle('#Delta R_{prong}')
          setattr(self, name, h)
          
          name = 'hProngMatching_{}_{}_JetPtDet_R{}_{}_Rmax{}'.format(prong, match, jetR, grooming_label, R_max)
          h = ROOT.TH3F(name, name, 20, 0, 200, 15, -0.4, 1.1, 20, 0., 2*jetR)
          h.GetXaxis().SetTitle('p_{T,pp-det}')
          h.GetYaxis().SetTitle('Prong matching fraction')
          h.GetZaxis().SetTitle('#Delta R_{prong}')
          setattr(self, name, h)
          
          name = 'hProngMatching_{}_{}_JetPtZ_R{}_{}_Rmax{}'.format(prong, match, jetR, grooming_label, R_max)
          h = ROOT.TH3F(name, name, 20, 0, 200, 15, -0.4, 1.1, 50, -0.5, 0.5)
          h.GetXaxis().SetTitle('p_{T,truth}')
          h.GetYaxis().SetTitle('Prong matching fraction')
          h.GetZaxis().SetTitle('#Delta z_{prong}')
          setattr(self, name, h)

      name = 'hProngMatching_subleading-leading_correlation_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)
      h = ROOT.TH3F(name, name, 20, 0, 200, 15, -0.4, 1.1, 15, -0.4, 1.1)
      h.GetXaxis().SetTitle('p_{T,pp-det}')
      h.GetYaxis().SetTitle('Prong matching fraction, leading_subleading')
      h.GetZaxis().SetTitle('Prong matching fraction, subleading_leading')
      setattr(self, name, h)

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # Fill 2D histogram of (pt, obs)
  #---------------------------------------------------------------
  def fill_observable_histograms(self, hname, jet, jet_groomed_lund, jetR, obs_setting,
                                 grooming_setting, obs_label, jet_pt_ungroomed):
  
    # For a given jet, find subjets of a given radius, and fill histograms
    cs_subjet = fj.ClusterSequence(jet.constituents(), self.subjet_def[obs_setting])
    subjets = fj.sorted_by_pt(cs_subjet.inclusive_jets())
    for subjet in subjets:
      z = subjet.pt() / jet.pt()
      getattr(self,  hname.format(self.observable, jetR, obs_label)).Fill(jet_pt_ungroomed, z)
      
  #---------------------------------------------------------------
  # Fill matched jet histograms
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, jet_det, jet_det_groomed_lund, jet_truth,
                                  jet_truth_groomed_lund, jet_pp_det, jetR,
                                  obs_setting, grooming_setting, obs_label,
                                  jet_pt_det_ungroomed, jet_pt_truth_ungroomed, R_max, suffix):
       
    # Find subjets
    cs_subjet_det = fj.ClusterSequence(jet_det.constituents(), self.subjet_def[subjetR])
    subjets_det = fj.sorted_by_pt(cs_subjet_det.inclusive_jets())

    cs_subjet_truth = fj.ClusterSequence(jet_truth.constituents(), self.subjet_def[subjetR])
    subjets_truth = fj.sorted_by_pt(cs_subjet_truth.inclusive_jets())
    
    if not self.is_pp:
      cs_subjet_det_pp = fj.ClusterSequence(jet_pp_det.constituents(), self.subjet_def[subjetR])
      subjets_det_pp = fj.sorted_by_pt(cs_subjet_truth.inclusive_jets())

    # Loop through subjets and set subjet matching candidates for each subjet in user_info
    if self.is_pp:
        [[self.set_matching_candidates(subjet_det, subjet_truth, subjetR, 'hDeltaR_All_subjet_z_R{}_{}'.format(jetR, subjetR)) for subjet_truth in subjets_truth] for subjet_det in subjets_det]
    else:
        # First fill the combined-to-pp matches, then the pp-to-pp matches
        [[self.set_matching_candidates(subjet_det_combined, subjet_det_pp, subjetR, 'hDeltaR_combined_ppdet_subjet_z_R{}_{}'.format(jetR, subjetR), fill_jet1_matches_only=True) for subjet_det_pp in subjets_det_pp] for subjet_det_combined in subjets_det]
        [[self.set_matching_candidates(subjet_det_pp, subjet_truth, subjetR, 'hDeltaR_ppdet_pptrue_subjet_z_R{}_{}'.format(jetR, subjetR)) for subjet_truth in subjets_truth] for subjet_det_pp in subjets_det_pp]
      
    # Loop through subjets and set accepted matches
    if self.is_pp:
        [self.set_matches_pp(subjet_det, 'hSubjetMatchingQA_R{}_{}'.format(jetR, subjetR)) for subjet_det in subjets_det]
    else:
        [self.set_matches_AA(subjet_det_combined, subjetR, 'hSubjetMatchingQA_R{}_{}'.format(jetR, subjetR)) for subjet_det_combined in subjets_det]

    # Loop through matches and fill histograms
    for subjet_det in subjets_det:

      if subjet_det.has_user_info():
        subjet_truth = subjet_det.python_info().match
      
        if subjet_truth:
          
          z_det = subjet_det.pt() / jet_det.pt()
          z_truth = subjet_truth.pt() / jet_truth.pt()
          
          # Fill histograms
          self.fill_response(self.observable, jetR, jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                             z_det, z_truth, obs_label, R_max, prong_match = False)
    
  #---------------------------------------------------------------
  # Do prong-matching
  #---------------------------------------------------------------
  def fill_prong_matching_histograms(self, jet_truth, jet_det, jet_det_groomed_lund, jet_pt_truth_ungroomed,
                                     jetR, grooming_setting, grooming_label, R_max):
    
    # Do grooming on pp-det jet, and get prongs
    jet_pp_det = jet_truth.python_info().match
      
    jet_pp_det_groomed_lund = self.utils.groom(jet_pp_det, grooming_setting, jetR)
    if not jet_pp_det_groomed_lund:
      return
    
    # Groomer shop returns a fjcontrib::LundGenerator
    #   The prongs can be retrieved directly from this object.
    #   If the object exists, then it has passed grooming
    jet_pp_det_prong1 = jet_pp_det_groomed_lund.harder()
    jet_pp_det_prong2 = jet_pp_det_groomed_lund.softer()
    has_parents_pp_det = jet_pp_det_groomed_lund

    # Get prongs of combined jet
    jet_combined_prong1 = jet_det_groomed_lund.harder()
    jet_combined_prong2 = jet_det_groomed_lund.softer()
    has_parents_combined = jet_det_groomed_lund
    
    # Get the fastjet::PseudoJets from the fjcontrib::LundGenerators
    jet_pp_det_groomed = jet_pp_det_groomed_lund.pair()
    jet_det_groomed = jet_det_groomed_lund.pair()
          
    if self.debug_level > 1:

        if jet_pt_truth_ungroomed > 80.:
        
            print('=======================================================')
            print('jet_pt_truth_ungroomed: {}'.format(jet_pt_truth_ungroomed))
            print('jet_pt_pp_det_ungroomed: {}'.format(jet_pp_det.pt()))
            print('jet_pt_pp_det_groomed: {}'.format(jet_pp_det_groomed.pt()))
            print('jet_pt_combined_groomed: {}'.format(jet_det_groomed.pt()))
            print('')
            print('jet_pp_det tracks: {}'.format([track.user_index() for track in jet_pp_det.constituents()]))
            print('         track pt: {}'.format([np.around(track.pt(),2) for track in jet_pp_det.constituents()]))
            print('jet_pp_det_groomed tracks: {}'.format([track.user_index() for track in jet_pp_det_groomed.constituents()]))
            print('                 track pt: {}'.format([np.around(track.pt(),2) for track in jet_pp_det_groomed.constituents()]))
            print('jet_combined groomed tracks: {}'.format([track.user_index() for track in jet_det_groomed.constituents()]))
            print('                   track pt: {}'.format([np.around(track.pt(),2) for track in jet_det_groomed.constituents()]))
            print('jet_combined ungroomed tracks: {}'.format([track.user_index() for track in jet_det.constituents()]))
            print('                     track pt: {}'.format([np.around(track.pt(),2) for track in jet_det.constituents()]))

    # Compute fraction of pt of the pp-det prong tracks that is contained in the combined-jet prong,
    # in order to have a measure of whether the combined-jet prong is the "same" prong as the pp-det prong
    deltaR_prong1 = -1.
    deltaR_prong2 = -1.
    deltaZ = -1.
    if has_parents_pp_det and has_parents_combined:
    
        # Subleading jet pt-matching
        # --------------------------
        # (1) Fraction of pt matched: subleading pp-det in subleading combined
        matched_pt_subleading_subleading = fjtools.matched_pt(jet_combined_prong2, jet_pp_det_prong2)
        
        # (2) Fraction of pt matched: subleading pp-det in leading combined
        matched_pt_subleading_leading = fjtools.matched_pt(jet_combined_prong1, jet_pp_det_prong2)
        
        # (3) Fraction of pt matched: subleading pp-det in ungroomed combined jet
        matched_pt_subleading_groomed = fjtools.matched_pt(jet_det_groomed, jet_pp_det_prong2)
        matched_pt_subleading_ungroomed = fjtools.matched_pt(jet_det, jet_pp_det_prong2)
        matched_pt_subleading_ungroomed_notgroomed = matched_pt_subleading_ungroomed - matched_pt_subleading_groomed
        
        # (4) Fraction of pt matched: subleading pp-det not in ungroomed combined jet
        matched_pt_subleading_outside = 1 - matched_pt_subleading_ungroomed

        # Leading jet pt-matching
        # --------------------------
        # (1) Fraction of pt matched: leading pp-det in subleading combined
        matched_pt_leading_subleading = fjtools.matched_pt(jet_combined_prong2, jet_pp_det_prong1)
        
        # (2) Fraction of pt matched: leading pp-det in leading combined
        matched_pt_leading_leading = fjtools.matched_pt(jet_combined_prong1, jet_pp_det_prong1)

        # (3) Fraction of pt matched: leading pp-det in ungroomed combined jet
        matched_pt_leading_groomed = fjtools.matched_pt(jet_det_groomed, jet_pp_det_prong1)
        matched_pt_leading_ungroomed = fjtools.matched_pt(jet_det, jet_pp_det_prong1)
        matched_pt_leading_ungroomed_notgroomed = matched_pt_leading_ungroomed - matched_pt_leading_groomed
        
        # (4) Fraction of pt matched: leading pp-det not in ungroomed combined jet
        matched_pt_leading_outside = 1 - matched_pt_leading_ungroomed

        # Compute delta-R between pp-det prong and combined prong
        # --------------------------
        deltaR_prong1 = jet_combined_prong1.delta_R(jet_pp_det_prong1)
        deltaR_prong2 = jet_combined_prong2.delta_R(jet_pp_det_prong2)
        deltaZ = self.zg(jet_det_groomed) - self.zg(jet_pp_det_groomed)
        
        if self.debug_level > 1:
        
            if jet_pt_truth_ungroomed > 80.:
            
                print('subleading prong tracks -- combined: {}'.format([track.user_index() for track in jet_combined_prong2.constituents()]))
                print('subleading prong tracks -- pp-det: {}'.format([track.user_index() for track in jet_pp_det_prong2.constituents()]))
                print('leading prong tracks -- combined: {}'.format([track.user_index() for track in jet_combined_prong1.constituents()]))
                print('leading prong tracks -- pp-det: {}'.format([track.user_index() for track in jet_pp_det_prong1.constituents()]))
                print('')
                print('leading_prong_pt: {}'.format(jet_combined_prong1.pt()))
                print('matched_pt_leading_subleading fraction: {}'.format(matched_pt_leading_subleading))
                print('matched_pt_leading_leading fraction: {}'.format(matched_pt_leading_leading))
                print('matched_pt_leading_ungroomed_notgroomed fraction: {}'.format(matched_pt_leading_ungroomed_notgroomed))
                print('matched_pt_leading_outside fraction: {}'.format(matched_pt_leading_outside))
                print('')
                print('subleading_prong_pt: {}'.format(jet_combined_prong2.pt()))
                print('matched_pt_subleading_subleading fraction: {}'.format(matched_pt_subleading_subleading))
                print('matched_pt_subleading_leading fraction: {}'.format(matched_pt_subleading_leading))
                print('matched_pt_subleading_ungroomed_notgroomed fraction: {}'.format(matched_pt_subleading_ungroomed_notgroomed))
                print('matched_pt_subleading_outside fraction: {}'.format(matched_pt_subleading_outside))
                print('')
                print('deltaR_prong1: {}'.format(deltaR_prong1))
                print('deltaR_prong2: {}'.format(deltaR_prong2))

    elif has_parents_pp_det: # pp-det passed grooming, but combined jet failed grooming
        matched_pt_leading_leading = matched_pt_leading_subleading = matched_pt_leading_ungroomed_notgroomed = matched_pt_leading_outside = matched_pt_subleading_leading = matched_pt_subleading_subleading = matched_pt_subleading_ungroomed_notgroomed = matched_pt_subleading_outside = -0.1
        
    elif has_parents_combined: # combined jet passed grooming, but pp-det failed grooming
        matched_pt_leading_leading = matched_pt_leading_subleading = matched_pt_leading_ungroomed_notgroomed = matched_pt_leading_outside = matched_pt_subleading_leading = matched_pt_subleading_subleading = matched_pt_subleading_ungroomed_notgroomed = matched_pt_subleading_outside = -0.2
        
    else: # both pp-det and combined jet failed SoftDrop
        matched_pt_leading_leading = matched_pt_leading_subleading = matched_pt_leading_ungroomed_notgroomed = matched_pt_leading_outside = matched_pt_subleading_leading = matched_pt_subleading_subleading = matched_pt_subleading_ungroomed_notgroomed = matched_pt_subleading_outside = -0.3

    # Leading prong
    getattr(self, 'hProngMatching_leading_leading_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_leading, deltaR_prong1)
    getattr(self, 'hProngMatching_leading_subleading_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_subleading, deltaR_prong1)
    getattr(self, 'hProngMatching_leading_ungroomed_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_ungroomed_notgroomed, deltaR_prong1)
    getattr(self, 'hProngMatching_leading_outside_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_outside, deltaR_prong1)
    
    getattr(self, 'hProngMatching_leading_leading_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_leading_leading, deltaR_prong1)
    getattr(self, 'hProngMatching_leading_subleading_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_leading_subleading, deltaR_prong1)
    getattr(self, 'hProngMatching_leading_ungroomed_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_leading_ungroomed_notgroomed, deltaR_prong1)
    getattr(self, 'hProngMatching_leading_outside_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_leading_outside, deltaR_prong1)
    
    getattr(self, 'hProngMatching_leading_leading_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_leading, deltaZ)
    getattr(self, 'hProngMatching_leading_subleading_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_subleading, deltaZ)
    getattr(self, 'hProngMatching_leading_ungroomed_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_ungroomed_notgroomed, deltaZ)
    getattr(self, 'hProngMatching_leading_outside_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_leading_outside, deltaZ)

    # Subleading prong
    getattr(self, 'hProngMatching_subleading_leading_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_leading, deltaR_prong2)
    getattr(self, 'hProngMatching_subleading_subleading_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_subleading, deltaR_prong2)
    getattr(self, 'hProngMatching_subleading_ungroomed_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_ungroomed_notgroomed, deltaR_prong2)
    getattr(self, 'hProngMatching_subleading_outside_JetPt_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_outside, deltaR_prong2)

    getattr(self, 'hProngMatching_subleading_leading_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_subleading_leading, deltaR_prong2)
    getattr(self, 'hProngMatching_subleading_subleading_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_subleading_subleading, deltaR_prong2)
    getattr(self, 'hProngMatching_subleading_ungroomed_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_subleading_ungroomed_notgroomed, deltaR_prong2)
    getattr(self, 'hProngMatching_subleading_outside_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_subleading_outside, deltaR_prong2)

    getattr(self, 'hProngMatching_subleading_leading_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_leading, deltaZ)
    getattr(self, 'hProngMatching_subleading_subleading_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_subleading, deltaZ)
    getattr(self, 'hProngMatching_subleading_ungroomed_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_ungroomed_notgroomed, deltaZ)
    getattr(self, 'hProngMatching_subleading_outside_JetPtZ_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pt_truth_ungroomed, matched_pt_subleading_outside, deltaZ)
    
    # Plot correlation of matched pt fraction for leading-subleading and subleading-leading
    getattr(self, 'hProngMatching_subleading-leading_correlation_JetPtDet_R{}_{}_Rmax{}'.format(jetR, grooming_label, R_max)).Fill(jet_pp_det.pt(), matched_pt_leading_subleading, matched_pt_subleading_leading)
    
    subleading_match = (matched_pt_subleading_subleading > 0.5)
    leading_match = (matched_pt_leading_leading > 0.5)
    prong_match = subleading_match and leading_match
    return prong_match

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process MC')
  parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='config/analysis_config.yaml',
                      help="Path of config file for analysis")
  parser.add_argument('-o', '--outputDir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for output to be written to')
  
  # Parse the arguments
  args = parser.parse_args()
  
  print('Configuring...')
  print('inputFile: \'{0}\''.format(args.inputFile))
  print('configFile: \'{0}\''.format(args.configFile))
  print('ouputDir: \'{0}\"'.format(args.outputDir))

  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessMC_subjet_z(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()