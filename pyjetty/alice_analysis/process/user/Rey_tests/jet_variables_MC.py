#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information and do jet-finding
  
  Most code adapted from rg analysis by James Mulligan (james.mulligan@berkeley.edu)
  and angularity analysis Ezra Lesser (elesser@berkeley.edu)
  Rey Cruz-Torres (reynier@lbl.gov)
"""

from __future__ import print_function

# General
import os
import sys
import argparse
import math
import time

# Data analysis and plotting
import uproot
import pandas
import numpy as np
from array import *
import ROOT
import yaml

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io, process_utils, jet_info, process_base
#from pyjetty.alice_analysis.process.user.ang_pp.helpers import deltaR, lambda_beta_kappa, pT_bin
from math import pi
from pyjetty.mputils import treewriter

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
def deltaR(pjet1, pjet2):

  # Check that there's not a problem with +/- pi in phi part
  phi1 = pjet1.phi()
  if phi1 - pjet2.phi() > 5:
    phi1 -= 2*pi
  elif pjet2.phi() - phi1 > 5:
    phi1 += 2*pi

  return np.sqrt( (pjet1.eta() - pjet2.eta())**2 + (phi1 - pjet2.phi())**2 )

################################################################
def softkeep(jet_full, jet_groomed):
  sk_idx_list = []
  for cons1 in jet_full.constituents():
     notgroomed = False
     for cons2 in jet_groomed.constituents():
       if(cons1.user_index()==cons2.user_index()):
         notgroomed = True
     if(notgroomed==False):
       #sk_idx_list.append(cons1)
       sk_idx_list.append(cons1.user_index())
  return sk_idx_list

################################################################
class process_ang_mc(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
    super(process_ang_mc, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

  #---------------------------------------------------------------
  # Main processing function
  #---------------------------------------------------------------
  def process_ang_mc(self):
   
    self.jet_matching_distance = 0.6        # Match jets with deltaR < jet_matching_distance*jetR
    self.reject_tracks_fraction = 0.04
 
    start_time = time.time() 
    
    # ------------------------------------------------------------------------
    
    # Use IO helper class to convert detector-level ROOT TTree into
    # a SeriesGroupBy object of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - start_time))
    io_det = process_io.ProcessIO(input_file=self.input_file, tree_dir="PWGHF_TreeCreator",
                                   track_tree_name="tree_Particle", event_tree_name="tree_event_char")
    df_fjparticles_det = io_det.load_data(self.reject_tracks_fraction)
    self.nEvents_det = len(df_fjparticles_det.index)
    self.nTracks_det = len(io_det.track_df.index)
    print('--- {} seconds ---'.format(time.time() - start_time))
    
    # ------------------------------------------------------------------------

    # Use IO helper class to convert truth-level ROOT TTree into
    # a SeriesGroupBy object of fastjet particles per event
    io_truth = process_io.ProcessIO(input_file=self.input_file, tree_dir="PWGHF_TreeCreator",
                                     track_tree_name="tree_Particle_gen", 
                                     event_tree_name="tree_event_char")
    df_fjparticles_truth = io_truth.load_data()
    self.nEvents_truth = len(df_fjparticles_truth.index)
    self.nTracks_truth = len(io_truth.track_df.index)
    print('--- {} seconds ---'.format(time.time() - start_time))
    
    # ------------------------------------------------------------------------

    # Now merge the tw_deto SeriesGroupBy to create a groupby df with [ev_id, run_number, fj_1, fj_2]
    # (Need a structure such that we can iterate event-by-event through both fj_1, fj_2 simultaneously)
    print('Merge det-level and truth-level into a single dataframe grouped by event...')
    self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
    self.df_fjparticles.columns = ['fj_particles_det', 'fj_particles_truth']
    print('--- {} seconds ---'.format(time.time() - start_time))

    # ------------------------------------------------------------------------

    # Initialize configuration and histograms
    self.initialize_config()
    print(self)
    
    # Find jets and fill histograms
    print('Find jets...')
    self.analyzeEvents()
    
    # Plot histograms
    #print('Save histograms...')
    #process_base.ProcessBase.save_output_objects(self)
    
    print('--- {} seconds ---'.format(time.time() - start_time))
  
  #---------------------------------------------------------------
  # Initialize config file into class members
  #---------------------------------------------------------------
  def initialize_config(self):
    
    # Set configuration for analysis
    self.jetR_list = [0.4,0.2]

    # SoftDrop configuration
    self.sd_zcut = [0.05,0.1,0.2]
    self.sd_beta_par = [0,1]

  #---------------------------------------------------------------
  # Main function to loop through and analyze events
  #---------------------------------------------------------------
  def analyzeEvents(self):
       
    fj.ClusterSequence.print_banner()
    print()
   
    # Create an output root file and output trees
    outf = ROOT.TFile('RTreeWriter_mc.root', 'recreate')
    outf.cd()

    # detector level
    tw_det = []
    tw_det_sd_2d = []
    tw_det_sd_3d = []
    tw_det_sd = []
    tw_det_sk_2d = []
    tw_det_sk_3d = []
    tw_det_sk = []

    # truth level
    tw_truth = []
    tw_truth_sd_2d = []
    tw_truth_sd_3d = []
    tw_truth_sd = []
    tw_truth_sk_2d = []
    tw_truth_sk_3d = []
    tw_truth_sk = []

    # matched
    tw_match = []
    tw_match_sd_2d = []
    tw_match_sd_3d = []
    tw_match_sd = []
    tw_match_sk_2d = []
    tw_match_sk_3d = []
    tw_match_sk = []

    for njetR in range(len(self.jetR_list)):
      tw_det.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_det_R{}'.format(self.jetR_list[njetR]),'Tree_R{}'.format(self.jetR_list[njetR])))) 
      tw_truth.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_truth_R{}'.format(self.jetR_list[njetR]),'Tree_truth_R{}'.format(self.jetR_list[njetR]))))
      tw_match.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_match_R{}'.format(self.jetR_list[njetR]),'Tree_match_R{}'.format(self.jetR_list[njetR]))))

      # Tree to store results with softdrop
      for itm_z in range(len(self.sd_zcut)):
        for itm_b in range(len(self.sd_beta_par)):
          tw_det_sd.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_det_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
          tw_det_sk.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_det_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
          tw_truth_sd.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_truth_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_truth_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
          tw_truth_sk.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_truth_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_truth_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
          tw_match_sd.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_match_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_match_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
          tw_match_sk.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_match_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_match_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
        tw_det_sd_2d.append(tw_det_sd)
        tw_det_sk_2d.append(tw_det_sk)
        tw_truth_sd_2d.append(tw_truth_sd)
        tw_truth_sk_2d.append(tw_truth_sk)
        tw_match_sd_2d.append(tw_match_sd)
        tw_match_sk_2d.append(tw_match_sk)
        tw_det_sd = []
        tw_det_sk = []
        tw_truth_sd = []
        tw_truth_sk = []
        tw_match_sd = []
        tw_match_sk = []
      tw_det_sd_3d.append(tw_det_sd_2d)
      tw_det_sk_3d.append(tw_det_sk_2d)
      tw_truth_sd_3d.append(tw_truth_sd_2d)
      tw_truth_sk_3d.append(tw_truth_sk_2d)
      tw_match_sd_3d.append(tw_match_sd_2d)
      tw_match_sk_3d.append(tw_match_sk_2d)
      tw_det_sd_2d = []
      tw_det_sk_2d = []
      tw_truth_sd_2d = []
      tw_truth_sk_2d = []
      tw_match_sd_2d = []
      tw_match_sk_2d = []

    # -----------------------------------------------------
    # Loop over jet radii list
    for jetR in self.jetR_list:
    
      jetRidx = self.jetR_list.index(jetR)
 
      # Set jet definition and a jet selector
      jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
      jet_selector_det = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR)
      jet_selector_truth_matched = fj.SelectorPtMin(5.0)
      print('jet definition is:', jet_def)
      print('jet selector for det-level is:', jet_selector_det,'\n')
      print('jet selector for truth-level matches is:', jet_selector_truth_matched,'\n')

      # Define SoftDrop settings
      sd_list = []
      sd_list_2d = []
      for sd_z in self.sd_zcut:
        for sd_par in self.sd_beta_par:
          sd_list.append(fjcontrib.SoftDrop(sd_par, sd_z, jetR))
        sd_list_2d.append(sd_list)
        sd_list = []

      for sd_z in self.sd_zcut:
        for sd_itm in range(len(sd_list)):
          print('SoftDrop groomer is: {}'.format(sd_list_2d[sd_z][sd_itm].description()));

      for fj_particles_det, fj_particles_truth in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth']):
        # Check that the entries exist appropriately
        # (need to check how this can happen -- but it is only a tiny fraction of events)
        if type(fj_particles_det) != fj.vectorPJ or type(fj_particles_truth) != fj.vectorPJ:
          print('fj_particles type mismatch -- skipping event')
          #return
          continue

        # Do jet finding
        cs_det = fj.ClusterSequence(fj_particles_det, jet_def)
        jets_det = fj.sorted_by_pt(cs_det.inclusive_jets())
        jets_det_selected = jet_selector_det(jets_det)

        cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)
        jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
        jets_truth_selected = jet_selector_det(jets_truth)
        jets_truth_selected_matched = jet_selector_truth_matched(jets_truth)

        # Loop through jets
        jetR = jet_def.R()

        # ---------------------------------------------------------------------------------------
        # Loop through detector-level jets
        for jet_det in jets_det_selected:
                                                                          
          # Check additional acceptance criteria
          if not self.utils.is_det_jet_accepted(jet_det):
            continue

          for constit in jet_det.constituents():
            theta_i_jet = jet_det.delta_R(constit)

            tw_det[jetRidx].fill_branch("constit_over_jet_pt",constit.pt()/jet_det.pt()  )
            tw_det[jetRidx].fill_branch("theta_constit_jet"  ,theta_i_jet                )
            tw_det[jetRidx].fill_branch("n_constituents"     ,len(jet_det.constituents()))
            tw_det[jetRidx].fill_branch("jet_pt"             ,jet_det.pt()               )
          
            tw_det[jetRidx].fill_tree()

          # Soft-drop groomed jet
          for itm_z in range(len(self.sd_zcut)):
            for itm in range(len(self.sd_beta_par)):
              sd_jet_det = (sd_list_2d[itm_z][itm]).result(jet_det)
  
              for constit in sd_jet_det.constituents(): 
                sd_theta_i_jet = jet_det.delta_R(constit)
  
                tw_det_sd_3d[jetRidx][itm_z][itm].fill_branch("constit_over_jet_pt",constit.pt()/jet_det.pt()     )
                tw_det_sd_3d[jetRidx][itm_z][itm].fill_branch("theta_constit_jet"  ,sd_theta_i_jet                )
                tw_det_sd_3d[jetRidx][itm_z][itm].fill_branch("n_constituents"     ,len(sd_jet_det.constituents()))
                tw_det_sd_3d[jetRidx][itm_z][itm].fill_branch("jet_pt"             ,jet_det.pt()                  )
                         
                tw_det_sd_3d[jetRidx][itm_z][itm].fill_tree()
  
              # 'Soft-kept' stuff
              sk_idx = softkeep(jet_det,sd_jet_det) # List with indices of hadrons that were groomed away
              for constit in jet_det.constituents():
                if(constit.user_index() in sk_idx):
                  sk_theta_i_jet = jet_det.delta_R(constit)
  
                  tw_det_sk_3d[jetRidx][itm_z][itm].fill_branch("constit_over_jet_pt",constit.pt()/jet_det.pt()  )
                  tw_det_sk_3d[jetRidx][itm_z][itm].fill_branch("theta_constit_jet"  ,sk_theta_i_jet             )
                  tw_det_sk_3d[jetRidx][itm_z][itm].fill_branch("n_constituents"     ,len(sk_idx)                )
                  tw_det_sk_3d[jetRidx][itm_z][itm].fill_branch("jet_pt"             ,jet_det.pt()               )
                           
                  tw_det_sk_3d[jetRidx][itm_z][itm].fill_tree()
  
        # ---------------------------------------------------------------------------------------
        # Loop through truth-level jets
        for jet_truth in jets_truth_selected:

          for constit in jet_truth.constituents():
            theta_i_jet = jet_truth.delta_R(constit)

            tw_truth[jetRidx].fill_branch("constit_over_jet_pt",constit.pt()/jet_truth.pt()  )
            tw_truth[jetRidx].fill_branch("theta_constit_jet"  ,theta_i_jet                  )
            tw_truth[jetRidx].fill_branch("n_constituents"     ,len(jet_truth.constituents()))
            tw_truth[jetRidx].fill_branch("jet_pt"             ,jet_truth.pt()               )

            tw_truth[jetRidx].fill_tree()

          # Soft-drop groomed jet
          for itm_z in range(len(self.sd_zcut)):
            for itm in range(len(self.sd_beta_par)):
              sd_jet_truth = (sd_list_2d[itm_z][itm]).result(jet_truth)
 
              for constit in sd_jet_truth.constituents():
                sd_theta_i_jet = jet_truth.delta_R(constit)
 
                tw_truth_sd_3d[jetRidx][itm_z][itm].fill_branch("constit_over_jet_pt",constit.pt()/jet_truth.pt()     )
                tw_truth_sd_3d[jetRidx][itm_z][itm].fill_branch("theta_constit_jet"  ,sd_theta_i_jet                  )
                tw_truth_sd_3d[jetRidx][itm_z][itm].fill_branch("n_constituents"     ,len(sd_jet_truth.constituents()))
                tw_truth_sd_3d[jetRidx][itm_z][itm].fill_branch("jet_pt"             ,jet_truth.pt()                  )
 
                tw_truth_sd_3d[jetRidx][itm_z][itm].fill_tree()
 
              # 'Soft-kept' stuff
              sk_idx = softkeep(jet_truth,sd_jet_truth) # List with indices of hadrons that were groomed away
              for constit in jet_truth.constituents():
                if(constit.user_index() in sk_idx):
                  sk_theta_i_jet = jet_truth.delta_R(constit)
 
                  tw_truth_sk_3d[jetRidx][itm_z][itm].fill_branch("constit_over_jet_pt",constit.pt()/jet_truth.pt()  )
                  tw_truth_sk_3d[jetRidx][itm_z][itm].fill_branch("theta_constit_jet"  ,sk_theta_i_jet               )
                  tw_truth_sk_3d[jetRidx][itm_z][itm].fill_branch("n_constituents"     ,len(sk_idx)                  )
                  tw_truth_sk_3d[jetRidx][itm_z][itm].fill_branch("jet_pt"             ,jet_truth.pt()               )
 
                  tw_truth_sk_3d[jetRidx][itm_z][itm].fill_tree()

        # ---------------------------------------------------------------------------------------
        # Loop through jets and set jet matching candidates for each jet in user_info     
        # Adapted from function 'set_matching_candidates' in 'process/base/process_base.py'
        '''
        for jet_det in jets_det_selected:
          for jet_truth in jets_truth_selected_matched:
            deltaR_det_truth = jet_det.delta_R(jet_truth)
            # Add a matching candidate to the list if it is within the geometrical cut
            if deltaR_det_truth < self.jet_matching_distance*jetR:
              print('matched')
        '''      

    outf.Write()
    outf.Close()

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

  analysis = process_ang_mc(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_ang_mc()
