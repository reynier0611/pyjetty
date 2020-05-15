#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of track information and do jet-finding.
  
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
import ROOT
import yaml
from array import *

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io, process_utils, process_base
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
class process_ang_data(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
    super(process_ang_data, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    self.initialize_config()

  #---------------------------------------------------------------
  # Main processing function
  #---------------------------------------------------------------
  def process_ang_data(self):
    
    start_time = time.time()

    # Use IO helper class to convert ROOT TTree into a SeriesGroupBy object
    # of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - start_time))
    io = process_io.ProcessIO(input_file=self.input_file, track_tree_name='tree_Particle')
    self.df_fjparticles = io.load_data()
    self.nEvents = len(self.df_fjparticles.index)
    self.nTracks = len(io.track_df.index)
    print('--- {} seconds ---'.format(time.time() - start_time))

    # Initialize configuration and histograms
    self.initialize_config()
    print(self)

    # Find jets and fill histograms
    print('Find jets...')
    self.analyzeEvents()

    # Plot histograms
    print('Save histograms...')
    #process_base.ProcessBase.save_output_objects(self)

    print('--- {} seconds ---'.format(time.time() - start_time))

  #---------------------------------------------------------------
  # Initialize config file into class members
  #---------------------------------------------------------------
  def initialize_config(self):

    # Set configuration for analysis
    self.jetR_list = [0.4,0.2,0.1,0.3,0.5]

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
    outf = ROOT.TFile('RTreeWriter_test.root', 'recreate')
    outf.cd()

    tw = []
    tw_sd_2d = []
    tw_sd_3d = []
    tw_sd = []
    tw_sk_2d = []
    tw_sk_3d = []
    tw_sk = []

    for njetR in range(len(self.jetR_list)):
      tw.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_R{}'.format(self.jetR_list[njetR]),'Tree_R{}'.format(self.jetR_list[njetR])))) 

      # Tree to store results with softdrop
      for itm_z in range(len(self.sd_zcut)):
        for itm_b in range(len(self.sd_beta_par)):
          tw_sd.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_R{}_sd_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
          tw_sk.append(treewriter.RTreeWriter(tree=ROOT.TTree('Tree_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]),'Tree_R{}_sk_z_{}_beta_{}'.format(self.jetR_list[njetR],self.sd_zcut[itm_z],self.sd_beta_par[itm_b]))))
        tw_sd_2d.append(tw_sd)
        tw_sk_2d.append(tw_sk)
        tw_sd = []
        tw_sk = []
      tw_sd_3d.append(tw_sd_2d)
      tw_sk_3d.append(tw_sk_2d)
      tw_sd_2d = []
      tw_sk_2d = []

    # -----------------------------------------------------
    # Loop over jet radii list
    for jetR in self.jetR_list:

      jetRidx = self.jetR_list.index(jetR)

      # Set jet definition and a jet selector
      jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
      jet_selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR)
      print('jet definition is:', jet_def)
      print('jet selector is:', jet_selector,'\n')
      
      # Define SoftDrop settings
      sd_list = []
      sd_list_2d = []
      for sd_z in self.sd_zcut:
        for sd_par in self.sd_beta_par:
          sd_list.append(fjcontrib.SoftDrop(sd_par,sd_z, jetR))
        sd_list_2d.append(sd_list)
        sd_list = []

      for sd_z in self.sd_zcut:
        for sd_itm in range(len(sd_list)):
          print('SoftDrop groomer is: {}'.format(sd_list_2d[sd_z][sd_itm].description()));

      for fj_particles in self.df_fjparticles:
        # Do jet finding
        cs = fj.ClusterSequence(fj_particles, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = jet_selector(jets)
                                                                          
        # Loop through jets
        jetR = jet_def.R()
        for jet in jets_selected:
                                                                          
          # Check additional acceptance criteria
          if not self.utils.is_det_jet_accepted(jet):
            continue
                                                                          
          for constit in jet.constituents():
            theta_i_jet = float(deltaR(constit,jet))

            #print(theta_i_jet,' ',jet.delta_R(constit),' ',theta_i_jet-jet.delta_R(constit))

            tw[jetRidx].fill_branch("constit_over_jet_pt",constit.pt()/jet.pt()  )
            tw[jetRidx].fill_branch("theta_constit_jet"  ,theta_i_jet            )
            tw[jetRidx].fill_branch("n_constituents"     ,len(jet.constituents()))
            tw[jetRidx].fill_branch("jet_pt"             ,jet.pt()               )

            tw[jetRidx].fill_tree()

          # Soft-drop groomed jet
          for itm_z in range(len(self.sd_zcut)):
            for itm in range(len(self.sd_beta_par)):
              sd_jet = (sd_list_2d[itm_z][itm]).result(jet)
           
              for constit in sd_jet.constituents(): 
                sd_theta_i_jet = float(deltaR(constit,jet))
           
                tw_sd_3d[jetRidx][itm_z][itm].fill_branch("constit_over_jet_pt",constit.pt()/jet.pt()     )
                tw_sd_3d[jetRidx][itm_z][itm].fill_branch("theta_constit_jet"  ,sd_theta_i_jet            )
                tw_sd_3d[jetRidx][itm_z][itm].fill_branch("n_constituents"     ,len(sd_jet.constituents()))
                tw_sd_3d[jetRidx][itm_z][itm].fill_branch("jet_pt"             ,jet.pt()                  )
           
                tw_sd_3d[jetRidx][itm_z][itm].fill_tree()
           
              # 'Soft-kept' stuff
              sk_idx = softkeep(jet,sd_jet) # List with indices of hadrons that were groomed away
              for constit in jet.constituents():
                if(constit.user_index() in sk_idx):
                  sk_theta_i_jet = float(deltaR(constit,jet))
           
                  tw_sk_3d[jetRidx][itm_z][itm].fill_branch("constit_over_jet_pt",constit.pt()/jet.pt()  )
                  tw_sk_3d[jetRidx][itm_z][itm].fill_branch("theta_constit_jet"  ,sk_theta_i_jet         )
                  tw_sk_3d[jetRidx][itm_z][itm].fill_branch("n_constituents"     ,len(sk_idx)            )
                  tw_sk_3d[jetRidx][itm_z][itm].fill_branch("jet_pt"             ,jet.pt()               )
           
                  tw_sk_3d[jetRidx][itm_z][itm].fill_tree()

    outf.Write()
    outf.Close()

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Plot analysis histograms')
  parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='config/angularity.yaml',
                      help="Path of config file for jetscape analysis")
  parser.add_argument('-o', '--outputDir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for QA plots to be written to')
  
  # Parse the arguments
  args = parser.parse_args()
  
  print('Configuring...')
  print('inputFile: \'{0}\''.format(args.inputFile))
  print('configFile: \'{0}\''.format(args.configFile))
  print('ouputDir: \'{0}\"'.format(args.outputDir))
  print('----------------------------------------------------------------')
  
  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = process_ang_data(input_file=args.inputFile, config_file=args.configFile,output_dir=args.outputDir) 
  analysis.process_ang_data()
