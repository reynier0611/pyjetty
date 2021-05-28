#! /usr/bin/env python

"""
Code to do theory folding in order to compare to the measured distributions
The class 'TheoryFolding' below inherits from the 'TheoryFolding' class in:
pyjetty/alice_analysis/analysis/user/substructure/run_fold_theory.py
"""

import sys
import os
import argparse
from array import *
import numpy as np
import ROOT
ROOT.gSystem.Load("$HEPPY_DIR/external/roounfold/roounfold-current/lib/libRooUnfold.so")
import yaml

from pyjetty.alice_analysis.analysis.user.substructure import run_fold_theory

# Load pyjetty ROOT utils
ROOT.gSystem.Load('libpyjetty_rutil')

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
################################################################
################################################################
class TheoryFolding(run_fold_theory.TheoryFolding):

  def load_theory_curves(self):
    self.theory_obs_points = {}
    self.theory_scale_vars = {}

    # Loop over each jet R specified in the config file
    for jetR in self.jetR_list:
      th_obs_R = []
      scale_var = []

      # Loop through subconfigurations to fold (e.g. in the jet-axis analysis there Standard_WTA, Standard_SD_1, ...)
      for i in range(0,len(self.obs_subconfig_list)):
        obs_setting = self.obs_settings[i]            # labels such as 'Standard_WTA'
        grooming_setting = self.grooming_settings[i]  # grooming parameters
        if grooming_setting and 'SD' in obs_setting:
          label = obs_setting[:obs_setting.find('SD_')]
          label += self.utils.grooming_label(grooming_setting)
        else:
          label = obs_setting 

        path_to_theory = os.path.join(self.theory_dir,label)

        # Assume for a given subconfiguration all files have the same observable binning and only open the first file
        pt_min = self.theory_pt_bins[0]
        pt_max = self.theory_pt_bins[1]
        th_fname = 'R_%s_pT_%i-%i.dat' % (str(jetR).replace('.', '') , int(pt_min), int(pt_max) )
        th_fname = os.path.join( path_to_theory , th_fname )

        # Open theory file and load its contents
        with open( th_fname ) as f:
          #lines = [line for line in f.read().split('\n') if line] #[0] != '#']
          lines = [line for line in f.read().split('\n') if line[0] != '#']

        th_obs = [float(line.split()[0]) for line in lines]
        th_obs_R.append(th_obs)

        n_scale_variations = len(lines[0].split())-1 # number of scale variations
        scale_var.append(n_scale_variations)

        # Up to this point we were gathering some information about the theory files:
        # (number of scale variations and observable granularity)

        pt_bins = array('d', self.theory_pt_bins)
        obs_points = array('d', th_obs )               # points provided in the theory calculation

        if self.theory_obs_bins:
          obs_bins = array('d', self.theory_obs_bins)   # bins which we want to have in the result
        else:
          obs_bins = array('d',getattr(self,'binning_'+obs_setting))

        # Add bin for underflow value (tagging fraction)
        if grooming_setting and self.use_tagging_fraction:
          obs_bins = np.insert(obs_bins, 0, -0.001)

        obs_width = np.subtract(obs_bins[1:],obs_bins[:-1])

        # -----------------------------------------------------        
        # Create histograms where theory curves will be stored
        th_hists_no_scaling = []          # Basically a copy of the theory calculations, but binned
        th_hists = []                     # Histograms that will actually be used in the folding
        hist_names = []

        # Loop over all scale variations
        for sv in range(0,n_scale_variations):
          hist_name = 'h2_input_%s_R%s_obs_pT_%s' % ( self.observable , (str)(jetR).replace('.','') , obs_setting )
          if grooming_setting:
            hist_name += '_'
            hist_name += self.utils.grooming_label(grooming_setting)
          hist_name += '_sv%i' % (sv)

          hist_name_no_scaling = hist_name + '_no_scaling'

          th_hist                     = ROOT.TH2D(hist_name                    ,';p_{T}^{jet};%s'%(self.observable), len(pt_bins)-1, pt_bins, len(obs_bins)-1, obs_bins)
          th_hist_no_scaling          = ROOT.TH2D(hist_name_no_scaling         ,';p_{T}^{jet};%s'%(self.observable), len(pt_bins)-1, pt_bins, len(obs_bins)-1, obs_bins) 

          th_hists.append(th_hist)
          hist_names.append(hist_name)
          th_hists_no_scaling.append(th_hist_no_scaling)

        # -----------------------------------------------------
        # opening theory file by file and fill histograms
        th_path = os.path.join(self.theory_dir,label)
        print('reading from files in:',label)

        # loop over pT bins
        for p, pt in enumerate(pt_bins[:-1]):
          pt_min = self.theory_pt_bins[p]
          pt_max = self.theory_pt_bins[p+1]

          # Get scale factor for this pT bin.
          # This reverses the self-normalization of 1/sigma for correct pT scaling when doing projections onto the y-axis.
          scale_f = self.pt_scale_factor_jetR(pt,pt_bins[p+1],jetR)

          # load theory file, grab the data, and fill histograms with it
          th_file = 'R_%s_pT_%i-%i.dat' % ( (str)(jetR).replace('.','') , (int)(pt_min) , (int)(pt_max) )
          th_file = os.path.join(th_path,th_file)

          # ------------------------------------------------------------------------------------------------------------
          # Load data from theory file
          with open( th_file ) as f:

            lines = [line for line in f.read().split('\n') if line[0] != '#']
            x_val = [float(line.split()[0]) for line in lines]

            # loop over scale variations and fill histograms
            for sv in range(0,n_scale_variations):
              y_val_n = [float(line.split()[sv+1]) for line in lines]

              # Interpolate the given values and return the value at the requested bin center
              y_val_bin_ctr = self.interpolate_values_linear(x_val,y_val_n,obs_bins)

              # Save content into histogram before any scaling has been applied (to compare to the theory curves and make sure everything went fine)
              for ob in range(0,len(obs_bins)-1):
                th_hists_no_scaling[sv].SetBinContent(p+1,ob+1,y_val_bin_ctr[ob])

              # Multiply by bin width and scale with pT-dependent factor
              y_val_bin_ctr = np.multiply(y_val_bin_ctr,obs_width)
              integral_y_val_bin_ctr = sum(y_val_bin_ctr)
              y_val_bin_ctr = [ val * scale_f / integral_y_val_bin_ctr for val in y_val_bin_ctr ]

              # Save scaled content into the histograms
              for ob in range(0,len(obs_bins)-1):
                th_hists[sv].SetBinContent(p+1,ob+1,y_val_bin_ctr[ob])

          f.close()

        # ------------------------------------------------------------------------------------------------------------  
        new_obs_lab = obs_setting
        if grooming_setting:
          new_obs_lab += '_'
          new_obs_lab += self.utils.grooming_label(grooming_setting)

        # ------------------------------------------------------------------------------------------------------------  
        for n_pt in range(0,len(self.final_pt_bins)-1):
          histo_list = []
          for sv in range(0,n_scale_variations):
            projection_name = 'h1_input_%s_R%s_%s_sv%i_pT_%i_%i' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,sv,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))

            # Determine the bin number that corresponds to the pT edges given           
            min_bin, max_bin = self.bin_position( self.theory_pt_bins , self.final_pt_bins[n_pt],self.final_pt_bins[n_pt+1] )

            h1_input_hist = th_hists[sv].ProjectionY(projection_name,min_bin,max_bin)
            h1_input_hist.SetTitle(projection_name)
            h1_input_hist.SetDirectory(0)
           
            # Undo the bin width scaling and set correct normalization
            norm_factor = h1_input_hist.Integral()
            if norm_factor == 0: norm_factor = 1
            h1_input_hist.Scale(1./norm_factor, "width")
           
            for b in range(0,h1_input_hist.GetNbinsX()):
              h1_input_hist.SetBinError(b+1,0)

            histo_list.append(h1_input_hist)

          # Create envelope histograms
          hist_min, hist_max = self.min_max( histo_list )

          # Rename some objects
          name_h_cent = 'h1_input_%s_R%s_%s_pT_%i_%i'     % ( self.observable,(str)(jetR).replace('.',''),new_obs_lab,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))
          name_h_min  = 'h1_min_input_%s_R%s_%s_pT_%i_%i' % ( self.observable,(str)(jetR).replace('.',''),new_obs_lab,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))
          name_h_max  = 'h1_max_input_%s_R%s_%s_pT_%i_%i' % ( self.observable,(str)(jetR).replace('.',''),new_obs_lab,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))

          h_central = histo_list[0]
          h_central.SetName(name_h_cent)
          hist_min .SetName(name_h_min )
          hist_max .SetName(name_h_max )

          # Create a graph out of these histograms
          graph_cent = self.histo_to_graph(h_central,hist_min,hist_max)
          graph_min  = ROOT.TGraph(hist_min)
          graph_max  = ROOT.TGraph(hist_max)

          graph_cent.SetName('g_input_%s_R%s_%s_pT_%i_%i'     % ( self.observable,(str)(jetR).replace('.',''),new_obs_lab,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1])))
          graph_min .SetName('g_min_input_%s_R%s_%s_pT_%i_%i' % ( self.observable,(str)(jetR).replace('.',''),new_obs_lab,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1])))
          graph_max .SetName('g_max_input_%s_R%s_%s_pT_%i_%i' % ( self.observable,(str)(jetR).replace('.',''),new_obs_lab,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1])))

          self.outfile.cd()
          h_central .Write()
          hist_min  .Write()
          hist_max  .Write()
          graph_cent.Write()
          graph_min .Write()
          graph_max .Write()

        # -----------------------------------------------------
        # Setting the filled histograms as attributes
        self.outfile.cd()
        for sv in range(0,n_scale_variations):
          setattr(self,hist_names[sv],th_hists[sv])
          
        # Only save the 2D histograms for the central scale case
        th_hists_no_scaling[0].Write()
        th_hists[0].Write()

      self.theory_obs_points[jetR] = th_obs_R
      self.theory_scale_vars[jetR] = scale_var

#----------------------------------------------------------------------
if __name__ == '__main__':

  # Define arguments
  parser = argparse.ArgumentParser(description='Folding theory predictions')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='analysis_config.yaml',
                      help='Path of config file for analysis')

  # Parse the arguments
  args = parser.parse_args()

  print('Configuring...')
  print('configFile: \'{0}\''.format(args.configFile))

  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = TheoryFolding(config_file = args.configFile)
  analysis.run_theory_folding()
