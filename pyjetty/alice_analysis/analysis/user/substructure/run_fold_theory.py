#! /usr/bin/env python
"""
run_fold_theory.py
Code to fold theory curves from a given to a desired 'level'
Adapted from Ezra Lesser's code by Rey Cruz-Torres (reynier@lbl.gov)
"""

import sys
import os
import argparse
from array import *
import numpy as np
import ROOT
ROOT.gSystem.Load("$HEPPY_DIR/external/roounfold/roounfold-current/lib/libRooUnfold.so")
import yaml

from pyjetty.alice_analysis.analysis.user.substructure import analysis_utils_obs

# Load pyjetty ROOT utils
ROOT.gSystem.Load('libpyjetty_rutil')

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
################################################################
################################################################
class TheoryFolding():
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file='', **kwargs):
    super(TheoryFolding, self).__init__(**kwargs)

    # Initialize utils class
    self.utils = analysis_utils_obs.AnalysisUtils_Obs()
    self.config_file = config_file

    # Initialize yaml config
    self.initialize_user_config()

    #print(self)
  
  #---------------------------------------------------------------
  # Initialize config file into class members
  #---------------------------------------------------------------
  def initialize_user_config(self):
    
    # Read config file
    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)
      
    self.histutils = ROOT.RUtil.HistUtils()

    if 'theory_dir' in config:
      self.jetR_list = config['jetR']
      self.observable = config['analysis_observable']
      self.obs_config_dict = config[self.observable]

      # If the user specifies certain subconfigurations to fold via the th_subconfigs parameter,
      # only fold those. Otherwise, assume we want to ufold all subconfigs in the config file
      if 'th_subconfigs' in config:
        self.obs_subconfig_list = config['th_subconfigs']
      else:
        self.obs_subconfig_list = [name for name in list(self.obs_config_dict.keys()) if 'config' in name ]

      self.obs_settings = self.utils.obs_settings(self.observable, self.obs_config_dict, self.obs_subconfig_list)
      #self.grooming_settings = self.utils.grooming_settings(self.obs_config_dict) # This only works if all  
      self.grooming_settings = [ self.get_grooming_setting(self.obs_config_dict[cf]) for cf in self.obs_subconfig_list]
      self.obs_labels = [self.utils.obs_label(self.obs_settings[i], self.grooming_settings[i]) for i in range(len(self.obs_subconfig_list))]

      # this corresponds to a label in the RM name which indicates whether we are going from parton to charged-hadron level, or
      # from hadron to charged-hadron level, ...
      self.folding_type = config['folding_type']

      # Scale histograms to account for MPI effects? Default to yes
      self.do_mpi_scaling = True
      if 'do_mpi_scaling' in config:
        self.do_mpi_scaling = config['do_mpi_scaling']

      self.theory_dir = config['theory_dir']

      # binning
      self.theory_obs_bins = None
      if 'theory_obs_bins' in config:
        self.theory_obs_bins = config['theory_obs_bins'] # Binning desired for observable

      self.theory_pt_bins  = config['theory_pt_bins' ] # pT binning of theory calculations
      self.final_pt_bins   = config['final_pt_bins'  ] # pT binning wanted for the final curves

      # response matrices for the folding, and labels describing them
      self.theory_response_fname = config['response_files']
      self.theory_response_files = [ROOT.TFile(f, 'READ') for f in config['response_files']]
      self.theory_response_labels = config['response_labels']

      # scale factors needed to scale distributions
      self.theory_pt_scale_factors_filepath = os.path.join(self.theory_dir, config['pt_scale_factors_path'])

      self.use_tagging_fraction = False
      if 'use_tagging_fraction' in config:
        self.use_tagging_fraction = config['use_tagging_fraction']

      self.output_dir = config['output_dir']
      self.output_dir_theory = os.path.join(self.output_dir, self.observable, 'theory_response') 
    else:
      print('Missing several parameters in the config file!')
      exit()

  #---------------------------------------------------------------
  # Main processing function
  #---------------------------------------------------------------
  def run_theory_folding(self):
      # Creating a root file to store results
      outfilename = os.path.join( self.theory_dir , 'folded_scet_calculations.root' )
      self.outfile = ROOT.TFile(outfilename,'recreate')
      # ------------
      print('Loading pT scale factors...')
      self.load_pt_scale_factors(self.theory_pt_scale_factors_filepath)
      # ------------
      print('Loading response matrix for folding theory predictions...')
      self.load_theory_response()
      # ------------
      print('Loading theory curves...')
      self.load_theory_curves()
      # ------------
      print("Folding theory histograms...")
      self.fold_theory()
      # ------------
      print("Undoing some scalings...")
      self.final_processing(self.do_mpi_scaling)
      # ------------
      # Closing the root file with all results from this code
      self.outfile.Close()

  #---------------------------------------------------------------
  # Loads pT scale factors from q/g fraction theory predictions
  #---------------------------------------------------------------
  def load_pt_scale_factors(self, filepath):

    for i, jetR in enumerate(self.jetR_list):
      full_path = os.path.join(filepath,'qg_fractions-ALICE-R%s.txt' % ((str)(jetR).replace('.','')) )
      # Open file and save pT distribution
      pt_li = None; val_li = None;
      with open(full_path) as f:
        lines = [line.split() for line in f.read().split('\n') if (line and line[0] != '#')]
        pt_li = [int(float(line[0])) for line in lines]
        val_li = [float(line[1]) + float(line[2]) for line in lines]

      n_entries = len(val_li)
      val_li_jetR = val_li[i*n_entries:(i+1)*n_entries]
      pt_li_jetR = pt_li[i*n_entries:(i+1)*n_entries]
      setattr(self, "pt_scale_factor_R%s" % jetR, (pt_li_jetR, val_li_jetR))

  #---------------------------------------------------------------
  # Load theory calculations
  #---------------------------------------------------------------
  def load_theory_curves(self):
    # The user needs to implement this function
    raise NotImplementedError('You must implement initialize_user_output_objects()!')

    # The theory curves are given as 1/σ dσ/d(obs). Before doing the folding,
    # we need to convert these quantities to dσ/d(obs). To do this, we scale
    # the theory curves by scale factors (loaded in the previous function)

  #---------------------------------------------------------------
  # Linear interpolation
  #---------------------------------------------------------------
  def interpolate_values_linear(self,x_val,y_val_n,new_bin_edges):
    # given the input bin edges, determine the bin center by taking the average
    bin_ctr = np.add(new_bin_edges[1:],new_bin_edges[:-1])/2.
    # do the interpolation
    return np.interp(bin_ctr,x_val,y_val_n,left=0,right=0,period=None)

  #---------------------------------------------------------------
  # Load 4D response matrices used for folding the theory predictions
  #---------------------------------------------------------------
  def load_theory_response(self):

    # Check to see if Roounfold file already exists
    if not os.path.exists(self.output_dir_theory):
      os.makedirs(self.output_dir_theory)
    roounfold_filename = os.path.join(self.output_dir_theory, 'fRoounfold.root')

    # Loop over jet R
    for jetR in self.jetR_list:
      # Loop through subconfigurations to fold
      # (e.g. Standard_WTA, Standard_SD_1, ... in the jet-axis analysis, or
      # beta = 1.5, 2, 3, ... in the angularities analysis)
      for i in range(0,len(self.obs_subconfig_list)):

        obs_setting = self.obs_settings[i]           # labels such as 'Standard_WTA'
        grooming_setting = self.grooming_settings[i] # grooming parameters
        label = self.create_label( jetR , obs_setting , grooming_setting ) 

        # loop over response files (e.g. Pythia, Herwig, ...)
        for ri, response in enumerate(self.theory_response_files):

          # Load response matrix 
          name_RM = "hResponse_JetPt_" + self.observable + "_" + self.folding_type + "_" + label
          thn = response.Get(name_RM)
          if thn == None:
            print('Could not find RM:',name_RM,'in',self.theory_response_fname[ri])
            exit()
          setattr(self, '%s_%i' % (name_RM, ri), thn)

          # Create Roounfold object
          name_roounfold_obj = '%s_Roounfold_%i' % (name_RM, ri)
          name_roounfold_thn = '%s_Rebinned_%i'  % (name_RM, ri)

          '''
          Response axes:
          ['p_{T}^{final}', 'p_{T}^{initial}', 'obs^{final}', 'obs^{initial}']
          e.g. ['p_{T}^{ch}', 'p_{T}^{h}', 'obs^{ch}', 'obs^{h}']
          
          as compared to the usual
          ['p_{T}^{initial}', 'p_{T}^{final}', 'obs^{initial}', 'obs^{final}']
          e.g. ['p_{T}^{det}', 'p_{T}^{truth}', 'obs^{det}', 'obs_{truth}']
          '''

          # If no binning was specified by the user, take the RM binning
          if self.theory_obs_bins == None:
            binning = self.return_histo_binning_1D( thn.Projection(3) )
            self.theory_obs_bins = [l for l in binning if l >= 0]
            print('WARNING: No observable binning was specified, so will take whichever binning the RM comes with.')
            print('         To change this, please, add a parameter theory_obs_bins to the config file')

          det_pt_bin_array = array('d', self.theory_pt_bins)
          tru_pt_bin_array = det_pt_bin_array
          det_obs_bin_array = array('d', self.theory_obs_bins)
          tru_obs_bin_array = det_obs_bin_array

          if grooming_setting:
            # Add bin for underflow value (tagging fraction)
            det_obs_bin_array = np.insert(det_obs_bin_array, 0, -0.001)
            tru_obs_bin_array = det_obs_bin_array

          n_dim = 4
          self.histutils.rebin_thn( roounfold_filename, thn, name_roounfold_thn , name_roounfold_obj, n_dim,
            len(det_pt_bin_array )-1, det_pt_bin_array ,
            len(det_obs_bin_array)-1, det_obs_bin_array,
            len(tru_pt_bin_array )-1, tru_pt_bin_array ,
            len(tru_obs_bin_array)-1, tru_obs_bin_array,
            label,0,1, grooming_setting!=None )

          f_resp = ROOT.TFile(roounfold_filename, 'READ')
          roounfold_response = f_resp.Get(name_roounfold_obj)
          roounfold_thn      = f_resp.Get(name_roounfold_thn )
          f_resp.Close() 

          setattr(self, name_roounfold_obj, roounfold_response)

          # Save the response matrix to the root file, in case we want to check something later
          self.outfile.cd()
          roounfold_thn.Write()

  #----------------------------------------------------------------------
  # Extract binning from a 1D histogram
  #----------------------------------------------------------------------
  def return_histo_binning_1D(self,h1):
    nBins = h1.GetNbinsX()
    binning = []
    for b in range(0,nBins):
      binning.append(h1.GetBinLowEdge(b+1))
      if b == nBins-1:
        binning.append(h1.GetBinLowEdge(b+1)+h1.GetBinWidth(b+1))
    return array('d',binning)

  #----------------------------------------------------------------------
  # Fold theoretical predictions
  #----------------------------------------------------------------------
  def fold_theory(self):

    # Loop over jet R
    for jetR in self.jetR_list:
     # Loop through subconfigurations to fold
     # (e.g. Standard_WTA, Standard_SD_1, ... in the jet-axis analysis, or
     # beta = 1.5, 2, 3, ... in the angularities analysis)
     for i, subconfig in enumerate(self.obs_subconfig_list):

       obs_setting = self.obs_settings[i]
       grooming_setting = self.grooming_settings[i]
       label = self.create_label( jetR , obs_setting , grooming_setting )

       # Retrieve theory histograms to be folded
       th_hists = []

       for sv in range(0,self.theory_scale_vars[jetR][i]):
         hist_name = 'h2_th_%s_R%s_obs_pT_%s_sv%i' % ( self.observable , (str)(jetR).replace('.','') , obs_setting , sv )
         if grooming_setting:
           hist_name += '_'
           hist_name += self.utils.grooming_label(grooming_setting)
         
         th_hist = getattr(self,hist_name)
         th_hists.append(th_hist) 

       # loop over response files (e.g. Pythia, Herwig, ...)
       for ri, response in enumerate(self.theory_response_files):
         name_RM = "hResponse_JetPt_" + self.observable + "_" + self.folding_type + "_" + label
         name_roounfold_obj = '%s_Roounfold_%i' % (name_RM, ri)
         response = getattr(self,name_roounfold_obj)

         for sv in range(0,self.theory_scale_vars[jetR][i]):
           h_folded_ch = response.ApplyToTruth(th_hists[sv])
           folded_hist_name = 'h2_folded_%s_R%s_obs_pT_%s_%i_sv%i' % ( self.observable , (str)(jetR).replace('.','') , obs_setting , ri, sv )
           h_folded_ch.SetNameTitle(folded_hist_name,folded_hist_name)

           setattr(self, folded_hist_name, h_folded_ch)

  #----------------------------------------------------------------------
  # Undoing some scalings that had been introduced before
  #----------------------------------------------------------------------
  def final_processing(self,do_MPI_corr=True):

    # Loop over jet R
    for jetR in self.jetR_list:
     # Loop through subconfigurations to fold
     # (e.g. Standard_WTA, Standard_SD_1, ... in the jet-axis analysis, or
     # beta = 1.5, 2, 3, ... in the angularities analysis)
     for i, subconfig in enumerate(self.obs_subconfig_list):

       obs_setting = self.obs_settings[i]
       grooming_setting = self.grooming_settings[i]
       label = self.create_label( jetR , obs_setting , grooming_setting ) 

       pt_bins = array('d', self.theory_pt_bins)

       # loop over response files (e.g. Pythia, Herwig, ...)
       for ri, response in enumerate(self.theory_response_files):
         
          # ----------------------------------------------------------------------------------------
         # Preparing MPI correction
         if do_MPI_corr:
           # Grab the two histograms that will be used for the MPI correction
           name_mpi_off = 'h_'+self.observable+'_JetPt_ch_'+label
           name_mpi_on = 'h_'+self.observable+'_JetPt_ch_MPIon_'+label

           h2_mpi_off = response.Get(name_mpi_off)
           h2_mpi_on = response.Get(name_mpi_on)

           # Gotta make sure the histograms we will use for the correction have the proper binning
           y_bins = array('d', self.theory_obs_bins)
           if grooming_setting:
             y_bins = np.insert(y_bins, 0, -0.001)
           h2_mpi_off = self.histutils.rebin_th2(h2_mpi_off, name_mpi_off+'_Rebinned_%i' % ri, pt_bins, len(pt_bins)-1, y_bins, len(y_bins)-1, grooming_setting!=None )
           h2_mpi_on  = self.histutils.rebin_th2(h2_mpi_on , name_mpi_on +'_Rebinned_%i' % ri, pt_bins, len(pt_bins)-1, y_bins, len(y_bins)-1, grooming_setting!=None )

           h2_mpi_ratio = h2_mpi_on.Clone()
           title = 'h_mpi_on_over_off_'+self.observable+'_JetPt_ch_'+label
           h2_mpi_ratio.SetNameTitle(title,title)
           h2_mpi_ratio.Divide(h2_mpi_off)
           h2_mpi_ratio.SetDirectory(0)

         # ----------------------------------------------------------------------------------------
         # Loop over scale variations
         for sv in range(0,self.theory_scale_vars[jetR][i]):
           
           folded_hist_name = 'h2_folded_%s_R%s_obs_pT_%s_%i_sv%i' % ( self.observable , (str)(jetR).replace('.','') , obs_setting , ri, sv )
           h2_folded_hist = getattr(self,folded_hist_name)

           if do_MPI_corr:
             h2_folded_hist.Multiply(h2_mpi_ratio) 

           self.outfile.cd()
           h2_folded_hist.Write()

           # If desired binning is different from what was used for the folding, need to take that into account before changing the pT normalization
           for n_pt in range(0,len(self.final_pt_bins)-1):
             projection_name = 'h1_folded_%s_R%s_%s_%i_sv%i_pT_%i_%i_Scaled' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,ri,sv,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1])) 

             min_bin, max_bin = self.bin_position( self.theory_pt_bins , self.final_pt_bins[n_pt],self.final_pt_bins[n_pt+1] )
            
             h1_folded_hist = h2_folded_hist.ProjectionY(projection_name,min_bin,max_bin)
             h1_folded_hist.SetTitle(projection_name)

             h1_folded_hist.SetDirectory(0)

             # Undo the bin width scaling and set correct normalization
             norm_factor = h1_folded_hist.Integral()
             if norm_factor == 0: norm_factor = 1
             h1_folded_hist.Scale(1./norm_factor, "width")

             for b in range(0,h1_folded_hist.GetNbinsX()):
               h1_folded_hist.SetBinError(b+1,0)

             self.outfile.cd()
             h1_folded_hist.Write()
             setattr(self,projection_name,h1_folded_hist) 

         # Do the loop backwards and find min and max histograms
         for n_pt in range(0,len(self.final_pt_bins)-1):
           histo_list = []
           for sv in range(0,self.theory_scale_vars[jetR][i]):
             projection_name = 'h1_folded_%s_R%s_%s_%i_sv%i_pT_%i_%i_Scaled' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,ri,sv,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))
             histo_list.append(getattr(self,projection_name))
           hist_min, hist_max = self.min_max( histo_list )

           # Create a graph out of these histograms
           name_central = 'h1_folded_%s_R%s_%s_%i_sv0_pT_%i_%i_Scaled' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,ri,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))
           h_central = getattr(self,name_central)
           graph = self.histo_to_graph(h_central,hist_min,hist_max)           
           name_graph = 'g_folded_%s_R%s_%s_%i_pT_%i_%i_Scaled' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,ri,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1]))
           graph.SetName(name_graph)

           graph_min = ROOT.TGraph(hist_min)
           graph_min.SetName('g_min_folded_%s_R%s_%s_%i_pT_%i_%i_Scaled' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,ri,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1])))

           graph_max = ROOT.TGraph(hist_max)
           graph_max.SetName('g_max_folded_%s_R%s_%s_%i_pT_%i_%i_Scaled' % ( self.observable,(str)(jetR).replace('.',''),obs_setting,ri,(int)(self.final_pt_bins[n_pt]),(int)(self.final_pt_bins[n_pt+1])))

           self.outfile.cd()
           hist_min.Write()
           hist_max.Write()
           graph.Write()
           graph_min.Write()
           graph_max.Write()

  #---------------------------------------------------------------
  # Given a pair of bin-edge values, return their index
  #---------------------------------------------------------------
  def bin_position( self , list_pt_th , min_p , max_p ):
    min_b = list_pt_th.index(min_p)+1
    max_b = list_pt_th.index(max_p)
    return min_b, max_b

  #---------------------------------------------------------------
  # Put together a label commonly used by several functions in code
  #---------------------------------------------------------------
  def create_label(self,jetR,obs_setting,grooming_setting):

    label = "R%s_" % (str(jetR).replace('.', ''))
    label += self.subobs_label(obs_setting)
    label += '_Scaled'

    if grooming_setting:
      label += '_'
      label += self.utils.grooming_label(grooming_setting)

    return label

  #---------------------------------------------------------------
  # Returns number proportional to shape of inclusive jet pT
  #     distribution theory prediction (val = jetR)
  #---------------------------------------------------------------
  def pt_scale_factor_jetR(self, ptmin, ptmax, jetR):

    pt_li, val_li = getattr(self, "pt_scale_factor_R%s" % jetR)

    # Fit a log function between the two endpoints and approx avg integral for bin
    start_i = pt_li.index(ptmin)
    end_i = pt_li.index(ptmax)
    # y = a * x^k
    k = np.log(val_li[start_i] / val_li[end_i]) / np.log(ptmin / ptmax)
    a = val_li[start_i] / ptmin**k

    return self.pt_scale_factor_k(ptmin, ptmax, k, a)

  #---------------------------------------------------------------
  # Returns number proportional to the integral of power law pTjet^k
  #---------------------------------------------------------------
  def pt_scale_factor_k(self, ptmin, ptmax, k, a=1e9):
    if k == -1:
      return a * np.log(ptmax / ptmin)
    return a * (ptmax**(k + 1) - ptmin**(k + 1)) / (k + 1)

  #----------------------------------------------------------------------
  # Extrapolate y-values for values in xlist_new given points (x,y) in xlist and ylist
  # Use power=1 for linear, or power=2 for quadratic extrapolation
  #----------------------------------------------------------------------
  def list_interpolate(self,xlist, ylist, xlist_new, power=1, require_positive=False):

    if len(xlist) < (power + 1):
      raise ValueError("list_interpolate() requires at least %i points!" % (power + 1))

    ylist_new = []
    ix = 0
    for xval in xlist_new:

      while (ix + power) < len(xlist) and xlist[ix+power] <= xval:
        ix += 1

      x1 = xlist[ix]; y1 = ylist[ix]

      # Check if data point is identical
      if xval == x1:
        if require_positive and y1 < 0:
          ylist_new.append(0)
          continue
        ylist_new.append(y1)
        continue

      # Set value to 0 if out-of-range for extrapolation
      if x1 > xval or (ix + power) >= len(xlist):
        ylist_new.append(0)
        continue

      x2 = xlist[ix+1]; y2 = ylist[ix+1]

      yval = None
      if power == 1:  # linear
        yval = self.linear_extrapolate(x1, y1, x2, y2, xval)
      else:
        raise ValueError("Unrecognized power", power, "/ please use either 1 or 2")

      # Require positive values
      if require_positive and yval < 0:
        ylist_new.append(0)
        continue

      ylist_new.append(yval)

    return ylist_new

  #---------------------------------------------------------------
  # Given two data points, find linear fit and y-value for x
  #---------------------------------------------------------------
  def linear_extrapolate(self,x1, y1, x2, y2, x):
    return (y2 - y1) / (x2 - x1) * x + (y1 - (y2 - y1) / (x2 - x1) * x1)

  #---------------------------------------------------------------
  # Set LHS of distributions to 0 if crosses to 0 at some point (prevents multiple peaks)
  #---------------------------------------------------------------
  def set_zero_range(self,yvals):

    found_nonzero_val = False

    # Step through list backwards
    for i in range(len(yvals)-1, -1, -1):
      if yvals[i] <= 0:
        if found_nonzero_val:
          for j in range(0, i+1):
            yvals[j] = 0
          break
        yvals[i] = 0
        continue
      else:
        found_nonzero_val = True
        continue

    return yvals

  #---------------------------------------------------------------
  # Where there are single values pos/neg between two neg/pos,
  # interpolate point
  #---------------------------------------------------------------
  def fix_fluctuations(self,yvals):
 
    for i in range(1, len(yvals) - 1):
      if yvals[i] > 0:
        if yvals[i+1] < 0 and yvals[i-1] < 0:
          yvals[i] = (yvals[i+1] + yvals[i-1]) / 2
      else:  # yvals[i] <= 0
        if yvals[i+1] > 0 and yvals[i-1] > 0:
          yvals[i] = (yvals[i+1] + yvals[i-1]) / 2

    return yvals

  #---------------------------------------------------------------
  # Given a list of histograms, return two histograms with the min
  # and max histograms
  #---------------------------------------------------------------
  def min_max( self , histo_list ):
    nHist = len(histo_list)
    nBins = histo_list[0].GetNbinsX()
    
    hist_min = histo_list[0].Clone()
    hist_min.SetName(hist_min.GetName()+'_min')

    hist_max = histo_list[0].Clone()
    hist_max.SetName(hist_max.GetName()+'_max')
    
    for b in range(0,nBins):
        bin_content = []
        for h in range(0,nHist):
            bin_content.append(histo_list[h].GetBinContent(b+1))
        min_cont = min(bin_content)
        max_cont = max(bin_content)
    
        hist_min.SetBinContent(b+1,min_cont)
        hist_max.SetBinContent(b+1,max_cont)
    
    return hist_min, hist_max

  #---------------------------------------------------------------
  # Given a central, min, and max histograms, return a graph
  #---------------------------------------------------------------
  def histo_to_graph(self,hc,hmin,hmax):
    tlx = []
    tly = []
    tl_min = []
    tl_max = []
    
    nBins = hc.GetNbinsX()
    listofzeros = [0] * nBins
 
    for b in range(0,nBins):
      tlx.append(hc.GetXaxis().GetBinCenter(b+1))
      cent = hc.GetBinContent(b+1)
      tly.append(cent)
      tl_min.append(cent-hmin.GetBinContent(b+1))
      tl_max.append(hmax.GetBinContent(b+1)-cent)
    
    lx = array('d',tlx)
    ly = array('d',tly)
    l_min = array('d',tl_min)
    l_max = array('d',tl_max)
    l_0 = array('d',listofzeros) 

    graph = ROOT.TGraphAsymmErrors(nBins,lx,ly,l_0,l_0,l_min,l_max)
    
    return graph 

  #----------------------------------------------------------------------
  def subobs_label( self , subobs ):
    label = ''
    if type(subobs)==str:
      label = subobs
    elif type(subobs)==int:
      label = '%i' % subobs
    elif type(subobs)==float:
      label = str(subobs).replace('.','')
    else:
      'This option has not been created yet. Bailing out!'
      exit()
    return label

  #----------------------------------------------------------------------
  # Return Grooming Parameters
  #----------------------------------------------------------------------
  def get_grooming_setting( self , subconf ):
    if 'SoftDrop' in subconf:
      zcut = subconf['SoftDrop']['zcut']
      beta = subconf['SoftDrop']['beta']
      return {'sd':[zcut,beta]}
    else:
      return None

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

