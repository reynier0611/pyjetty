#! /usr/bin/env python
"""
run_fold_theory.py
Code to fold theory curves from a given to a desired 'level'
Adapted from Ezra Lesser's code
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

    print(self)
  
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
      self.obs_subconfig_list = [name for name in list(self.obs_config_dict.keys()) if 'config' in name ]
      self.obs_settings = self.utils.obs_settings(self.observable, self.obs_config_dict, self.obs_subconfig_list)
      self.grooming_settings = self.utils.grooming_settings(self.obs_config_dict)
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
      self.theory_obs_bins = config['theory_obs_bins'] # Binning desired for observable
      self.theory_pt_bins  = config['theory_pt_bins' ] # pT binning of theory calculations
      self.final_pt_bins   = config['final_pt_bins'  ] # pT binning wanted for the final curves

      # response matrices for the folding, and labels describing them
      self.theory_response_files = [ROOT.TFile(f, 'READ') for f in config['response_files']]
      self.theory_response_labels = config['response_labels']

      # scale factors needed to scale distributions
      self.theory_pt_scale_factors_filepath = os.path.join(self.theory_dir, config['pt_scale_factors_path'])

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
      outfilename = os.path.join( self.theory_dir , 'out_file.root' )
      self.outfile = ROOT.TFile(outfilename,'recreate')

      print('Loading pT scale factors...')
      self.load_pt_scale_factors(self.theory_pt_scale_factors_filepath)
      print('Loading theory curves...')
      self.load_theory_curves()
      print('Loading response matrix for folding theory predictions...')
      self.load_theory_response()
      print("Folding theory histograms...")
      self.fold_theory()
      print("Undoing some scalings...")
      self.undo_scalings(self.do_mpi_scaling)

      self.outfile.Close()

  #---------------------------------------------------------------
  # Load theory calculations
  #---------------------------------------------------------------
  def load_theory_curves(self):
    # The user needs to implement this function
    raise NotImplementedError('You must implement initialize_user_output_objects()!')

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
      # Loop through subconfigurations to unfold
      # (e.g. Standard_WTA, Standard_SD_1, ...)
      for i in range(0,len(self.obs_subconfig_list)):

        obs_setting = self.obs_settings[i]           # labels such as 'Standard_WTA'
        grooming_setting = self.grooming_settings[i] # grooming parameters

        label = "R%s_" % (str(jetR).replace('.', ''))
        label += self.subobs_label(obs_setting)
        label += '_Scaled'

        if grooming_setting:
          label += '_'
          label += self.utils.grooming_label(grooming_setting)

        # loop over response files (e.g. Pythia, Herwig, ...)
        for ri, response in enumerate(self.theory_response_files):

          # Load response matrix to take input from full hadron to charged hadron level
          name_RM = "hResponse_JetPt_" + self.observable + "_" + self.folding_type + "_" + label
          print('Loading response matrix:',name_RM)

          thn = response.Get(name_RM)
          setattr(self, '%s_%i' % (name_RM, ri), thn)

          # Create Roounfold object
          name_roounfold_obj = '%s_Roounfold_%i' % (name_RM, ri)
          name_roounfold_thn = '%s_Rebinned_%i'  % (name_RM, ri)

          '''
          Response axes:
          ['p_{T}^{ch jet}', 'p_{T}^{jet, hadron}', 'obs^{ch}', 'obs^{hadron}']
          as compared to the usual
          ['p_{T,det}', 'p_{T,truth}', '#lambda_{#beta,det}', '#lambda_{#beta,truth}']
          '''
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

          self.outfile.cd()
          roounfold_thn.Write()

  #----------------------------------------------------------------------
  # Fold theoretical predictions
  #----------------------------------------------------------------------
  def fold_theory(self):

    # Loop over jet R
    for jetR in self.jetR_list:

     # Loop through subconfigurations to unfold
     # (e.g. Standard_WTA, Standard_SD_1, ...)
     for i, subconfig in enumerate(self.obs_subconfig_list):

       obs_setting = self.obs_settings[i]
       grooming_setting = self.grooming_settings[i]

       label = "R%s_" % (str(jetR).replace('.', ''))
       label += self.subobs_label(obs_setting)
       label += '_Scaled'

       if grooming_setting:
         label += '_'
         label += self.utils.grooming_label(grooming_setting)

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
  def undo_scalings(self,do_MPI_corr=True):
    '''
    Before doing the folding in 'fold_theory', the theory predictions were
    scaled. This function takes care of reversing that scaling.
    '''
    # Loop over jet R
    for jetR in self.jetR_list:

     # Loop through subconfigurations to unfold
     for i, subconfig in enumerate(self.obs_subconfig_list):

       obs_setting = self.obs_settings[i]
       grooming_setting = self.grooming_settings[i]
 
       label = "R%s_" % (str(jetR).replace('.', ''))
       label += self.subobs_label(obs_setting)
       label += '_Scaled'
       if grooming_setting:
         label += '_'
         label += self.utils.grooming_label(grooming_setting)

       pt_bins = array('d', self.theory_pt_bins)

       # loop over response files (e.g. Pythia, Herwig, ...)
       for ri, response in enumerate(self.theory_response_files):
         
         # Grab the two histograms that will be used for the MPI correction
         name_mpi_off = 'h_'+self.observable+'_JetPt_ch_'+label
         name_mpi_on = 'h_'+self.observable+'_JetPt_ch_MPIon_'+label

         print('Histogram MPI off:',name_mpi_off)
         print('Histogram MPI on:' ,name_mpi_on )

         h2_mpi_off = response.Get(name_mpi_off)
         h2_mpi_on = response.Get(name_mpi_on)

         if do_MPI_corr:
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

  #---------------------------------------------------------------
  def bin_position( self , list_pt_th , min_p , max_p ):
    min_b = list_pt_th.index(min_p)+1
    max_b = list_pt_th.index(max_p)
    return min_b, max_b
 
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
  def plot_theory_response(self, jetR, obs_label, obs_setting, grooming_setting,
                           min_pt_truth, max_pt_truth, maxbin):

    label = "R%s_%s" % (str(jetR).replace('.', ''), str(obs_label).replace('.', ''))
    outf = ROOT.TFile(os.path.join(self.output_dir_theory, 'fTheoryResponseProj.root'), 'UPDATE')

    for ri in range(len(self.theory_response_files)):
      # Get histograms
      thn_ch = getattr(self, "hResponse_theory_ch_%s_%i" % (label, ri))
      thn_h = getattr(self, "hResponse_theory_h_%s_%i" % (label, ri))

      # Make projections in pT bins at (charged-)/hadron level
      thn_ch.GetAxis(0).SetRangeUser(int(min_pt_truth), int(max_pt_truth))
      thn_h.GetAxis(0).SetRangeUser(int(min_pt_truth), int(max_pt_truth))

      #print(thn_ch.GetBinContent(array('i', [3, 3, 20, 20])))

      hTheoryProjection_ch = thn_ch.Projection(2, 3)
      hTheoryProjection_h = thn_h.Projection(2, 3)

      #print(hTheoryProjection_ch.GetBinContent(20, 20))
      #exit()

      name_ch = "hResponse_theory_ch_%s_PtBin%i-%i_%s" % \
                (label, min_pt_truth, max_pt_truth, self.theory_response_labels[ri])
      name_h = "hResponse_theory_h_%s_PtBin%i-%i_%s" % \
               (label, min_pt_truth, max_pt_truth, self.theory_response_labels[ri])

      hTheoryProjection_ch.SetNameTitle(name_ch, name_ch)
      hTheoryProjection_h.SetNameTitle(name_h, name_h)

      # Save the histograms
      output_dir = self.output_dir_theory
      if not os.path.exists(output_dir):
        os.mkdir(output_dir)

      text_h = str(min_pt_truth) + ' < #it{p}_{T, h jet} < ' + str(max_pt_truth)
      text_ch = str(min_pt_truth) + ' < #it{p}_{T, ch jet} < ' + str(max_pt_truth)
      self.utils.plot_hist(hTheoryProjection_h, os.path.join(self.output_dir_theory, name_h+'.pdf'),
                           'colz', False, True, text_h)
      self.utils.plot_hist(hTheoryProjection_ch, os.path.join(self.output_dir_theory, name_ch+'.pdf'),
                           'colz', False, True, text_ch)

      hTheoryProjection_h.Write()
      hTheoryProjection_ch.Write()

      # Reset axes zoom in case histograms used later
      thn_ch.GetAxis(0).UnZoom()
      thn_h.GetAxis(0).UnZoom()

    outf.Close()

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

  # Where there are single values pos/neg between two neg/pos, interpolate point
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
  '''
  def plot_parton_comp(self, jetR, obs_label, obs_setting, grooming_setting):

    label = "R%s_%s" % (str(jetR).replace('.', ''), str(obs_label).replace('.', ''))

    # Directory to save the histograms
    output_dir = os.path.join(self.output_dir_theory, "parton_comp")
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    hcent_p = getattr(self, "theory_cent_%s_%s_parton" % (self.observable, label))
    hmin_p = getattr(self, "theory_min_%s_%s_parton" % (self.observable, label))
    hmax_p = getattr(self, "theory_max_%s_%s_parton" % (self.observable, label))

    n_obs_bins = len(self.theory_obs_bins) - 1
    obs_edges = self.theory_obs_bins

    pt_bins = array('d', self.theory_pt_bins)

    # Make projections in pT bins at parton level
    for i, min_pt in list(enumerate(self.theory_pt_bins))[:-1]:
      max_pt = self.theory_pt_bins[i+1]

      # Get the theory prediction
      hcent_p.GetXaxis().SetRangeUser(int(min_pt), int(max_pt))
      hmin_p.GetXaxis().SetRangeUser(int(min_pt), int(max_pt))
      hmax_p.GetXaxis().SetRangeUser(int(min_pt), int(max_pt))

      hcent_proj_p = hcent_p.ProjectionY()
      hmin_proj_p = hmin_p.ProjectionY()
      hmax_proj_p = hmax_p.ProjectionY()

      # Fix normalization
      hmin_proj_p.Scale(1/hcent_proj_p.Integral(), "width")
      hmax_proj_p.Scale(1/hcent_proj_p.Integral(), "width")
      hcent_proj_p.Scale(1/hcent_proj_p.Integral(), "width")

      # Initialize canvas & pad for plotting
      name = 'cTheoryComp_R{}_{}_{}-{}'.format(jetR, obs_label, min_pt, max_pt)
      c = ROOT.TCanvas(name, name, 600, 450)
      c.Draw()
      c.cd()

      name = 'pTheoryComp_R{}_{}_{}-{}'.format(jetR, obs_label, min_pt, max_pt)
      myPad = ROOT.TPad(name, name, 0, 0, 1, 1)
      myPad.SetLeftMargin(0.2)
      myPad.SetTopMargin(0.07)
      myPad.SetRightMargin(0.04)
      myPad.SetBottomMargin(0.13)
      myPad.Draw()
      myPad.cd()

      # Find the max bin: the last bin where the angularity is 0
      maxbin = hcent_proj_p.GetNbinsX()   # initialize
      while hmax_proj_p.GetBinContent(maxbin) < 1e-3 and maxbin > 1:
        maxbin -= 1

      # Use blank histogram to initialize this range
      bin_array = array('d', obs_edges[0:maxbin+1])
      name = 'hTheoryComp_R{}_{}_{}-{}_Blank'.format(jetR, obs_label, min_pt, max_pt)
      myBlankHisto = ROOT.TH1F(name, name, maxbin, bin_array)
      myBlankHisto.SetNdivisions(505)
      myBlankHisto.SetXTitle(self.xtitle)
      myBlankHisto.GetXaxis().SetTitleOffset(1.02)
      myBlankHisto.GetXaxis().SetTitleSize(0.055)
      myBlankHisto.SetYTitle(self.ytitle)
      myBlankHisto.GetYaxis().SetTitleOffset(1.1)
      myBlankHisto.GetYaxis().SetTitleSize(0.055)
      myBlankHisto.SetMinimum(0.)
      myBlankHisto.SetMaximum(1.7*hmax_proj_p.GetMaximum())
      myBlankHisto.Draw()

      x = array('d', [round(hcent_proj_p.GetXaxis().GetBinCenter(i), 5) for i in range(1, maxbin+1)])
      y = array('d', [hcent_proj_p.GetBinContent(i) for i in range(1, maxbin+1)])
      xerrup = array('d', [(x[i+1] - x[i]) / 2. for i in range(maxbin-1)] + [0])
      xerrdn = array('d', [0] + [(x[i+1] - x[i]) / 2. for i in range(maxbin-1)])
      yerrup = array('d', [hmax_proj_p.GetBinContent(i)-y[i-1] for i in range(1, maxbin+1)])
      yerrdn = array('d', [y[i-1]-hmin_proj_p.GetBinContent(i) for i in range(1, maxbin+1)])
      h_theory = ROOT.TGraphAsymmErrors(maxbin, x, y, xerrdn, xerrup, yerrdn, yerrup)
      color = self.ColorArray[4]
      h_theory.SetFillColorAlpha(color, 0.25)
      h_theory.SetLineColor(color)
      h_theory.SetLineWidth(3)
      h_theory.Draw('L 3 same')

      h_resp_list = []
      for ri in range(len(self.theory_response_files)):
        # Get event generator histogram
        thn = getattr(self, "hResponse_theory_ch_%s_%i" % (label, ri))

        # Get the response matrix prediction
        thn.GetAxis(1).SetRangeUser(int(min_pt), int(max_pt))
        h_response_projection = thn.Projection(3)
        name = "hTheoryComp_p_%s_PtBin%i-%i_%s" % \
                  (label, min_pt, max_pt, self.theory_response_labels[ri])
        h_response_projection.SetNameTitle(name, name)
        h_response_projection.SetDirectory(0)

        # Rescale by integral for correct normalization
        h_response_projection.Scale(1/h_response_projection.Integral(), "width")

        color = self.ColorArray[4+1+ri]
        h_response_projection.SetLineColor(color)
        h_response_projection.SetLineWidth(3)
        h_response_projection.Draw('L hist same')
        h_resp_list.append(h_response_projection)

        # Reset thn range in case used later
        thn.GetAxis(1).UnZoom()

      text_latex = ROOT.TLatex()
      text_latex.SetNDC()
      text_xval = 0.61
      text = 'ALICE {}'.format(self.figure_approval_status)
      text_latex.DrawLatex(text_xval, 0.87, text)

      text = 'pp #sqrt{#it{s}} = 5.02 TeV'
      text_latex.SetTextSize(0.045)
      text_latex.DrawLatex(text_xval, 0.8, text)

      text = "anti-#it{k}_{T} jets,   #it{R} = %s" % str(jetR)
      text_latex.DrawLatex(text_xval, 0.73, text)

      text = str(min_pt) + ' < #it{p}_{T,jet}^{parton} < ' + str(max_pt) + ' GeV/#it{c}'
      text_latex.DrawLatex(text_xval, 0.66, text)

      text = '| #it{#eta}_{jet}| < %s' % str(0.9 - jetR)
      subobs_label = self.utils.formatted_subobs_label(self.observable)
      if subobs_label:
        text += ',   %s = %s' % (subobs_label, obs_setting)
      delta = 0.07
      text_latex.DrawLatex(text_xval, 0.66-delta, text)

      myLegend = ROOT.TLegend(0.27, 0.7, 0.55, 0.9)
      self.utils.setup_legend(myLegend, 0.035)
      myLegend.AddEntry(h_theory, 'NLO+NLL', 'lf')
      for rl, l in enumerate(self.theory_response_labels):
        myLegend.AddEntry(h_resp_list[rl], l, 'lf')
      myLegend.Draw()

      name = 'hTheoryRatio_R{}_{}_{}-{}{}'.format(
        self.utils.remove_periods(jetR), obs_label,
        int(min_pt), int(max_pt), self.file_format)
      outputFilename = os.path.join(output_dir, name)
      c.SaveAs(outputFilename)
      c.Close()

    # Reset parton-level range in case used later
    hcent_p.GetXaxis().UnZoom()
    hmin_p.GetXaxis().UnZoom()
    hmax_p.GetXaxis().UnZoom()
  '''
  #----------------------------------------------------------------------
  # Return maximum & minimum y-values of unfolded results in a subconfig list
  def get_max_min(self, name, overlay_list, maxbins):

    total_min = 1e10
    total_max = -1e10

    for i, subconfig_name in enumerate(self.obs_subconfig_list):
    
      if subconfig_name not in overlay_list:
        continue

      obs_setting = self.obs_settings[i]
      grooming_setting = self.grooming_settings[i]
      obs_label = self.utils.obs_label(obs_setting, grooming_setting)
      maxbin = maxbins[i]
      
      h = getattr(self, name.format(obs_label))
      if 'SD' in obs_label:
        content = [ h.GetBinContent(j) for j in range(2, maxbin+2) ]
      else:
        content = [ h.GetBinContent(j) for j in range(1, maxbin+1) ]

      min_val = min(content)
      if min_val < total_min:
        total_min = min_val
      max_val = max(content)
      if max_val > total_max:
        total_max = max_val

        
    return (total_max, total_min)

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

