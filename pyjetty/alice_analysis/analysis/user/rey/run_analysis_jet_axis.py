#! /usr/bin/env python

import sys
import os
import argparse
from array import *
import numpy as np
import ROOT
import yaml

from pyjetty.alice_analysis.analysis.user.substructure import run_analysis
from pyjetty.alice_analysis.analysis.user.rey import plotting_utils

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class RunAnalysisJetAxis(run_analysis.RunAnalysis):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, config_file='', **kwargs):
    super(RunAnalysisJetAxis, self).__init__(config_file, **kwargs)
    
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
      
    self.figure_approval_status = config['figure_approval_status']
    self.plot_overlay_list = self.obs_config_dict['common_settings']['plot_overlay_list']
    
    self.jet_matching_distance = config['jet_matching_distance']
    
    if 'constituent_subtractor' in config:
      self.max_distance = config['constituent_subtractor']['max_distance']

    # Theory comparisons
    if 'fPythia' in config:
      self.fPythia_name = config['fPythia']

    self.fastsim_response_list = config['fastsim_response']

    user_wants_herwig_plot_in_final = True #Set to false to impose no Herwig in final plot, even if file exists
    if 'fastsim_generator0' in config['systematics_list'] and user_wants_herwig_plot_in_final:
      self.plot_herwig = True
    else:
      self.plot_herwig = False

    # Parameters for SCET comparison
    self.do_theory_comp = False
    if 'do_theory_comp' in config:
      self.do_theory_comp = config['do_theory_comp']

      if 'th_subconfigs' in config:
        self.th_subconfigs = config['th_subconfigs']
      else:
        self.th_subconfigs = self.obs_subconfig_list

      self.theory_dir = config['theory_dir']
      self.response_labels = config['response_labels']
 
  #---------------------------------------------------------------
  # This function is called once for each subconfiguration
  #---------------------------------------------------------------
  def plot_single_result(self, jetR, obs_label, obs_setting, grooming_setting):
    print('Plotting each individual result...')
  
    # Plot final result for each 1D substructure distribution (with PYTHIA)
    self.plot_final_result(jetR, obs_label, obs_setting, grooming_setting)         
  
  #---------------------------------------------------------------
  # This function is called once after all subconfigurations have been looped over, for each R
  #---------------------------------------------------------------
  def plot_all_results(self, jetR):
    print('Plotting overlay of all results...')
    
    for i_config, overlay_list in enumerate(self.plot_overlay_list):
    
      if len(overlay_list) > 1:
      
        self.plot_final_result_overlay(i_config, jetR, overlay_list)

  #----------------------------------------------------------------------
  # This function is called once after all subconfigurations and jetR have been looped over
  #----------------------------------------------------------------------
  def plot_performance(self):
    
    if not self.do_plot_performance:
      return
    print('Plotting performance plots...')
    
    # Initialize performance plotting class, and plot
    if self.is_pp:
    
      self.plotting_utils = plotting_utils.PlottingUtils(self.output_dir_performance, self.config_file)
      self.plot_single_performance(self.output_dir_performance)
      
    else:
      
      # Plot for each R_max
      for R_max in self.max_distance:
      
        output_dir_performance = os.path.join(self.output_dir_performance, 'Rmax{}'.format(R_max))
        self.plotting_utils = plotting_utils.PlottingUtils(output_dir_performance, self.config_file, R_max = R_max)
        self.plot_single_performance(output_dir_performance, R_max)

        # Plot for thermal model
        if self.do_thermal_closure and R_max == self.R_max:
          
          output_dir_performance = os.path.join(self.output_dir_performance, 'thermal')
          self.plotting_utils = plotting_utils.PlottingUtils(output_dir_performance, self.config_file, R_max = R_max, thermal = True)
          self.plot_single_performance(output_dir_performance, R_max)

  #----------------------------------------------------------------------
  # This function is called once after all subconfigurations and jetR have been looped over
  #----------------------------------------------------------------------
  def plot_single_performance(self, output_dir_performance, R_max = None):
  
    if R_max:
      suffix = '_Rmax{}'.format(R_max)
    else:
      suffix = ''
      
    # Create output subdirectories
    self.create_output_subdir(output_dir_performance, 'jet')
    self.create_output_subdir(output_dir_performance, 'resolution')
    self.create_output_subdir(output_dir_performance, 'residual_pt')
    self.create_output_subdir(output_dir_performance, 'residual_obs')
    self.create_output_subdir(output_dir_performance, 'mc_projections_det')
    self.create_output_subdir(output_dir_performance, 'mc_projections_truth')
    self.create_output_subdir(output_dir_performance, 'mc_projections_both')
    self.create_output_subdir(output_dir_performance, 'truth')
    self.create_output_subdir(output_dir_performance, 'data')
    self.create_output_subdir(output_dir_performance, 'lund')
    if not self.is_pp:
      self.create_output_subdir(output_dir_performance, 'delta_pt')
      self.create_output_subdir(output_dir_performance, 'prong_matching_fraction_pt')
      self.create_output_subdir(output_dir_performance, 'prong_matching_fraction_ptdet')
      self.create_output_subdir(output_dir_performance, 'prong_matching_deltaR')
      self.create_output_subdir(output_dir_performance, 'prong_matching_deltaZ')
      self.create_output_subdir(output_dir_performance, 'prong_matching_correlation')
    
    # Generate performance plots
    for jetR in self.jetR_list:
  
      # Plot some subobservable-independent performance plots
      self.plotting_utils.plot_DeltaR(jetR, self.jet_matching_distance)
      self.plotting_utils.plot_JES(jetR)
      self.plotting_utils.plot_JES_proj(jetR, self.pt_bins_reported)
      self.plotting_utils.plotJER(jetR, self.utils.obs_label(self.obs_settings[0], self.grooming_settings[0]))
      self.plotting_utils.plot_jet_reco_efficiency(jetR, self.utils.obs_label(self.obs_settings[0], self.grooming_settings[0]))
      
      if not self.is_pp:
        self.plotting_utils.plot_delta_pt(jetR, self.pt_bins_reported)
      
      # Plot subobservable-dependent performance plots
      for i, _ in enumerate(self.obs_subconfig_list):

        obs_setting = self.obs_settings[i]
        grooming_setting = self.grooming_settings[i]
        obs_label = self.utils.obs_label(obs_setting, grooming_setting)
    
        self.plotting_utils.plot_obs_resolution(jetR, obs_label, self.xtitle, self.pt_bins_reported)
        self.plotting_utils.plot_obs_residual_pt(jetR, obs_label, self.xtitle, self.pt_bins_reported, obs_label, grooming_setting)
        self.plotting_utils.plot_obs_residual_obs(jetR, obs_label, self.xtitle)
        self.plotting_utils.plot_obs_projections(jetR, obs_label, obs_setting, grooming_setting, self.xtitle, self.pt_bins_reported)
        self.plotting_utils.plot_obs_truth(jetR, obs_label, obs_setting, grooming_setting, self.xtitle, self.pt_bins_reported)
        
        if grooming_setting and self.observable != 'jet_axis':
          self.plotting_utils.plot_lund_plane(jetR, obs_label, grooming_setting)

      # Plot prong matching histograms
      if not self.is_pp:
        self.prong_match_threshold = 0.5
        min_pt = 80.
        max_pt = 100.
        prong_list = ['leading', 'subleading']
        match_list = ['leading', 'subleading', 'ungroomed', 'outside']
        for i, overlay_list in enumerate(self.plot_overlay_list):
          for prong in prong_list:
            for match in match_list:

              hname = 'hProngMatching_{}_{}_JetPt_R{}'.format(prong, match, jetR)
              self.plotting_utils.plot_prong_matching(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold)
              self.plotting_utils.plot_prong_matching_delta(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold, min_pt, max_pt, plot_deltaz=False)

              hname = 'hProngMatching_{}_{}_JetPtDet_R{}'.format(prong, match, jetR)
              self.plotting_utils.plot_prong_matching(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold)

              if 'subleading' in prong or 'leading' in prong:
                hname = 'hProngMatching_{}_{}_JetPtZ_R{}'.format(prong, match, jetR)
                self.plotting_utils.plot_prong_matching_delta(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold, min_pt, max_pt, plot_deltaz=True)

          hname = 'hProngMatching_subleading-leading_correlation_JetPtDet_R{}'.format(jetR)
          self.plotting_utils.plot_prong_matching_correlation(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold)

  #----------------------------------------------------------------------
  def plot_final_result(self, jetR, obs_label, obs_setting, grooming_setting):
    print('Plot final results for {}: R = {}, {}'.format(self.observable, jetR, obs_label)) 
    
    self.utils.set_plotting_options()
    ROOT.gROOT.ForceStyle()
    
    # Construct histogram of tagging fraction, to write to file
    if grooming_setting and 'sd' in grooming_setting:
      name = 'h_tagging_fraction_R{}_{}'.format(jetR, obs_label)
      h_tagging = ROOT.TH1D(name, name, len(self.pt_bins_reported) - 1, array('d', self.pt_bins_reported))

    # Loop through pt slices, and plot final result for each 1D observable distribution
    for bin in range(0, len(self.pt_bins_reported) - 1):
      min_pt_truth = self.pt_bins_reported[bin]
      max_pt_truth = self.pt_bins_reported[bin+1]
      maxbin = None
      maxbin = self.obs_max_bins(obs_label)[bin]

      # Plot each result independently with the pythia prediction
      self.plot_observable(jetR, obs_label, obs_setting, grooming_setting, min_pt_truth, max_pt_truth, plot_pythia=True)
    
      # --------------------------------------------------------------
      # Plot each result independently with the scet folded prediction 
      if self.do_theory_comp:
        for itm in self.th_subconfigs:
         if obs_setting == self.obs_config_dict[itm]['axis']: 
           self.plot_observable(jetR, obs_label, obs_setting, grooming_setting, min_pt_truth, max_pt_truth, plot_pythia=False, plot_scet=True)
           continue
      # --------------------------------------------------------------

      self.plot_RM_slices( jetR, obs_label, grooming_setting , min_pt_truth, max_pt_truth , maxbin )
 
      # Fill tagging fraction
      if grooming_setting and 'sd' in grooming_setting:
        fraction_tagged = getattr(self, 'tagging_fraction_R{}_{}_{}-{}'.format(jetR, obs_label, min_pt_truth, max_pt_truth))
        pt = (min_pt_truth + max_pt_truth)/2
        h_tagging.Fill(pt, fraction_tagged)
      
    # Write tagging fraction to ROOT file
    if grooming_setting and 'sd' in grooming_setting:
      output_dir = getattr(self, 'output_dir_final_results')
      final_result_root_filename = os.path.join(output_dir, 'fFinalResults.root')
      fFinalResults = ROOT.TFile(final_result_root_filename, 'UPDATE')
      h_tagging.Write()
      fFinalResults.Close()
      
  #----------------------------------------------------------------------
  def plot_observable(self, jetR, obs_label, obs_setting, grooming_setting, min_pt_truth, max_pt_truth, plot_pythia=False, plot_scet=False):
    
    name = 'cResult_R{}_{}_{}-{}'.format(jetR, obs_label, min_pt_truth, max_pt_truth)
    c = ROOT.TCanvas(name, name, 600, 450)
    c.Draw()
    
    c.cd()
    myPad = ROOT.TPad('myPad', 'The pad',0,0,1,1)
    myPad.SetLeftMargin(0.2)
    myPad.SetTopMargin(0.07)
    myPad.SetRightMargin(0.05)
    myPad.SetBottomMargin(0.13)
    myPad.Draw()
    myPad.cd()
    
    xtitle = getattr(self, 'xtitle')
    ytitle = getattr(self, 'ytitle')
    color = 1
    
    # Get histograms
    name = 'hmain_{}_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth)
    h = getattr(self, name)
    h.SetName(name)
    h.SetMarkerSize(1.3)
    h.SetMarkerStyle(20)
    h.SetMarkerColor(color)
    h.SetLineStyle(1)
    h.SetLineWidth(2)
    h.SetLineColor(color)
    
    name = 'hResult_{}_systotal_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth)
    h_sys = getattr(self, name)
    h_sys.SetName(name)
    h_sys.SetLineColor(0)
    h_sys.SetFillColor(color)
    h_sys.SetFillColorAlpha(color, 0.3)
    h_sys.SetFillStyle(1001)
    h_sys.SetLineWidth(0)
    
    n_obs_bins_truth = self.n_bins_truth(obs_label)
    truth_bin_array = self.truth_bin_array(obs_label)
    myBlankHisto = ROOT.TH1F('myBlankHisto','Blank Histogram', n_obs_bins_truth, truth_bin_array)
    myBlankHisto.SetNdivisions(108)
    myBlankHisto.SetXTitle(xtitle)
    myBlankHisto.GetYaxis().SetTitleOffset(1.5)
    myBlankHisto.SetYTitle(ytitle)
    myBlankHisto.SetMaximum(2.5*h.GetMaximum()) 
    myBlankHisto.SetMinimum(0.)
    myBlankHisto.Draw("E")

    # ------------------------------------------------------------------------------------------------
    # Overlay Pythia with the data
    if plot_pythia:
      hPythia, fraction_tagged_pythia = self.pythia_prediction(jetR, obs_setting, grooming_setting, obs_label, min_pt_truth, max_pt_truth)
      if hPythia:
        hPythia.SetFillStyle(0)
        hPythia.SetMarkerSize(1.3)
        hPythia.SetMarkerStyle(21)
        hPythia.SetMarkerColor(62)
        hPythia.SetLineColor(62)
        hPythia.SetLineWidth(2)
        hPythia.Draw('E2 same')
      else:
        print('No PYTHIA prediction for {} {}'.format(self.observable, obs_label))
        plot_pythia = False
    # ------------------------------------------------------------------------------------------------
    # Overlay the (already processed) SCET calculations
    clr_arr = [62,8,92]

    if plot_scet:
      g_scet_orig_c, g_scet_orig_min, g_scet_orig_max = self.scet_prediction(jetR, obs_setting, grooming_setting, obs_label,min_pt_truth, max_pt_truth)
      
      g_scet_orig_c.SetMarkerColor(2)
      g_scet_orig_c.SetLineColor(2)
      g_scet_orig_c.SetFillColorAlpha(2,0.2)
      g_scet_orig_c.Draw('sameLE3')

      # Folded theory curves
      lg_scet_folded_c = []
      lg_scet_folded_noMPI_c = []
      for g, gen in enumerate(self.response_labels):
        g_scet_folded_c, g_scet_folded_min, g_scet_folded_max = self.scet_folded_prediction(jetR, obs_setting, grooming_setting, obs_label,min_pt_truth, max_pt_truth, self.response_labels[g])
        g_scet_folded_noMPI_c, g_scet_folded_noMPI_min, g_scet_folded_noMPI_max = self.scet_folded_prediction_noMPIcorr(jetR, obs_setting, grooming_setting, obs_label,min_pt_truth, max_pt_truth, self.response_labels[g])

        g_scet_folded_c.SetMarkerColor(clr_arr[g])
        g_scet_folded_c.SetLineColor(clr_arr[g])
        g_scet_folded_c.SetFillColorAlpha(clr_arr[g],0.2)
        g_scet_folded_c.Draw('sameLE3')
        lg_scet_folded_c.append(g_scet_folded_c)

        g_scet_folded_noMPI_c.SetMarkerColor(clr_arr[g+1])
        g_scet_folded_noMPI_c.SetLineColor(clr_arr[g+1])
        g_scet_folded_noMPI_c.SetFillColorAlpha(clr_arr[g+1],0.2)
        g_scet_folded_noMPI_c.Draw('sameLE3')
        lg_scet_folded_noMPI_c.append(g_scet_folded_noMPI_c)

    # ------------------------------------------------------------------------------------------------
    
    h_sys.DrawCopy('E2 same')
    h.DrawCopy('PE X0 same')
  
    text_latex = ROOT.TLatex()
    text_latex.SetNDC()
    text = 'ALICE {}'.format(self.figure_approval_status)
    text_latex.DrawLatex(0.57, 0.87, text)
    
    text = 'pp #sqrt{#it{s}} = 5.02 TeV'
    text_latex.SetTextSize(0.045)
    text_latex.DrawLatex(0.57, 0.8, text)

    text = str(min_pt_truth) + ' < #it{p}_{T, ch jet} < ' + str(max_pt_truth) + ' GeV/#it{c}'
    text_latex.DrawLatex(0.57, 0.73, text)

    text = '#it{R} = ' + str(jetR) + '   | #eta_{jet}| < 0.5'
    text_latex.DrawLatex(0.57, 0.66, text)
    
    subobs_label = self.utils.formatted_subobs_label(self.observable)
    delta = 0.
    if subobs_label:
      text = '{} = {}'.format(subobs_label, obs_setting)
      text_latex.DrawLatex(0.57, 0.59, text)
      delta = 0.07
    
    if grooming_setting:
      text = self.utils.formatted_grooming_label(grooming_setting)
      text_latex.DrawLatex(0.57, 0.59-delta, text)
      
      if 'sd' in grooming_setting:
        fraction_tagged = getattr(self, 'tagging_fraction_R{}_{}_{}-{}'.format(jetR, obs_label, min_pt_truth, max_pt_truth))
        text_latex.SetTextSize(0.04)
        text = '#it{f}_{tagged}^{data} = %3.3f' % fraction_tagged
        text_latex.DrawLatex(0.57, 0.52-delta, text)
    
        if plot_pythia:
          text_latex.SetTextSize(0.04)
          text = ('#it{f}_{tagged}^{data} = %3.3f' % fraction_tagged) + (', #it{f}_{tagged}^{pythia} = %3.3f' % fraction_tagged_pythia)
          text_latex.DrawLatex(0.57, 0.52-delta, text)

    myLegend = ROOT.TLegend(0.22,0.60,0.40,0.9)
    self.utils.setup_legend(myLegend,0.035)
    myLegend.AddEntry(h, 'ALICE pp', 'pe')
    myLegend.AddEntry(h_sys, 'Sys. uncertainty', 'f')
    if plot_pythia:
      myLegend.AddEntry(hPythia, 'PYTHIA8 Monash2013', 'pe')
    if plot_scet:
      myLegend.AddEntry(g_scet_orig_c,'SCET full-hadron')
      for g, gen in enumerate(self.response_labels):
        myLegend.AddEntry(lg_scet_folded_noMPI_c[g],'SCET charged, no MPI')
        myLegend.AddEntry(lg_scet_folded_c[g],'SCET charged ('+gen+')')
    myLegend.Draw()

    name = 'hUnfolded_R{}_{}_{}-{}{}'.format(self.utils.remove_periods(jetR), obs_label, int(min_pt_truth), int(max_pt_truth), self.file_format)
    if plot_pythia:
      name = 'hUnfolded_R{}_{}_{}-{}_Pythia{}'.format(self.utils.remove_periods(jetR), obs_label, int(min_pt_truth), int(max_pt_truth), self.file_format)
    output_dir = getattr(self, 'output_dir_final_results')
    output_dir_single = output_dir + '/single_results'
    if not os.path.exists(output_dir_single):
      os.mkdir(output_dir_single)
    outputFilename = os.path.join(output_dir_single, name)
    c.SaveAs(outputFilename)
    c.Close()

    # Write result to ROOT file
    final_result_root_filename = os.path.join(output_dir, 'fFinalResults.root')
    fFinalResults = ROOT.TFile(final_result_root_filename, 'UPDATE')
    h.Write()
    h_sys.Write()
    if plot_pythia:
      hPythia.Write()
    fFinalResults.Close()

  #----------------------------------------------------------------------
  def scet_prediction(self, jetR, obs_setting, grooming_setting, obs_label,min_pt_truth, max_pt_truth):
   
    scet_file = 'folded_scet_calculations.root'
    scetFilename = os.path.join(self.theory_dir, scet_file)

    F_scet = ROOT.TFile(scetFilename)

    label = '_input_jet_axis_R%s_' % ((str)(jetR).replace('.',''))
    label += obs_label
    label += '_pT_%i_%i' % ( (int)(min_pt_truth) , (int)(max_pt_truth) )

    g_scet_c = F_scet.Get('g'+label)
    g_scet_min = F_scet.Get('g_min'+label)
    g_scet_max = F_scet.Get('g_max'+label)

    return g_scet_c, g_scet_min, g_scet_max

  #----------------------------------------------------------------------
  def scet_folded_prediction(self, jetR, obs_setting, grooming_setting, obs_label,min_pt_truth, max_pt_truth, model):
    scet_file = 'folded_scet_calculations.root'
    scetFilename = os.path.join(self.theory_dir, scet_file)

    F_scet = ROOT.TFile(scetFilename)

    label = '_folded_jet_axis_R%s_' % ((str)(jetR).replace('.',''))
    label += obs_label
    label += '_%s_pT_%i_%i' % ( model , (int)(min_pt_truth) , (int)(max_pt_truth) )

    g_scet_c = F_scet.Get('g'+label)
    g_scet_min = F_scet.Get('g_min'+label)
    g_scet_max = F_scet.Get('g_max'+label)

    return g_scet_c, g_scet_min, g_scet_max

  #----------------------------------------------------------------------
  def scet_folded_prediction_noMPIcorr(self, jetR, obs_setting, grooming_setting, obs_label,min_pt_truth, max_pt_truth, model):
    scet_file = 'folded_scet_calculations.root'
    scetFilename = os.path.join(self.theory_dir, scet_file)

    F_scet = ROOT.TFile(scetFilename)

    label = '_folded_noMPIcorr_jet_axis_R%s_' % ((str)(jetR).replace('.',''))
    label += obs_label
    label += '_%s_pT_%i_%i' % ( model , (int)(min_pt_truth) , (int)(max_pt_truth) )

    g_scet_c = F_scet.Get('g'+label)
    g_scet_min = F_scet.Get('g_min'+label)
    g_scet_max = F_scet.Get('g_max'+label)

    return g_scet_c, g_scet_min, g_scet_max

  #----------------------------------------------------------------------
  def pythia_prediction(self, jetR, obs_setting, grooming_setting, obs_label, min_pt_truth, max_pt_truth):
  
    plot_pythia_from_response = True
    plot_pythia_from_mateusz = False
    
    if plot_pythia_from_response:
    
      hPythia = self.get_pythia_from_response(jetR, obs_label, min_pt_truth, max_pt_truth)
      
      if grooming_setting and 'sd' in grooming_setting:
      
        # If SD, the untagged jets are in the first bin
        n_jets_inclusive = hPythia.Integral(1, hPythia.GetNbinsX()+1)
        n_jets_tagged = hPythia.Integral(hPythia.FindBin(self.truth_bin_array(obs_label)[0]), hPythia.GetNbinsX()+1)
        
      else:
        n_jets_inclusive = hPythia.Integral(1, hPythia.GetNbinsX()+1)
        n_jets_tagged = hPythia.Integral(hPythia.FindBin(self.truth_bin_array(obs_label)[0]), hPythia.GetNbinsX())
      
    fraction_tagged_pythia =  n_jets_tagged/n_jets_inclusive
    hPythia.Scale(1./n_jets_inclusive, 'width')
      
    return [hPythia, fraction_tagged_pythia]

  #----------------------------------------------------------------------
  def get_pythia_from_response(self, jetR, obs_label, min_pt_truth, max_pt_truth):
  
    output_dir = getattr(self, 'output_dir_main')
    file = os.path.join(output_dir, 'response.root')
    f = ROOT.TFile(file, 'READ')

    thn_name = 'hResponse_JetPt_{}_R{}_{}_rebinned'.format(self.observable, jetR, obs_label)
    thn = f.Get(thn_name)
    thn.GetAxis(1).SetRangeUser(min_pt_truth, max_pt_truth)

    h = thn.Projection(3)
    h.SetName('hPythia_{}_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth))
    h.SetDirectory(0)
    
    for i in range(1, h.GetNbinsX() + 1):
      h.SetBinError(i, 0)

    return h

  #----------------------------------------------------------------------
  def herwig_prediction(self, jetR, obs_setting, grooming_setting, obs_label, min_pt_truth, max_pt_truth):

    hHerwig = self.get_herwig_from_fast_response(jetR, obs_label, min_pt_truth, max_pt_truth)

    if grooming_setting and 'sd' in grooming_setting:

      # If SD, the untagged jets are in the first bin
      n_jets_inclusive = hHerwig.Integral(1, hHerwig.GetNbinsX()+1)
      n_jets_tagged = hHerwig.Integral(hHerwig.FindBin(self.truth_bin_array(obs_label)[0]), hHerwig.GetNbinsX()+1)

    else:
      n_jets_inclusive = hHerwig.Integral(1, hHerwig.GetNbinsX()+1)
      n_jets_tagged = hHerwig.Integral(hHerwig.FindBin(self.truth_bin_array(obs_label)[0]), hHerwig.GetNbinsX())

    fraction_tagged_herwig =  n_jets_tagged/n_jets_inclusive
    hHerwig.Scale(1./n_jets_inclusive, 'width')

    return [hHerwig, fraction_tagged_herwig]

  #----------------------------------------------------------------------
  def get_herwig_from_fast_response(self, jetR, obs_label, min_pt_truth, max_pt_truth): 
    # Assuming HERWIG is the first fast-simulation file declared in the config
    output_dir = getattr(self, 'output_dir_fastsim_generator0') 
    file = os.path.join(output_dir, 'response.root')
    f = ROOT.TFile(file, 'READ')

    thn_name = 'hResponse_JetPt_{}_R{}_{}_rebinned'.format(self.observable, jetR, obs_label)
    thn = f.Get(thn_name)
    thn.GetAxis(1).SetRangeUser(min_pt_truth, max_pt_truth)

    h = thn.Projection(3)
    h.SetName('hPythia_{}_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth))
    h.SetDirectory(0)

    for i in range(1, h.GetNbinsX() + 1):
      h.SetBinError(i, 0)

    return h

  #----------------------------------------------------------------------
  def plot_final_result_overlay(self, i_config, jetR, overlay_list):
    print('Plotting overlay of {}'.format(overlay_list))

    # Plot overlay of different subconfigs, for fixed pt bin
    for bin in range(0, len(self.pt_bins_reported) - 1):
      min_pt_truth = self.pt_bins_reported[bin]
      max_pt_truth = self.pt_bins_reported[bin+1]
      maxbins = [self.obs_max_bins(obs_label)[bin] for obs_label in self.obs_labels]

      # Plot PYTHIA
      self.plot_observable_overlay_subconfigs(i_config, jetR, overlay_list, min_pt_truth, max_pt_truth, maxbins, plot_pythia=True, plot_ratio = True)

  #----------------------------------------------------------------------
  def plot_observable_overlay_subconfigs(self, i_config, jetR, overlay_list, min_pt_truth, max_pt_truth, maxbins, plot_pythia=False, plot_nll=False, plot_ratio=False):
    
    name = 'cResult_overlay_R{}_allpt_{}-{}'.format(jetR, min_pt_truth, max_pt_truth)

    if plot_ratio and self.plot_herwig:
      c = ROOT.TCanvas(name, name, 600, 800)
    elif plot_ratio:
      c = ROOT.TCanvas(name, name, 600, 650)
    else:
      c = ROOT.TCanvas(name, name, 600, 450)
    c.Draw()
    
    c.cd()

    if plot_ratio and self.plot_herwig:
      pad1 = ROOT.TPad('myPad','The pad',0,0.4,1,1)
    elif plot_ratio:
      pad1 = ROOT.TPad('myPad','The pad',0,0.3,1,1)
    else:
      pad1 = ROOT.TPad('myPad','The pad',0,0  ,1,1)

    pad1.SetLeftMargin(0.2)
    pad1.SetTopMargin(0.06)
    pad1.SetRightMargin(0.05)
    pad1.SetBottomMargin(0.13)
    if plot_ratio:
      pad1.SetBottomMargin(0.)
    pad1.SetTicks(0,1)
    pad1.Draw()
    pad1.cd()

    myLegend = ROOT.TLegend(0.25,0.55,0.61,0.93)
    self.utils.setup_legend(myLegend,0.04)
    
    name = 'hmain_{}_R{}_{{}}_{}-{}'.format(self.observable, jetR, min_pt_truth, max_pt_truth)
    ymax = self.get_maximum(name, overlay_list)
      
    for i, subconfig_name in enumerate(self.obs_subconfig_list):
    
      if subconfig_name not in overlay_list:
        continue

      obs_setting = self.obs_settings[i]
      grooming_setting = self.grooming_settings[i]
      obs_label = self.utils.obs_label(obs_setting, grooming_setting)
      maxbin = maxbins[i]
      
      if subconfig_name == overlay_list[0]:
        marker = 20
        marker_pythia = marker+4
        color = 1
      elif subconfig_name == overlay_list[1]:
        marker = 21
        marker_pythia = marker+4
        color = 62
      elif subconfig_name == overlay_list[2]:
        marker = 33
        marker_pythia = marker+4
        color = 2
      elif subconfig_name == overlay_list[3]:
        marker = 34
        marker_pythia = 32
        color = 8
      elif subconfig_name == overlay_list[4]:
        marker = 24
        marker_pythia = 27
        color = 92
      else:
        marker = 25
        marker_pythia = 30
        color = 50

      name = 'hmain_{}_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth)
      h = getattr(self, name)
      h.SetMarkerSize(1.3)
      h.SetMarkerStyle(marker)
      h.SetMarkerColor(color)
      h.SetLineStyle(1)
      h.SetLineWidth(2)
      h.SetLineColor(color)

      name_sys = 'hResult_{}_systotal_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth)      
      h_sys = getattr(self, name_sys)
      h_sys.SetLineColor(0)
      h_sys.SetFillColor(color)
      h_sys.SetFillColorAlpha(color, 0.3)
      h_sys.SetFillStyle(1001)
      h_sys.SetLineWidth(0)
      h_sys.SetMarkerSize(1.3)
      h_sys.SetMarkerStyle(marker)
      h_sys.SetMarkerColor(color)

      if grooming_setting and maxbin:
        h = self.truncate_hist(getattr(self, name), maxbin+1, name+'_trunc')
      else:
        h = self.truncate_hist(getattr(self, name), maxbin, name+'_trunc')
      
      if subconfig_name == overlay_list[0]:

        pad1.cd()
        pad1.SetLogy()
        xtitle = getattr(self, 'xtitle')
        ytitle = getattr(self, 'ytitle')
        xmin = self.obs_config_dict[subconfig_name]['obs_bins_truth'][0]
        xmax = self.obs_config_dict[subconfig_name]['obs_bins_truth'][-1]
        if maxbin: 
            xmax = self.obs_config_dict[subconfig_name]['obs_bins_truth'][maxbin]

        myBlankHisto = ROOT.TH1F('myBlankHisto','Blank Histogram', 1, xmin, xmax)
        myBlankHisto.SetNdivisions(108)
        myBlankHisto.GetXaxis().SetTitleSize(0.085)
        myBlankHisto.SetXTitle(xtitle)
        myBlankHisto.GetYaxis().SetTitleOffset(1.5)
        myBlankHisto.SetYTitle(ytitle)
        myBlankHisto.SetMaximum(100*ymax)
        myBlankHisto.SetMinimum(0.)
        if plot_ratio:
          myBlankHisto.SetMinimum(3e-2) # Don't draw 0 on top panel 
          myBlankHisto.GetYaxis().SetTitleSize(0.075)
          myBlankHisto.GetYaxis().SetTitleOffset(1.2)
          myBlankHisto.GetYaxis().SetLabelSize(0.06)
        myBlankHisto.Draw('E')
        
        # Plot ratio
        if plot_ratio:
          l1, b10, b20 = self.line_box10_box20(xmin,xmax)

          c.cd()
          if self.plot_herwig:
            pad2 = ROOT.TPad("pad2", "pad2", 0, 0.25, 1, 0.4)
            pad3 = ROOT.TPad("pad3", "pad3", 0, 0.01, 1, 0.25)
            pad2.SetTopMargin(0)
            pad3.SetTopMargin(0)
            pad2.SetBottomMargin(0)
            pad3.SetBottomMargin(0.4)
            pad2.SetLeftMargin(0.2)
            pad3.SetLeftMargin(0.2)
            pad2.SetRightMargin(0.05)
            pad3.SetRightMargin(0.05)
            pad2.SetTicks(0,1)
            pad3.SetTicks(0,1)
            pad2.Draw()
            pad3.Draw()

            pad2.cd()
            myBlankHisto2 = myBlankHisto.Clone("myBlankHisto_C")
            self.pretty_blank_histo(myBlankHisto2,xtitle,"#frac{Data}{PYTHIA}",0.61,1.45)
            myBlankHisto2.Draw()

            l1.Draw("same")
            #b10.Draw("same")
            #b20.Draw("same")

            pad3.cd()
            myBlankHisto3 = myBlankHisto.Clone("myBlankHisto_C")
            self.pretty_blank_histo(myBlankHisto3,xtitle,"#frac{Data}{HERWIG}",0.61,1.45)
            myBlankHisto3.Draw()

            l1.Draw("same")
            #b10.Draw("same")
            #b20.Draw("same")

          else:
            pad2 = ROOT.TPad("pad2", "pad2", 0, 0.02, 1, 0.3)
            pad2.SetTopMargin(0)
            pad2.SetBottomMargin(0.4)
            pad2.SetLeftMargin(0.2)
            pad2.SetRightMargin(0.05)
            pad2.SetTicks(0,1)
            pad2.Draw()
            pad2.cd()
            
            pad2.cd()
            myBlankHisto2 = myBlankHisto.Clone("myBlankHisto_C")
            self.pretty_blank_histo(myBlankHisto2,xtitle,"#frac{Data}{PYTHIA}")
            myBlankHisto2.Draw()
          
            l1.Draw("same")
            #b10.Draw("same")
            #b20.Draw("same")

      # ------------------------------- overlay PYTHIA with final results -------------------------------
      if plot_pythia: 
        hPythia, fraction_tagged_pythia = self.pythia_prediction(jetR, obs_setting, grooming_setting, obs_label, min_pt_truth, max_pt_truth)

        if grooming_setting and maxbin:
          hPythia = self.truncate_hist(hPythia, maxbin+1,'final_pythia_trunc')
        else:
          hPythia = self.truncate_hist(hPythia, maxbin, 'final_pythia_trunc')

        plot_errors = False
        if plot_errors:
          hPythia.SetMarkerSize(0)
          hPythia.SetMarkerStyle(0)
          hPythia.SetMarkerColor(color)
          hPythia.SetFillColor(color)
        else:
          hPythia.SetLineColor(color)
          hPythia.SetLineColorAlpha(color, 0.5)
          hPythia.SetLineWidth(4)
      # ------------------------------- overlay HERWIG with final results -------------------------------
      if self.plot_herwig:
        hHerwig, fraction_tagged_herwig = self.herwig_prediction(jetR, obs_setting, grooming_setting, obs_label, min_pt_truth, max_pt_truth)
   
        if grooming_setting and maxbin:
          hHerwig = self.truncate_hist(hHerwig, maxbin+1, 'final_herwig_trunc')
        else:
          hHerwig = self.truncate_hist(hHerwig, maxbin, 'final_herwig_trunc')

        hHerwig.SetLineColor(color)
        hHerwig.SetLineColorAlpha(color, 0.5)
        hHerwig.SetLineWidth(4)
        hHerwig.SetLineStyle(2)
      
      # -------------------------------------------------------------------------------------------------
      # Create graphs that will be plotted on the bottom pads (ratios)
      if plot_ratio: 
        hRatioSys = h_sys.Clone()
        hRatioSys.SetName('{}_Ratio'.format(h_sys.GetName()))
        if plot_pythia:
          hRatioSys.Divide(hPythia)
          hRatioSys.SetLineColor(0)
          hRatioSys.SetFillColor(color)
          hRatioSys.SetFillColorAlpha(color, 0.3)
          hRatioSys.SetFillStyle(1001)
          hRatioSys.SetLineWidth(0)
          
        hRatioStat = h.Clone()
        hRatioStat.SetName('{}_Ratio'.format(h.GetName()))
        if plot_pythia:
          hRatioStat.Divide(hPythia)

        if self.plot_herwig:
          hRatioSys2 = h_sys.Clone()
          hRatioSys2.SetName('{}_Ratio_herwig'.format(h_sys.GetName()))
          hRatioSys2.Divide(hHerwig)
          hRatioSys2.SetLineColor(0)
          hRatioSys2.SetFillColor(color)
          hRatioSys2.SetFillColorAlpha(color, 0.3)
          hRatioSys2.SetFillStyle(1001)
          hRatioSys2.SetLineWidth(0)
          hRatioStat2 = h.Clone()
          hRatioStat2.SetName('{}_Ratio_herwig'.format(h.GetName()))
          hRatioStat2.Divide(hHerwig)

      # -------------------------------------------------------------------------------------------------
      # Actually start plotting the graphs previously defined
      pad1.cd()
      
      if plot_pythia:
        plot_errors = False
        if plot_errors:
          hPythia.DrawCopy('E3 same')
        else:
          hPythia.DrawCopy('L hist same')
      
      if self.plot_herwig:
        hHerwig.DrawCopy('L hist same')

      h_sys.DrawCopy('E2 same')
      h.DrawCopy('PE X0 same')

      # -------------------------------
      # Go to the lower panels and plot ratios      
      if plot_ratio:
        pad2.cd()
        if plot_pythia:
          hRatioSys.DrawCopy('E2 same')
       
        hRatioStat.DrawCopy('PE X0 same')

        if self.plot_herwig:
          pad3.cd()
          hRatioSys2.DrawCopy('E2 same')
          hRatioStat2.DrawCopy('PE X0 same')

      # -------------------------------
      # Go back to the upper panel and add text, legends, ...
      subobs_label = self.utils.formatted_subobs_label(self.observable)
      text = ''
      if subobs_label == '#Delta #it{R}_{axis}':
        if obs_setting == 'Standard_WTA':
          text += 'Standard - WTA'
        elif 'Standard_SD' in obs_setting:
          text += 'Standard - '
        elif 'WTA_SD' in obs_setting:
          text += 'WTA - '
      elif subobs_label:
        text += '{} = {}'.format(subobs_label, obs_setting)
        
      if grooming_setting:
        text += self.utils.formatted_grooming_label(grooming_setting, verbose=True).replace("Soft Drop","SD")

        if 'sd' in grooming_setting:
          fraction_tagged = getattr(self, 'tagging_fraction_R{}_{}_{}-{}'.format(jetR, obs_label, min_pt_truth, max_pt_truth))
          text += ' (#it{f}_{t}^{data} = %3.3f)' % fraction_tagged 

      myLegend.AddEntry(h, '{}'.format(text), 'pe')
        
    pad1.cd()
    myLegend.AddEntry(h_sys, 'Sys. uncertainty', 'f')
    if plot_pythia:
      myLegend.AddEntry(hPythia, 'PYTHIA8 Monash 2013', 'l') 
    if self.plot_herwig:
      myLegend.AddEntry(hHerwig, 'HERWIG7', 'l')
    
    text_latex = ROOT.TLatex()
    text_latex.SetNDC()
    
    text_latex.SetTextSize(0.05)
    
    text = 'ALICE {}'.format(self.figure_approval_status)
    text_latex.DrawLatex(0.5,0.5, text)

    x = 0.23
    y = 0.23
    
    text_latex.SetTextSize(0.05)
    text = 'pp #sqrt{#it{s}} = 5.02 TeV'
    text_latex.DrawLatex(x, y-0.00, text)

    text = 'Charged-particle jets   anti-#it{k}_{T}'
    text_latex.DrawLatex(x, y-0.18, text)
    
    text = '#it{R} = ' + str(jetR) + '   | #it{{#eta}}_{{jet}}| < {}'.format(0.9-jetR)
    text_latex.DrawLatex(x, y-0.06, text)
    
    text = str(min_pt_truth) + ' < #it{p}_{T, ch jet} < ' + str(max_pt_truth) + ' GeV/#it{c}'
    text_latex.DrawLatex(x, y-0.12, text)
    
    myLegend.Draw()
    
    name = 'h_{}_R{}_{}-{}_{}{}'.format(self.observable, self.utils.remove_periods(jetR), int(min_pt_truth), int(max_pt_truth), i_config, self.file_format)
    if plot_pythia:
      name = 'h_{}_R{}_{}-{}_Pythia_{}{}'.format(self.observable, self.utils.remove_periods(jetR), int(min_pt_truth), int(max_pt_truth), i_config, self.file_format)

    output_dir = getattr(self, 'output_dir_final_results') + '/all_results'
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    outputFilename = os.path.join(output_dir, name)
    c.SaveAs(outputFilename)
    c.Close()

  #----------------------------------------------------------------------
  # Make blank histogram look nice
  def pretty_blank_histo(self,histo,xtit,ytit,ymin=0.61,ymax=1.69):
    histo.SetYTitle(ytit)
    histo.SetXTitle(xtit)
    histo.GetXaxis().SetTitleSize(30)
    histo.GetXaxis().SetTitleFont(43)
    histo.GetXaxis().SetTitleOffset(4.)
    histo.GetXaxis().SetLabelFont(43)
    histo.GetXaxis().SetLabelSize(25)
    histo.GetYaxis().SetTitleSize(25)
    histo.GetYaxis().SetTitleFont(43)
    histo.GetYaxis().SetTitleOffset(2.2)
    histo.GetYaxis().SetLabelFont(43)
    histo.GetYaxis().SetLabelSize(25)
    histo.GetYaxis().SetNdivisions(107)
    histo.GetYaxis().SetRangeUser(ymin,ymax)

  #----------------------------------------------------------------------
  # return a line at y = 1, and two boxes representing 10 and 20% levels
  def line_box10_box20(self,xmin,xmax):
    line = ROOT.TLine(xmin,1,xmax,1)
    line.SetLineColor(920+2)
    line.SetLineStyle(2)

    box20 = ROOT.TBox(xmin,0.8,xmax,1.2)
    box20.SetFillColorAlpha(13,0.12)

    box10 = ROOT.TBox(xmin,0.9,xmax,1.1)
    box10.SetFillColorAlpha(13,0.18)

    return line, box10, box20

  #----------------------------------------------------------------------
  # Return maximum y-value of unfolded results in a subconfig list
  def get_maximum(self, name, overlay_list):
  
    max = 0.
    for i, subconfig_name in enumerate(self.obs_subconfig_list):
    
      if subconfig_name not in overlay_list:
        continue

      obs_setting = self.obs_settings[i]
      grooming_setting = self.grooming_settings[i]
      obs_label = self.utils.obs_label(obs_setting, grooming_setting)
      
      h = getattr(self, name.format(obs_label))
      if h.GetMaximum() > max:
        max = h.GetMaximum()
        
    return max

  #################################################################################################
  # Plot various slices of the response matrix (from the THn)
  #################################################################################################
  def plot_RM_slices( self , jetR, obs_label, grooming_setting , min_pt_truth, max_pt_truth , maxbin ):

    self.utils.set_plotting_options()
    ROOT.gROOT.ForceStyle()

    # (pt-det, pt-true, obs-det, obs-true)
    output_dir = getattr(self, 'output_dir_main')
    file = os.path.join(output_dir, 'response.root')
    f = ROOT.TFile(file, 'READ')

    thn_name = 'hResponse_JetPt_{}_R{}_{}_rebinned'.format(self.observable, jetR, obs_label)
    thn = f.Get(thn_name)
    thn.GetAxis(1).SetRangeUser(min_pt_truth, max_pt_truth)

    h2 = thn.Projection(3,2)
    h2.SetName('hPythia_proj_obs_{}_R{}_{}_{}-{}'.format(self.observable, jetR, obs_label, min_pt_truth, max_pt_truth))
    h2.SetDirectory(0) 

    # -------------------------
    h2 = self.utils.normalize_response_matrix(h2)
    # Set z-maximum in Soft Drop case, since otherwise the untagged bin will dominate the scale
    if grooming_setting and 'sd' in grooming_setting:
      h2.SetMaximum(0.3)
    # -------------------------

    c = ROOT.TCanvas("c","c: hist",1800,1400)
    c.cd()
    ROOT.gPad.SetLeftMargin(0.17)
    ROOT.gPad.SetBottomMargin(0.13)
    ROOT.gPad.SetRightMargin(0.17)
    ROOT.gPad.SetTopMargin(0.26)
    c.SetLogz()

    h2.GetYaxis().SetTitle('#it{#DeltaR}_{axis}^{truth}')
    h2.GetYaxis().SetTitleOffset(1.7)
    h2.GetYaxis().SetNdivisions(107)
    h2.GetYaxis().SetTitleSize(0.05)
    h2.GetYaxis().SetLabelSize(0.05)

    h2.GetXaxis().SetTitle('#it{#DeltaR}_{axis}^{det}')
    h2.GetXaxis().SetNdivisions(107)
    h2.GetXaxis().SetTitleSize(0.05)
    h2.GetXaxis().SetLabelSize(0.05)

    h2.GetZaxis().SetTitle('Probability density')
    h2.GetZaxis().SetTitleOffset(1.4)

    if maxbin: 
      h2.GetXaxis().SetRange(1,maxbin)
      h2.GetYaxis().SetRange(1,maxbin)

    h2.Draw('colz')

    # -------------------------
    subobs_label = self.utils.formatted_subobs_label(self.observable)
    text = '' 
 
    if obs_label == 'Standard_WTA':
      text += 'Standard - WTA'
    elif 'Standard_SD' in obs_label:
      text += 'Standard - '
    elif 'WTA_SD' in obs_label:
      text += 'WTA - ' 

    if grooming_setting:
      text += self.utils.formatted_grooming_label(grooming_setting, verbose=True).replace("Soft Drop","SD")
    # -------------------------

    text_latex = ROOT.TLatex()
    text_latex.SetNDC()
    text_latex.SetTextSize(0.05)
    text_latex.DrawLatex(0.06,0.95,'ALICE {}'.format(self.figure_approval_status))
    text_latex.DrawLatex(0.06,0.89,'PYTHIA8 Monash2013')
    text_latex.DrawLatex(0.06,0.83,'pp #sqrt{#it{s}} = 5.02 TeV')
    text_latex.DrawLatex(0.06,0.77,text)
    text_latex.DrawLatex(0.45,0.95,'Charged-particle jets   anti-#it{k}_{T}')
    text_latex.DrawLatex(0.45,0.89,str(min_pt_truth) + ' < #it{p}_{T, ch jet} < ' + str(max_pt_truth) + ' GeV/#it{c}')
    text_latex.DrawLatex(0.45,0.82,'#it{R} = ' + str(jetR) + '   | #it{{#eta}}_{{jet}}| < {}'.format(0.9-jetR))

    filename = os.path.join(output_dir, 'RM/RM_'+obs_label+'_{}_{}.pdf'.format(min_pt_truth,max_pt_truth))
    c.SaveAs(filename)
    c.Close()

#----------------------------------------------------------------------
if __name__ == '__main__':

  # Define arguments
  parser = argparse.ArgumentParser(description='Jet substructure analysis')
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

  analysis = RunAnalysisJetAxis(config_file = args.configFile)
  analysis.run_analysis()
