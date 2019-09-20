#! /usr/bin/env python

import sys
import os
import argparse
import itertools
from array import *
import ROOT
import numpy
import root_numpy
import pyunfold

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

# Suppress a lot of standard output
#ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Set plotting options
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

###########################################################################################
###########################################################################################
def pyunfold_inclusivejets(input_file_data, input_file_response, output_dir, file_format):

  fData = ROOT.TFile(input_file_data)
  fResponse = ROOT.TFile(input_file_response)

  # Set pT range of input spectrum for unfolding
  min_pt_det = 10
  max_pt_det = 100

  # Set pT range of output spectrum
  min_pt_reported = 20
  max_pt_reported = 100

  # Set pT range of response spectrum
  min_pt_gen = 10
  max_pt_gen = 300

  # Define pT-det and pT-truth binning
  bin_array_truth = ([min_pt_gen, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 190, 240, max_pt_gen])
  bin_array_det = ([min_pt_det, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, max_pt_det])
  
  n_bins_det = len(bin_array_det) - 1
  det_bin_array = array('d',bin_array_det)
  n_bins_truth = len(bin_array_truth) - 1
  truth_bin_array = array('d',bin_array_truth)
  print('n_bins_det: {}'.format(n_bins_det))
  print('n_bins_truth: {}'.format(n_bins_truth))

  #--------------------------------------------------------------
  
  # Create output dir for unfolding histograms and result
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

  # Get N events
  hname_event = 'hNevents'
  hNevent_data = fData.Get(hname_event)
  n_events_data = hNevent_data.GetBinContent(2)
  print('N accepted events in data: {}'.format(n_events_data))
  
  hNevent_response = fResponse.Get(hname_event)
  n_events_response = hNevent_response.GetBinContent(2)/20.
  print('N accepted events in response (avg per bin): {}'.format(n_events_response))

  #--------------------------------------------------------------
  # Get data jet spectrum
  
  hname_jetpt_data = 'hJetPt_R0.4'
  hJetSpectrumMeasuredPerBin = getMeasuredSpectrum(fData, hname_jetpt_data, min_pt_det, max_pt_det, n_bins_det, det_bin_array)
  hJetSpectrumMeasuredPerBin.Sumw2()
  visibleMBCrossSection = 50.87 # (mb) V0AND cross section (https://cds.cern.ch/record/2648933)
  vertexEfficiency = 0.95
  hJetSpectrumMeasuredPerBin.Scale(visibleMBCrossSection)
  hJetSpectrumMeasuredPerBin.Scale(vertexEfficiency)
  hJetSpectrumMeasuredPerBin.Scale(1./n_events_data)

  data = numpy.array(hJetSpectrumMeasuredPerBin)[1:-1] # exclude underflow/overflow bins
  data_err = 0.1*data

  #--------------------------------------------------------------
  # Get kinematic efficiency
  
  # Get truth-level spectrum (matched) from response matrix projection, before cutting the pT-det
  # range, do not rebin at this point since it will be cut to the range otherwise
  hname_response = 'hResponse_JetPt_R0.4Scaled'
  hResponseMatrixUncut = getResponseMatrix(fResponse, hname_response, 0, max_pt_gen, min_pt_gen, max_pt_gen, 0, 0, 0, 0, "uncut", output_dir)
  hJetSpectrumTrueUncutPerBin = hResponseMatrixUncut.ProjectionY()
  # rebin only the projcetion to keep an uncut range (for kinematic efficiency correction)
  hJetSpectrumTrueUncutPerBin = hJetSpectrumTrueUncutPerBin.Rebin(len(truth_bin_array)-1, "{}_NewBinning".format(hJetSpectrumTrueUncutPerBin.GetName()), truth_bin_array)
  hJetSpectrumTrueUncutPerBin.SetName("hJetSpectrumTrueUncutPerBin")
  
  # Get the truth-level jet spectrum (matched) from response matrix (already re-binned)
  hResponseMatrix = getResponseMatrix(fResponse, hname_response, min_pt_det, max_pt_det, min_pt_gen, max_pt_gen, n_bins_det, det_bin_array, n_bins_truth, truth_bin_array, "", output_dir)
  hJetSpectrumTruePerBin = hResponseMatrix.ProjectionY("_py",1,hResponseMatrix.GetNbinsX()) # Do exclude under and overflow bins
  hJetSpectrumTruePerBin.SetName("hJetSpectrumTruePerBin")
  
  hKinematicEfficiency = hJetSpectrumTrueUncutPerBin.Clone()
  hKinematicEfficiency.SetName("hKinematicEfficiency")
  hKinematicEfficiency.Divide(hJetSpectrumTruePerBin, hJetSpectrumTrueUncutPerBin, 1., 1., "B")
  outputFilename = os.path.join(output_dir, "hKinematicEfficiency" + file_format)
  plotHist(hKinematicEfficiency, outputFilename, "hist")
  
  efficiencies = numpy.array(hKinematicEfficiency)[1:-1]
  efficiencies_err = 0.01*efficiencies
  
  #--------------------------------------------------------------
  # Prior
  # Can use any numpy array that sums to 1
  cause_lim = numpy.logspace(0, 3, n_bins_truth)
  jeff_prior = pyunfold.priors.jeffreys_prior(cause_lim)
  
  for bin in range(1, n_bins_truth + 1):
    val = hJetSpectrumTrueUncutPerBin.GetBinContent(bin)
    bin_val = hJetSpectrumTrueUncutPerBin.GetBinCenter(bin)
    new_val = val * pow(bin_val, -0.5)
    hJetSpectrumTrueUncutPerBin.SetBinContent(bin, new_val)
  integral = hJetSpectrumTrueUncutPerBin.Integral()
  prior_truth = root_numpy.hist2array(hJetSpectrumTrueUncutPerBin) / integral
  
  #--------------------------------------------------------------
  # Get response matrix from response file (Measured, True) to be used for the unfolding,
  # with pT-det range cut to desired range, and re-bin.
  normalizeResponseMatrix(hResponseMatrix, min_pt_det, max_pt_det, min_pt_gen, max_pt_gen, output_dir, file_format)

  response = root_numpy.hist2array(hResponseMatrix)
  #response.shape = (-1, n_bins_det)
  response_err = 0*response

  # check response normalization:
  print('response column sum: {}'.format(response.sum(axis=0)))
  
  #--------------------------------------------------------------
  # Unfold spectrum
  # All histograms at this point are per-bin -- we will divide by bin width when plotting

  unfolded_result = pyunfold.iterative_unfold(data=data, data_err=data_err, response=response, response_err=response_err, efficiencies=efficiencies, efficiencies_err=efficiencies_err, callbacks=[pyunfold.callbacks.Logger()], prior=prior_truth)

  final_result = unfolded_result['unfolded']
  stat_err = unfolded_result['stat_err']
  sys_err = unfolded_result['sys_err']

  hFinalResult = ROOT.TH1F('hFinalResult', 'hFinalResult', n_bins_truth, truth_bin_array)
  root_numpy.array2hist(final_result, hFinalResult, stat_err)

  plot_unfolding_result(hJetSpectrumMeasuredPerBin, hJetSpectrumTrueUncutPerBin, hFinalResult, n_events_response, output_dir, file_format)
  
#--------------------------------------------------------------
def plot_unfolding_result(hData, hPrior, hUnfolded, n_events_response, output_dir, file_format):
  
  c = ROOT.TCanvas("c","c: hist",600,450)
  c.cd()
  c.cd().SetLeftMargin(0.15)
  c.SetLogy()
  
  hData.SetMarkerStyle(21)
  hData.SetMarkerColor(1)
  hData.Scale(1., 'width')
  
  hPrior.SetMarkerStyle(22)
  hPrior.SetMarkerColor(2)
  hPrior.Scale(1./n_events_response, 'width')
  
  hUnfolded.SetMarkerStyle(23)
  hUnfolded.SetMarkerColor(4)
  hUnfolded.Scale(1., 'width')
  
  hData.DrawCopy()
  hPrior.DrawCopy('same')
  hUnfolded.DrawCopy('same')
  
  leg = ROOT.TLegend(0.5,0.7,0.88,0.88,'')
  leg.SetFillColor(10)
  leg.SetBorderSize(0)
  leg.SetFillStyle(0)
  leg.SetTextSize(0.04)
  leg.AddEntry(hData, 'Data', 'P')
  leg.AddEntry(hPrior, 'Prior', 'P')
  leg.AddEntry(hUnfolded, 'Unfolded', 'P')
  leg.Draw("same")
  
  outputFilename = os.path.join(output_dir, "hFinalResult" + file_format)
  c.SaveAs(outputFilename)
  c.Close()

###################################################################################################
# Unfold jet spectrum from a single output list
###################################################################################################
def getMeasuredSpectrum(fData, hname_jetpt_data, min_pt_det, max_pt_det, n_bins_det, det_bin_array):

  jetHistogram = fData.Get(hname_jetpt_data)
  jetHistogram.GetXaxis().SetRangeUser(min_pt_det, max_pt_det)
  
  hJetSpectrumRebinned = jetHistogram.Rebin(n_bins_det, "{}New".format(hname_jetpt_data), det_bin_array)
  return hJetSpectrumRebinned

###################################################################################################
# Get response matrix from response file (Measured, True) and rebin it
###################################################################################################
def getResponseMatrix(fResponse, hname_response, min_pt_det, max_pt_det, min_pt_gen, max_pt_gen, n_bins_det, det_bin_array, n_bins_truth, truth_bin_array, label, output_dir):
  
  hResponseMatrixFineBinned = fResponse.Get(hname_response)
  hResponseMatrixFineBinned.GetYaxis().SetRangeUser(min_pt_gen, max_pt_gen)
  hResponseMatrixFineBinned.GetXaxis().SetRangeUser(min_pt_det, max_pt_det)
  
  # Create a new fine-binned histogram with the appropriate min,max cuts
  # Loop over all bins in fine-binned response matrix, and fill appropriate bin in new response matrix
  # Assume that the bin edges overlap appropriately
  histname = "{}_{}".format(hResponseMatrixFineBinned.GetName(), label)
  title = histname + ";#it{p}_{T,corr}^{det} (GeV/#it{c});#it{p}_{T}^{truth} (GeV/#it{c})"
  hResponseMatrixNew = ROOT.TH2D(histname, title, max_pt_det-min_pt_det, min_pt_det, max_pt_det, max_pt_gen-min_pt_gen, min_pt_gen, max_pt_gen)
  for ibin in range(1, hResponseMatrixFineBinned.GetNbinsX() + 1):
    for jbin in range(1, hResponseMatrixFineBinned.GetNbinsY() + 1):
      
      oldContent = hResponseMatrixFineBinned.GetBinContent(ibin, jbin)
      
      # Find the bin that should be filled in the new histogram, and fill it
      # Need to get (x,y) location from bins (ibin, jbin)
      x = hResponseMatrixNew.GetXaxis().GetBinCenter(ibin)
      y = hResponseMatrixNew.GetYaxis().GetBinCenter(jbin)
      if x > min_pt_det and x < max_pt_det and y > min_pt_gen and y < max_pt_gen:
        hResponseMatrixNew.Fill(x, y, oldContent)

      # Assume 0 errors on response matrix
      #for bin in range(1, hResponseMatrixNew.GetNcells() + 1):
      # hResponseMatrixNew.SetBinError(bin, 0)

  # Re-bin the response matrix, if a binning is provided
  if n_bins_det > 1 and n_bins_truth > 1:
    hResponseMatrixRebinned = rebinResponseMatrix(hResponseMatrixFineBinned, n_bins_det, det_bin_array, n_bins_truth, truth_bin_array)
    return hResponseMatrixRebinned
  else:
    return hResponseMatrixFineBinned

##################################################################################################
# Rebin the response matrix to have variable binning
##################################################################################################
def rebinResponseMatrix(hResponseMatrix, nBinsDet, detBinArray, nBinsTruth, truthBinArray):
  
  histname = "{}NewRebinned".format(hResponseMatrix.GetName())
  title = histname + ";#it{p}_{T,corr}^{det} (GeV/#it{c});#it{p}_{T}^{truth} (GeV/#it{c})"
  hResponseMatrixNew = ROOT.TH2D(histname, title, nBinsDet, detBinArray, nBinsTruth, truthBinArray)
  
  # Loop over all bins in fine-binned response matrix, and fill appropriate bin in new response matrix
  # Assume that the bin edges overlap appropriately
  for ibin in range(1, hResponseMatrix.GetNbinsX() + 1):
    for jbin in range(1, hResponseMatrix.GetNbinsY() + 1):
      
      oldContent = hResponseMatrix.GetBinContent(ibin, jbin)
      
      # Find the bin that should be filled in the new histogram, and fill it
      # Need to get (x,y) location from bins (ibin, jbin)
      x = hResponseMatrix.GetXaxis().GetBinCenter(ibin)
      y = hResponseMatrix.GetYaxis().GetBinCenter(jbin)
      hResponseMatrixNew.Fill(x, y, oldContent)

  # Assume 0 errors on response matrix
  for bin in range(1, hResponseMatrixNew.GetNcells() + 1):
    hResponseMatrixNew.SetBinError(bin, 0)
  
  return hResponseMatrixNew

################################################################################################
# Normalize response matrix
# Normalize the pT-truth projection to 1
################################################################################################
def normalizeResponseMatrix(hResponseMatrix, minPtDet, maxPtDet, minPtGen, maxPtGen, outputDir, fileFormat):
  
  # Plot response matrix before normalization
  outputFilename = os.path.join(outputDir, "hResponseMatrixBeforeNormalization" + fileFormat)
  #plotHist(hResponseMatrix, outputFilename, "colz")
  
  # Make projection onto pT-true axis (y-axis), and scale appropriately
  hTruthProjectionBefore = hResponseMatrix.ProjectionY("_py",1,hResponseMatrix.GetNbinsX()) # Do exclude under and overflow bins
  hTruthProjectionBefore.SetName("hTruthProjectionBefore")
  
  # Loop through truth-level bins, and apply normalization factor to all bins.
  nBinsY = hResponseMatrix.GetNbinsY() # pT-gen
  nBinsX = hResponseMatrix.GetNbinsX() # pT-det
  for truthBin in range(1,nBinsY+1):
    normalizationFactor = hTruthProjectionBefore.GetBinContent(truthBin)
    if normalizationFactor > 0:
      truthBinCenter = hTruthProjectionBefore.GetXaxis().GetBinCenter(truthBin)
      
      for detBin in range(1,nBinsX+1):
        binContent = hResponseMatrix.GetBinContent(detBin, truthBin)
        hResponseMatrix.SetBinContent(detBin, truthBin, binContent/normalizationFactor)

  # Plot response matrix
  outputFilename = os.path.join(outputDir, "{}_{}_{}{}".format(hResponseMatrix.GetName(), minPtDet, maxPtDet, fileFormat))
  c = ROOT.TCanvas("c","c: hist",600,450)
  c.cd()
  c.cd().SetLeftMargin(0.15)

  hResponseMatrix.Draw("colz")
  line = ROOT.TLine(minPtDet,0,minPtDet,250)
  line.SetLineColor(0)
  line.SetLineStyle(2)
  line.Draw("same")
  line2 = ROOT.TLine(maxPtDet,0,maxPtDet,250)
  line2.SetLineColor(0)
  line2.SetLineStyle(2)
  line2.Draw("same")
  line3 = ROOT.TLine(0,minPtGen,100,minPtGen)
  line3.SetLineColor(0)
  line3.SetLineStyle(2)
  line3.Draw("same")
  line4 = ROOT.TLine(0,maxPtGen,100,maxPtGen)
  line4.SetLineColor(0)
  line4.SetLineStyle(2)
  line4.Draw("same")
  
  c.SaveAs(outputFilename)
  c.Close()

###################################################################################################
# Plot basic histogram
###################################################################################################
def plotHist(h, outputFilename, drawOptions = "", setLogy = False, setLogz = False, text = ""):
  
  c = ROOT.TCanvas("c","c: hist",600,450)
  c.cd()
  c.cd().SetLeftMargin(0.15)
  
  if setLogy:
    c.SetLogy()
  if setLogz:
    c.SetLogz()
  h.DrawCopy(drawOptions)

  if text:
    textFit = ROOT.TLatex()
    textFit.SetTextSize(0.04)
    textFit.SetNDC()
    textFit.DrawLatex(0.6,0.8,text)

  c.SaveAs(outputFilename)
  c.Close()

#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description="Compare histograms to test the new EMCal corrections framework")
  parser.add_argument("-d", "--inputFileData", action="store",
                      type=str, metavar="inputFileData",
                      default="AnalysisResults.root",
                      help="Path of AnalysisResults.root file containing spectrum to be unfolded")
  parser.add_argument("-r", "--inputFileResponse", action="store",
                      type=str, metavar="inputFileResponse",
                      default="AnalysisResults.root",
                      help="Path of AnalysisResults.root file containing response matrix")
  parser.add_argument("-o", "--outputDir", action="store",
                      type=str, metavar="outputDir",
                      default="./unfolding_output/",
                      help="Output directory for QA plots to be written to")
  parser.add_argument("-i", "--imageFormat", action="store",
                      type=str, metavar="imageFormat",
                      default=".pdf",
                      help="Image format to save plots in, e.g. \".pdf\" or \".png\"")
                      
  # Parse the arguments
  args = parser.parse_args()
  
  print("Configuring...")
  print("inputFileData: \"{0}\"".format(args.inputFileData))
  print("inputFileResponse: \"{0}\"".format(args.inputFileResponse))
  print("ouputDir: \"{0}\"".format(args.outputDir))
  print("imageFormat: \"{0}\"".format(args.imageFormat))
  
  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFileData):
    print("File \"{0}\" does not exist! Exiting!".format(args.inputFileData))
    sys.exit(0)
  if not os.path.exists(args.inputFileResponse):
    print("File \"{0}\" does not exist! Exiting!".format(args.inputFileResponse))
    sys.exit(0)

  pyunfold_inclusivejets(input_file_data = args.inputFileData, input_file_response = args.inputFileResponse, output_dir = args.outputDir, file_format = args.imageFormat)
