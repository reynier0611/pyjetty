import ROOT
import argparse
import ctypes
import os
import sys
import yaml

from array import *

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

###################################################################################
# Main function
def scaleHistograms_fastHerwig():
  EndPtHardBin = 1

  for bin in range(1, EndPtHardBin+1):
    print('cleaning RM in bin ',bin,' out of ',EndPtHardBin)

    f = ROOT.TFile("{}/AnalysisResults.root".format(bin), "UPDATE")

    checked_histos = ''
    keys = f.GetListOfKeys()

    for key in keys:
      name = key.GetName()
      if "Scaled" in name and "hResponse" in name:
        if name in checked_histos:
          continue
        obj = f.Get(name)

        if obj.InheritsFrom(ROOT.THnBase.Class()): 
          print('removing outliers in: ',name)
          new_obj = removeOutliersRM(obj)
          
          checked_histos += name
          f.Delete(name+';1')
          new_obj.Write(name)

    f.Close()
########################################################################################################
# Function to rid RM from low-stats pT bins (to be used with Herwig fast sim)
def removeOutliersRM( RM , threshold = 50 ):
  pTbins = [20,40,60,80,100]
  size_pTbins = len(pTbins)
  integral = [0] * (size_pTbins-1)
 
  # Getting length of each axis of the RM
  n_bins_0 = RM.GetAxis(0).GetNbins()
  n_bins_1 = RM.GetAxis(1).GetNbins()
  n_bins_2 = RM.GetAxis(2).GetNbins()
  n_bins_3 = RM.GetAxis(3).GetNbins()

  for p in range(0,size_pTbins-1):
    for bin_1 in range(1,n_bins_1+1):
      x_1 = RM.GetAxis(1).GetBinCenter(bin_1)	# pT-truth bin center
      w_1 = RM.GetAxis(1).GetBinWidth (bin_1)	# pT-truth bin width  
      for bin_0 in range(1,n_bins_0+1): 
        for bin_2 in range(1,n_bins_2+1):
          for bin_3 in range(1,n_bins_3+1):
 
            x_list = (bin_0,bin_1,bin_2,bin_3)
            x = array('i', x_list)
            global_bin = RM.GetBin(x)
 
            content = RM.GetBinContent(global_bin);
 
            if content == 0:
              continue
 
            # Check the different pT-truth bins ([20,40],[40,60], ...) 
            if x_1 - w_1/2. >= pTbins[p] and x_1 + w_1/2. <= pTbins[p+1]:
              integral[p] += 1	# Count total number of entries each pT-truth bin
              continue
    #print('integral: ',integral[p])
    for bin_1 in range(1,n_bins_1+1):
      x_1 = RM.GetAxis(1).GetBinCenter(bin_1)   # pT-truth bin center
      w_1 = RM.GetAxis(1).GetBinWidth (bin_1)   # pT-truth bin width
      for bin_0 in range(1,n_bins_0+1):
        for bin_2 in range(1,n_bins_2+1):
          for bin_3 in range(1,n_bins_3+1):
 
            x_list = (bin_0,bin_1,bin_2,bin_3)
            x = array('i', x_list)
            global_bin = RM.GetBin(x)
 
            content = RM.GetBinContent(global_bin);
 
            if content == 0:
              continue
 
            # Check the different pT-truth bins ([20,40],[40,60], ...) 
            if x_1 - w_1/2. >= pTbins[p] and x_1 + w_1/2. <= pTbins[p+1]:
              if integral[p] < threshold: # If the number of entries is smaller than the defined threshold, erase that part of the RM
                RM.SetBinContent(global_bin,0);
                RM.SetBinError  (global_bin,0);
  return RM

#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  print("Executing scaleHistograms_fastHerwig.py...")
  print("")

  scaleHistograms_fastHerwig()
