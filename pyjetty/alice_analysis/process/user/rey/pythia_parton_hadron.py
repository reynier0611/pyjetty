#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT

import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext

from pyjetty.alice_analysis.process.base import process_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

################################################################
class pythia_parton_hadron(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(pythia_parton_hadron, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Defaults to None if not in use
        self.level = args.no_match_level

        self.jetR_list = config["jetR"] 

        self.user_seed = args.user_seed
        self.nev = args.nev

        # hadron level - ALICE tracking restriction
        self.max_eta_hadron = 0.9

        # Whether or not to rescale final jet histograms based on sigma/N
        self.no_scale = args.no_scale

        self.observable_list = config['process_observables']
        self.observable = self.observable_list[0] 
       
        self.obs_settings = {}
        self.obs_grooming_settings = {}
        for observable in self.observable_list:

          obs_config_dict = config[observable]
          obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]

          obs_subconfig_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
          self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_subconfig_list)
          self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_parton_hadron(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        # Initialize response histograms
        self.initialize_hist()

        pinfo('user seed for pythia', self.user_seed)
        mycfg = ['Random:setSeed=on', 'Random:seed={}'.format(self.user_seed)]
        mycfg.append('HadronLevel:all=off')

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        # -------------------------------
        # PYTHIA instance with MPI off and ISR on
        setattr(args, "py_noMPI", True)
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        self.init_jet_tools()
        self.calculate_events(pythia)
        pythia.stat()
        print()
        
        # -------------------------------
        # PYTHIA instance with MPI on and ISR on
        setattr(args, "py_noMPI", False)
        pythia_MPI = pyconf.create_and_init_pythia_from_args(args, mycfg)
        self.calculate_events(pythia_MPI, MPIon=True)
        print()
 
        self.scale_print_final_info(pythia, pythia_MPI)

        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)
        self.hNeventsMPI = ROOT.TH1I("hNeventsMPI", 'Number accepted events (unscaled)', 2, -0.5, 1.5)
        self.hNeventsNoUE = ROOT.TH1I("hNeventsNoUE", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            # list for MPIon (ISRon) histograms
            hist_list_name_MPIon = "hist_list_MPIon_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name_MPIon, [])

            R_label = str(jetR).replace('.', '') + 'Scaled'

            if self.level in [None, 'ch']:
              name = 'hJetPt_ch_R%s' % R_label
              h = ROOT.TH1F(name, name+';p_{T}^{ch jet} [GeV/#it{c}];#frac{dN}{dp_{T}^{ch jet}};', 300, 0, 300) 
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

              name = 'hNconstit_Pt_ch_R%s' % R_label
              h = ROOT.TH2F(name, name+';#it{p}_{T}^{ch jet} [GeV/#it{c}];#it{N}_{constit}^{ch jet}', 300, 0, 300, 50, 0.5, 50.5)
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

              name = 'hNconstit_Pt_ch_MPIon_R%s' % R_label
              h = ROOT.TH2F(name, name+';#it{p}_{T}^{ch jet} [GeV/#it{c}];#it{N}_{constit}^{ch jet}', 300, 0, 300, 50, 0.5, 50.5)
              setattr(self, name, h)
              getattr(self, hist_list_name_MPIon).append(h)

            if self.level in [None, 'h']:
              name = 'hJetPt_h_R%s' % R_label
              h = ROOT.TH1F(name, name+';p_{T}^{jet, h} [GeV/#it{c}];#frac{dN}{dp_{T}^{jet, h}};', 300, 0, 300)
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

              name = 'hNconstit_Pt_h_R%s' % R_label
              h = ROOT.TH2F(name, name+';#it{p}_{T}^{h jet} [GeV/#it{c}];#it{N}_{constit}^{h jet}', 300, 0, 300, 50, 0.5, 50.5)
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

            if self.level in [None, 'p']:
              name = 'hJetPt_p_R%s' % R_label
              h = ROOT.TH1F(name, name+';p_{T}^{jet, parton} [GeV/#it{c}];#frac{dN}{dp_{T}^{jet, parton}};',300, 0, 300)
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

              name = 'hNconstit_Pt_p_R%s' % R_label
              h = ROOT.TH2F(name, name+';#it{p}_{T}^{p jet} [GeV/#it{c}];#it{N}_{constit}^{p jet}', 300, 0, 300, 50, 0.5, 50.5)
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

            if self.level == None:
              name = 'hJetPtRes_R%s' % R_label
              h = ROOT.TH2F(name, name +';#it{p}_{T}^{parton jet} [GeV/#it{c}];#frac{#it{p}_{T}^{parton jet}-#it{p}_{T}^{ch jet}}{#it{p}_{T}^{parton jet}}', 300, 0, 300, 200, -1., 1.)
              setattr(self, name, h)
              getattr(self, hist_list_name).append(h)

            # ----------------------------------------------------------------
            # Loop over subobservable
            for i, axes in enumerate(self.obs_settings[self.observable]):
              common_name_1 = 'h_' + self.observable + '_'
              common_name_2 = '_R' + str(jetR).replace('.','') +'_' + axes + '_Scaled'
              grooming_setting = self.obs_grooming_settings[self.observable][i] 
              grooming_label = ''
              if grooming_setting:
                grooming_label = self.utils.grooming_label(grooming_setting)
                common_name_2 += '_' + grooming_label

              max_obs = 1. # Maximum value for the observable
              obs_label = self.observable
              if self.observable == 'jet_axis':
                obs_label = '#DeltaR'
                max_obs = jetR 
                if 'Standard_SD' in axes:
                  max_obs = jetR / 2.

              # ----- charged hadron-level histograms --------------------------
              if self.level in [None, 'ch']:
                name = common_name_1 + 'JetPt_ch' + common_name_2
                h = ROOT.TH2F(name, name + ';p_{T}^{ch jet} [GeV/#it{c}];'+obs_label+'^{ch}', 195, 5, 200, 160, 0, max_obs)
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)
             
                # histograms for MPI on, ISR on 
                name = common_name_1 + 'JetPt_ch_MPIon' + common_name_2
                h = ROOT.TH2F(name, name+';p_{T}^{ch jet} [GeV/#it{c}];'+obs_label+'^{ch}', 195, 5, 200,160, 0, max_obs)
                setattr(self, name, h)
                getattr(self, hist_list_name_MPIon).append(h)

              # ----------------------------------------------------------------
              if self.level == None:
                # Create THn of response
                dim = 4
                title = ['p_{T}^{ch jet}', 'p_{T}^{parton jet}',obs_label+'^{ch}', obs_label+'^{parton}']
                nbins = [28, 28, 80, 80]
                min_li = [ 10.,  10., 0.     , 0.     ]
                max_li = [150., 150., max_obs, max_obs]
                if grooming_setting:
                  min_li[2] = -max_obs/10.
                  min_li[3] = -max_obs/10.

                nbins = (nbins)
                xmin = (min_li)
                xmax = (max_li)
                nbins_array = array.array('i', nbins)
                xmin_array = array.array('d', xmin)
                xmax_array = array.array('d', xmax)

                # THn for full to charged hadron-level folding
                title = ['p_{T}^{ch jet}', 'p_{T}^{h jet}',obs_label+'^{ch}', obs_label+'^{h}']
                name = 'hResponse_JetPt_' + self.observable + '_h2ch' + common_name_2
                h = ROOT.THnF(name, name, dim, nbins_array, xmin_array, xmax_array)
                for i in range(0, dim):
                    h.GetAxis(i).SetTitle(title[i])
                h.Sumw2()
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')      
            
            # set up our jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            setattr(self, "jet_def_R%s" % jetR_str, jet_def)
            print(jet_def)

        pwarning('max eta for particles after hadronization set to', self.max_eta_hadron)
        parts_selector_h = fj.SelectorAbsEtaMax(self.max_eta_hadron)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            
            jet_selector = fj.SelectorPtMin(5.0) & \
                           fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

            count1 = 0  # Number of jets rejected from ch-h matching
            setattr(self, "count1_R%s" % jetR_str, count1)
            count2 = 0  # Number of jets rejected from h-p matching
            setattr(self, "count2_R%s" % jetR_str, count2)


    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def calculate_events(self, pythia, MPIon=False, ISRon=True):
        
        iev = 0  # Event loop count

        if MPIon and ISRon:                    # MPI = on and ISR = on
            hNevents = self.hNeventsMPI
        elif ISRon:                            # MPI = on and ISR = off
            hNevents = self.hNevents
        else:                                  # MPI = off and ISR = off
            hNevents = self.hNeventsNoUE

        # Loop over all events requested
        while hNevents.GetBinContent(1) < self.nev:
            if not pythia.next():
                continue

            # parton level
            parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)
            
            hstatus = pythia.forceHadronLevel()
            if not hstatus:
                #pwarning('forceHadronLevel false event', iev)
                continue
             
            # full-hadron level
            parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

            # charged-hadron level
            parts_pythia_hch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Some "accepted" events don't survive hadronization step -- keep track here
            hNevents.Fill(0)
            self.find_jets_fill_trees(parts_pythia_p, parts_pythia_h, parts_pythia_hch, iev, MPIon, ISRon)

            iev += 1

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def find_jets_fill_trees(self, parts_pythia_p, parts_pythia_h, parts_pythia_hch, iev, MPIon=False, ISRon=True):

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)
            count1 = getattr(self, "count1_R%s" % jetR_str)
            count2 = getattr(self, "count2_R%s" % jetR_str)
 
            jets_p = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_p)))    # parton level
            jets_h = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_h)))    # full hadron level
            jets_ch = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_hch))) # charged hadron level

            # instance with multi-parton interactions turned on 
            if MPIon and ISRon:
                for jet in jets_ch:
                    self.fill_MPI_histograms(jetR, jet)
                continue

            for i,jchh in enumerate(jets_ch):

                # match hadron jets and charged jets
                drhh_list = []
                for j, jh in enumerate(jets_h):
                    drhh = jchh.delta_R(jh)
                    if drhh < jetR / 2.:
                        drhh_list.append((j,jh))
                
                if len(drhh_list) != 1:
                    count1 += 1
                else:  # Require unique match
                    j, jh = drhh_list[0]

                    # match parton level jet
                    dr_list = []
                    for k, jp in enumerate(jets_p):
                        dr = jh.delta_R(jp)
                        if dr < jetR / 2.:
                            dr_list.append((k, jp))
                    if len(dr_list) != 1:
                        count2 += 1
                    else:
                        k, jp = dr_list[0]
                        
                        self.fill_jet_histograms(jetR, jp, jh, jchh)
                        
            setattr(self, "count1_R%s" % jetR_str, count1)
            setattr(self, "count2_R%s" % jetR_str, count2)

    #---------------------------------------------------------------
    # Compute angle between jet axes
    #---------------------------------------------------------------
    def angle_between_jet_axes(self, jetR, jet, obs_setting, grooming_setting):
      jet_def_wta = fj.JetDefinition(fj.cambridge_algorithm, 2*jetR)
      jet_def_wta.set_recombination_scheme(fj.WTA_pt_scheme)
      reclusterer_wta =  fjcontrib.Recluster(jet_def_wta)
      jet_wta = reclusterer_wta.result(jet)

      if 'Standard_SD' in obs_setting: 
        gshop = fjcontrib.GroomerShop(jet, jetR, self.reclustering_algorithm)
        jet_sd_lund = self.utils.groom(gshop, grooming_setting, jetR)
        jet_sd = jet_sd_lund.pair()
        deltaR = jet.delta_R(jet_sd) 
        if jet_sd_lund.Delta() < 0: # untagged jet (i.e. failed SD)
          deltaR = -1
      elif obs_setting == 'Standard_WTA':
        deltaR = jet.delta_R(jet_wta)
      elif 'WTA_SD' in obs_setting:
        gshop = fjcontrib.GroomerShop(jet, jetR, self.reclustering_algorithm)
        jet_sd_lund = self.utils.groom(gshop, grooming_setting, jetR)
        jet_sd = jet_sd_lund.pair()
        deltaR = jet_wta.delta_R(jet_sd)
        if jet_sd_lund.Delta() < 0: # untagged jet (i.e. failed SD)
          deltaR = -1

      return deltaR

    #---------------------------------------------------------------
    # Fill jet histograms for MPI-on PYTHIA run-through
    #---------------------------------------------------------------
    def fill_MPI_histograms(self, jetR, jet):

        R_label = str(jetR).replace('.', '') + 'Scaled'

        name = 'hNconstit_Pt_ch_MPIon_R%s' % R_label
        h = getattr(self, name)
        h.Fill(jet.pt(),len(jet.constituents()))

        if(self.observable=='jet_axis'):
          for i, axes in enumerate(self.obs_settings[self.observable]):
            grooming_setting = self.obs_grooming_settings[self.observable][i]
            deltaR = self.angle_between_jet_axes(jetR, jet, axes, grooming_setting)

            common_name_1 = 'h_' + self.observable + '_'
            common_name_2 = '_R' + str(jetR).replace('.','') +'_' + axes + '_Scaled'
            
            grooming_label = ''
            if grooming_setting:
              grooming_label = self.utils.grooming_label(grooming_setting)
              common_name_2 += '_' + grooming_label            
            name = common_name_1 + 'JetPt_ch_MPIon' + common_name_2

            h = getattr(self, name)
            h.Fill(jet.pt(),deltaR)

    #---------------------------------------------------------------
    # Fill jet histograms
    #---------------------------------------------------------------
    def fill_jet_histograms(self, jetR, jp, jh, jch):

        R_label = str(jetR).replace('.', '') + 'Scaled'

        # Fill jet histograms which are not dependent on observable
        if self.level in [None, 'ch']:
            getattr(self, 'hJetPt_ch_R%s' % R_label).Fill(jch.pt())
            getattr(self, 'hNconstit_Pt_ch_R%s' % R_label).Fill(jch.pt(), len(jch.constituents()))
        if self.level in [None, 'h']:
            getattr(self, 'hJetPt_h_R%s' % R_label).Fill(jh.pt())
            getattr(self, 'hNconstit_Pt_h_R%s' % R_label).Fill(jh.pt(), len(jh.constituents()))
        if self.level in [None, 'p']:
            getattr(self, 'hJetPt_p_R%s' % R_label).Fill(jp.pt())
            getattr(self, 'hNconstit_Pt_p_R%s' % R_label).Fill(jp.pt(), len(jp.constituents()))

        if self.level == None:
            if jp.pt():  # prevent divide by 0
                getattr(self, 'hJetPtRes_R%s' % R_label).Fill(jp.pt(), (jp.pt() - jch.pt()) / jp.pt())

        # Fill observable histograms and response matrices
        self.fill_RMs(jetR, jp, jh, jch) 

    #---------------------------------------------------------------
    # Fill jet histograms
    #---------------------------------------------------------------
    def fill_RMs(self, jetR, jp, jh, jch):
        
        if(self.observable=='jet_axis'):
          for i, axes in enumerate(self.obs_settings[self.observable]):
            grooming_setting = self.obs_grooming_settings[self.observable][i]

            deltaR_ch = self.angle_between_jet_axes(jetR, jch, axes, grooming_setting)
            deltaR_h  = self.angle_between_jet_axes(jetR, jh , axes, grooming_setting)
            #deltaR_p  = self.angle_between_jet_axes(jetR, jp , axes, grooming_setting) 
            
            common_name_1 = 'h_' + self.observable + '_'
            common_name_2 = '_R' + str(jetR).replace('.','') +'_' + axes + '_Scaled' 
            
            if grooming_setting:
              grooming_label = self.utils.grooming_label(grooming_setting)
              common_name_2 += '_' + grooming_label

            if self.level in [None, 'ch']:
              getattr(self, common_name_1 + 'JetPt_ch' + common_name_2 ).Fill(jch.pt(), deltaR_ch )

            if self.level == None:
              # 4D response matrices for "forward folding" from full to charged hadron-level
              x = ([jch.pt(), jh.pt(), deltaR_ch, deltaR_h])
              x_array = array.array('d', x)
              name = 'hResponse_JetPt_' + self.observable + '_h2ch' + common_name_2
              getattr(self, name ).Fill(x_array)
           
    #---------------------------------------------------------------
    # Initiate scaling of all histograms and print final simulation info
    #---------------------------------------------------------------
    def scale_print_final_info(self, pythia, pythia_MPI):

        # Scale all jet histograms by the appropriate factor from generated cross section
        # and the number of accepted events
        if not self.no_scale:
            scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
            print("Weight MPIoff tree by (cross section)/(N events) =", scale_f)
            MPI_scale_f = pythia_MPI.info.sigmaGen() / self.hNeventsMPI.GetBinContent(1)
            print("Weight MPIon tree by (cross section)/(N events) =", MPI_scale_f)
            self.scale_jet_histograms(scale_f, MPI_scale_f)
        print()

        print("N total final MPI-off events:", int(self.hNevents.GetBinContent(1)), "with",
              int(pythia.info.nAccepted() - self.hNevents.GetBinContent(1)),
              "events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            count1 = getattr(self, "count1_R%s" % jetR_str)
            count2 = getattr(self, "count2_R%s" % jetR_str)
            print(("For R=%s:  %i jets cut at first match criteria; " + \
                  "%i jets cut at second match criteria.") % 
                  (str(jetR), count1, count2))
        print()


    #---------------------------------------------------------------
    # Scale all jet histograms by sigma/N
    #---------------------------------------------------------------
    def scale_jet_histograms(self, scale_f, MPI_scale_f):

        for jetR in self.jetR_list:
            # scale histograms with MPIoff and ISRon
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '') 
            for h in getattr(self, hist_list_name):
                h.Scale(scale_f)

            # scale histograms with MPIon and ISRon
            hist_list_name_MPIon = "hist_list_MPIon_R%s" % str(jetR).replace('.', '')
            for h in getattr(self, hist_list_name_MPIon):
                h.Scale(MPI_scale_f)

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('--user-seed', help='PYTHIA starting seed', default=1111, type=int)
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./', 
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('--no-match-level', help="Save simulation for only one level with " + \
                        "no matching. Options: 'p', 'h', 'ch'", default=None, type=str)
    parser.add_argument('--no-scale', help="Turn off rescaling all histograms by cross section / N",
                        action='store_true', default=False)
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/angularity.yaml',
                        help="Path of config file for observable configurations")
    args = parser.parse_args()

    if args.no_match_level not in [None, 'p', 'h', 'ch']:
        print("ERROR: Unrecognized type %s. Please use 'p', 'h', or 'ch'" % args.type_only)
        exit(1)

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # Use PYTHIA seed for event generation
    if args.user_seed < 0:
        args.user_seed = 1111

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    if args.py_noMPI:
        print("\033[91m%s\033[00m" % "WARNING: py-noMPI flag ignored for this program")
        time.sleep(3)
        print()

    process = pythia_parton_hadron(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)
