OUTDIR="round_71"

CONFIGDIR="/home/rey/pyjetty/pyjetty/alice_analysis/config/jet_axis/configs_broken_down_in_pT_and_jetR/"
CONFIGCOPY=$CONFIGDIR$OUTDIR
OUTPATH="/rstorage/alice/AnalysisResults/rey/"
CODEPATH="~/pyjetty/pyjetty/alice_analysis/"

# Check if there's a directory called $OUTDIR in the main config directory. If it does not exist, create one.
if [[ ! -e $CONFIGCOPY ]]; then
	mkdir $CONFIGCOPY
fi

# loop over jet R values being considered
for R in 0.4 #0.2
do
	# loop over the pT bins being considered
	for pT in "20_40" "40_60" "60_80" "80_100" "20_100"
	do
		# loop over types of jet axes
		for ax in "Standard_SD" "WTA"
		do
                        # Run the analysis
			CONFIG=$CONFIGDIR"rey_pp_R_"$R"_pT_"$pT"_GeV_"$ax".yaml"

                        cp $CONFIG $CONFIGCOPY

			python analysis/user/rey/run_analysis_jet_axis.py -c $CONFIG 

                        # Go to the output file and organize the directories
			cd $OUTPATH
			cd jet_axis/

                        # Create the output directories	
                        name="new_R_"$R"_"$ax"_pT_"$pT
                        mkdir $name

                        # Move everything there
                        cp $CONFIG $name

			# Move all the resulting directories to $name
			for folder in "final_results" "main" "performance" "prior1" "prior2" "systematics" "trkeff" "truncation" "unfolding_tests" "fastsim_generator0" "fastsim_generator1" "binning"
			do
				mv $folder $name
			done

			cd $CODEPATH
		done
	done
done

cd $OUTPATH
cd jet_axis/
mkdir $OUTDIR
mv new_* $OUTDIR
cd $CODEPATH
