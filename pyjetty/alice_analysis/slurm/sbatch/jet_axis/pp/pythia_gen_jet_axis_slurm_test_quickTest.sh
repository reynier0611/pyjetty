#! /bin/bash

#SBATCH --job-name=rey_pythia_test
#SBATCH --partition=test
#SBATCH --time=1:00:00
#SBATCH --array=1
#SBATCH --output=/rstorage/alice/AnalysisResults/rey/slurm-%A_%a.out

OUTDIR="/rstorage/alice/AnalysisResults/rey/$SLURM_ARRAY_JOB_ID/$BIN/$CORE_IN_BIN"
mkdir -p $OUTDIR
module use ~/heppy/modules
module load heppy/1.0
module use ~/pyjetty/modules
module load pyjetty/1.0
echo "python is" $(which python)
cd /home/rey/pyjetty/pyjetty/alice_analysis/

python process/user/rey/pythia_parton_hadron.py -c config/jet_axis/gen_jet_axis_pythia_herwig.yaml --output-dir $OUTDIR --user-seed 9 --py-pthatmin 5 --py-ecm 5020 --nev 130000 --pythiaopts HardQCD:all=on,TimeShower:pTmin=0.2
