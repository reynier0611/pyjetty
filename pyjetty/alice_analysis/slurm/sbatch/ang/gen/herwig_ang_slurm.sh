#! /bin/bash

#SBATCH --job-name="pythiagen"
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --array=1-320
#SBATCH --output=/rstorage/alice/AnalysisResults/ang/slurm-%A_%a.out

# Number of events per pT-hat bin (for statistics)
NEV_DESIRED=3200000

# Lower edges of the pT-hat bins
PTHAT_BINS=(5 9 16 28 45 70 99 132 169 212)
echo "Number of pT-hat bins: ${#PTHAT_BINS[@]}"

# Currently we have 8 nodes * 20 cores active
NCORES=320
NEV_PER_JOB=$(( $NEV_DESIRED * ${#PTHAT_BINS[@]} / $NCORES ))
echo "Number of events per job: $NEV_PER_JOB"
NCORES_PER_BIN=$(( $NCORES / ${#PTHAT_BINS[@]} ))
echo "Number of cores per pT-hat bin: $NCORES_PER_BIN"

BIN=$(( ($SLURM_ARRAY_TASK_ID - 1) / $NCORES_PER_BIN + 1))
CORE_IN_BIN=$(( ($SLURM_ARRAY_TASK_ID - 1) % $NCORES_PER_BIN + 1))
PTHAT_MIN=${PTHAT_BINS[$(( $BIN - 1 ))]}
if [ $BIN -lt ${#PTHAT_BINS[@]} ]; then
	PTHAT_MAX=${PTHAT_BINS[$BIN]}
	echo "Calculating bin $BIN (pThat=[$PTHAT_MIN,$PTHAT_MAX]) with core number $CORE_IN_BIN"
else
	echo "Calculating bin $BIN (pThat_min=$PTHAT_MIN) with core number $CORE_IN_BIN"
fi

SEED=$(( ($CORE_IN_BIN - 1) * NEV_PER_JOB + 1111 ))

# Set modules and load Herwig environment
module use ~/heppy/modules
module load heppy/1.0
module use ~/pyjetty/modules
module load pyjetty/1.0
echo "python is" $(which python)
source /software/users/james/herwig/bin/activate

HERWIG_SCRIPT="/home/ezra/herwig_infiles/$BIN/LHC_5020.run"
PYTHON_SCRIPT="/home/ezra/pyjetty/pyjetty/alice_analysis/process/user/ang_pp/herwig_parton_hadron.py"
CONFIG="/home/ezra/pyjetty/pyjetty/alice_analysis/config/ang/process_angularity.yaml"
OUTDIR="/rstorage/alice/AnalysisResults/ang/$SLURM_ARRAY_JOB_ID/$BIN/$CORE_IN_BIN"
mkdir -p $OUTDIR
cd $OUTDIR

Herwig run $HERWIG_SCRIPT -d2 -N $NEV_PER_JOB -s $SEED

cd /home/ezra/analysis_env/
pipenv run python $PYTHON_SCRIPT -c $CONFIG --input-file $OUTDIR/LHC_5020.log --output-dir $OUTDIR

# Clean up Herwig7 files to save space
rm $OUTDIR/LHC_5020*
