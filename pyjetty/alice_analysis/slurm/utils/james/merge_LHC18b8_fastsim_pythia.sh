#! /bin/bash

# Script to merge output ROOT files
JOB_ID=493680
OUTPUT_DIR="/rstorage/alice/AnalysisResults/james/$JOB_ID"

# command line arguments
if [ "$1" != "" ]; then
  MERGE_JOB_ID=$1
  echo "Merge Job ID: $MERGE_JOB_ID"
else
  echo "Wrong command line arguments"
fi

if [ "$2" != "" ]; then
  BIN=$2
  echo "Bin: $BIN"
else
  echo "Wrong command line arguments"
fi

# Load modules
module use /software/users/james/heppy/modules
module load heppy/1.0
module use /software/users/james/pyjetty/modules
module load pyjetty/1.0
module list

# Merge all output files from each pt-hat bin
FILE_DIR_BASE=/rstorage/alice/AnalysisResults/james/$JOB_ID
FILES=$( find ${FILE_DIR_BASE}/tree_fastsim/*/*/*/*/*/${BIN}/* -name "*.root" )

OUT_DIR_BASE=/rstorage/alice/AnalysisResults/james/$JOB_ID
mkdir -p ${OUT_DIR_BASE}/Stage0/${BIN}
hadd -f -j 10 ${OUT_DIR_BASE}/Stage0/${BIN}/AnalysisResults.root $FILES

# Move stdout to appropriate folder
mv /rstorage/alice/AnalysisResults/james/slurm-${MERGE_JOB_ID}_${BIN}.out /rstorage/alice/AnalysisResults/james/${JOB_ID}/
