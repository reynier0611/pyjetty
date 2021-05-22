#! /bin/bash

# This script takes an input parquet file path as an argument, and runs a python script to
# process the input file and write an output ROOT file.
# The main use is to give this script to a slurm script.

if [ "$1" != "" ]; then
  INPUT_FILE=$1
  #echo "Input file: $INPUT_FILE"
else
  echo "Wrong command line arguments"
fi

if [ "$2" != "" ]; then
  JOB_ID=$2
  echo "Job ID: $JOB_ID"
else
  echo "Wrong command line arguments"
fi

if [ "$3" != "" ]; then
  TASK_ID=$3
  echo "Task ID: $TASK_ID"
else
  echo "Wrong command line arguments"
fi

# Define output path from relevant sub-path of input file
# Note: suffix depends on file structure of input file -- need to edit appropriately for each dataset
OUTPUT_SUFFIX=$(echo $INPUT_FILE | cut -d/ -f6-10)
echo $INPUT_FILE
echo $OUTPUT_SUFFIX
OUTPUT_DIR="/rstorage/generators/jetscape_alice/tree_gen/$JOB_ID/$OUTPUT_SUFFIX"
echo "Output dir: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# Load modules
module use /software/users/james/heppy/modules
module load heppy/1.0
module use /software/users/james/pyjetty/modules
module load pyjetty/1.0
module list

# Run main script
cd /software/users/james/pyjetty/pyjetty/alice_analysis/generation
pipenv run python parquet2antuple.py -i $INPUT_FILE -o $OUTPUT_DIR/AnalysisResultsGen.root --no-progress-bar

# Move stdout to appropriate folder
mv /rstorage/generators/jetscape_alice/tree_gen/slurm-${JOB_ID}_${TASK_ID}.out /rstorage/generators/jetscape_alice/tree_gen/${JOB_ID}/


