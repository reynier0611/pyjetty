#! /bin/bash

#SBATCH --job-name=rgtest
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --partition=std
#SBATCH --time=4:00:00
#SBATCH --array=1-140
#SBATCH --output=/storage/u/alice/AnalysisResults/slurm-%A_%a.out

FILE_PATHS='/rstorage/u/alice/LHC18b8/146/files_test.txt'
NFILES=$(wc -l < $FILE_PATHS)
echo "N files to process: ${NFILES}"

# Currently we have 7 nodes * 20 cores active
FILES_PER_JOB=$(( $NFILES / 140 + 1 ))
echo "Files per job: $FILES_PER_JOB"

STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_JOB ))
START=$(( $STOP - $(( $FILES_PER_JOB - 1 )) ))

if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi

echo "START=$START"
echo "STOP=$STOP"

OUTPUT_PREFIX="AnalysisResults/$SLURM_ARRAY_JOB_ID"
for (( JOB_N = $START; JOB_N <= $STOP; JOB_N++ ))
do
  FILE=$(sed -n "$JOB_N"p $FILE_PATHS)
  srun process_rg_LHC18b8.sh $FILE $OUTPUT_PREFIX
done