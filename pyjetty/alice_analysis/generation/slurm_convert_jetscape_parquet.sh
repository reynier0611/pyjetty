#! /bin/bash

#SBATCH --job-name=convert-jetscape
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --partition=std
#SBATCH --time=24:00:00
#SBATCH --array=1-583
#SBATCH --output=/rstorage/generators/jetscape_alice/tree_gen/slurm-%A_%a.out

FILE_PATHS='/rstorage/jetscape/JETSCAPE-AA-events/skim/452210/v2/OutputFile_Type5_qhatA10_B100_5020_PbPb_0-10_0.30_2.0_1/files.txt'
NFILES=$(wc -l < $FILE_PATHS)
echo "N files to process: ${NFILES}"

# Currently we have 8 nodes * 20 cores active
FILES_PER_JOB=1
echo "Files per job: $FILES_PER_JOB"

STOP=$(( SLURM_ARRAY_TASK_ID*FILES_PER_JOB ))
START=$(( $STOP - $(( $FILES_PER_JOB - 1 )) ))

if (( $STOP > $NFILES ))
then
  STOP=$NFILES
fi

echo "START=$START"
echo "STOP=$STOP"

for (( JOB_N = $START; JOB_N <= $STOP; JOB_N++ ))
do
  FILE=$(sed -n "$JOB_N"p $FILE_PATHS)
  srun process_convert_jetscape_parquet.sh $FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
done
