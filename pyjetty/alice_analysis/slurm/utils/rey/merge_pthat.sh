#! /bin/bash
#
# Script to merge output ROOT files from all pt-hat bins together, in stages
JOB_ID=469387
FILE_DIR=/rstorage/alice/AnalysisResults/rey/$JOB_ID
OUTPUT_DIR=/rstorage/alice/AnalysisResults/rey/$JOB_ID

# Merge all output files from each pt-hat bin
hadd -f -j 10 $OUTPUT_DIR/AnalysisResultsFinal.root $FILE_DIR/Stage0/*/*.root
