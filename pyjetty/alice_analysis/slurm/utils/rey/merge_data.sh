#! /bin/bash
#
# Script to merge output ROOT files

JOB_ID=470375

FILE_DIR="/rstorage/alice/AnalysisResults/rey/$JOB_ID"
FILES=$( find "$FILE_DIR" -name "*.root" )

OUTPUT_DIR=/rstorage/alice/AnalysisResults/rey/$JOB_ID
hadd -f -j 20 $OUTPUT_DIR/AnalysisResultsFinal.root $FILES
