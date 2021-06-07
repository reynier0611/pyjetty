for pthat in {1..20}
do
	echo "MERGING FILES FOR pT_hat BIN #"$pthat
	source local_merge_LHC20g4.sh 1 $pthat
done
