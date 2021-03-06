#!/bin/bash

# ./hjet.py --biaspow 4 --biasref 6  --nev 1000  -o hjet_6.root
# ./hjet.py --biaspow 4 --biasref 12 --nev 1000 -o hjet_12.root
# ./hjet.py --biaspow 4 --biasref 20 --nev 1000 -o hjet_20.root

# ./hjet.py --pthatmin 6  --nev 10000 -o hjet_6.root
# ./hjet.py --pthatmin 12 --nev 10000 -o hjet_12.root
# ./hjet.py --pthatmin 20 --nev 10000 -o hjet_20.root

runs=$(seq 1 8)
# parallel --dry-run --joblog hjet_simple_hTT.log --keep-order --tag ./hjet_simple.py --charged --py-pthatmin 20 --nev 100000 --t-min 20 --t-max 30 --runid {} ::: ${runs}
# parallel --dry-run --joblog hjet_simple_lTT.log --keep-order --tag ./hjet_simple.py --charged --py-pthatmin 6  --nev 100000 --t-min 6  --t-max 7 --runid {}  ::: ${runs}

#pthats="4 11 21 36 56 84 117 156 200 249 1000"
#parallel --joblog hjet_simple_hTT.log --keep-order --tag ./hjet_simple_mTT.py --noInel --charged --py-pthatmin {2} --nev 10000 --tranges 6-7,20-30 --runid {1} ::: ${runs} ::: ${pthats}

pthats="4 11 21 36 56 84 117 156 200 249 1000"

run=${1}
overw=""
[ "x${2}" == "xoverwrite" ] && overw="--overwrite"
if [ "x${run}" == "xrun" ]; then
	njobs=${2}
	dryrun=""
	[ "x${2}" == "xdry" ] && dryrun="--dry-run" && njobs=${3}
	[ "x${3}" == "xdry" ] && dryrun="--dry-run"
	[ -z ${njobs} ] && njobs=1
	set -x
	runs=$(seq 1 ${njobs})
	nev=100000
	# parallel ${dryrun} --joblog hjet_simple_hTT_softQCD.log --keep-order --tag ${PYJETTYDIR}/pyjetty/hjet/hjet_simple_mTT.py ${overw} --inel --charged --nev ${nev} --tranges 6-7,20-30 --runid {1} --py-pthatmin {2} ::: ${runs} ::: ${pthats}
	# parallel ${dryrun} --joblog hjet_simple_hTT_softQCD.log --keep-order --tag ${PYJETTYDIR}/pyjetty/hjet/hjet_simple_mTT.py ${overw} --charged --nev ${nev} --tranges 6-7,20-30 --runid {1} --py-pthatmin {2} ::: ${runs} ::: ${pthats}
	# nev=100000
	parallel ${dryrun} --joblog hjet_simple_hTT_hard.log --keep-order --tag ${PYJETTYDIR}/pyjetty/hjet/hjet_simple_mTT.py ${overw} --hard --py-biasref 50 --charged --nev ${nev} --tranges 6-7,20-30 --runid {1} ::: ${runs}
	# parallel ${dryrun} --joblog hjet_simple_hTT_hard.log --keep-order --tag ${PYJETTYDIR}/pyjetty/hjet/hjet_simple_mTT.py ${overw} --hard --py-biasref 6 --charged --nev ${nev} --tranges 6-7,20-30 --runid {1} ::: ${runs}
	# parallel ${dryrun} --joblog hjet_simple_hTT_inel.log --keep-order --tag ${PYJETTYDIR}/pyjetty/hjet/hjet_simple_mTT.py ${overw} --inel --charged --nev ${nev} --tranges 6-7,20-30 --runid {1} ::: ${runs}
	# parallel ${dryrun} --joblog hjet_simple_hTT_pthat.log --keep-order --tag ${PYJETTYDIR}/pyjetty/hjet/hjet_simple_mTT.py ${overw} --py-pthatmin 6. --charged --nev ${nev} --tranges 6-7,20-30 --runid {1} ::: ${runs}
	set +x
fi

if [ "x${run}" == "xtdraw" ]; then
	tdraw_cfg.py tdraw_hjet.cfg --clean
fi

if [ "x${run}" == "xhadd" ]; then
	_slist=$(ls h_jet_ch_R04_tranges_6-7_20-30_runid_*_hard_houtput.root)
	if [ ! -z "${_slist}" ]; then
		hadd -f hjet_hard.root h_jet_ch_R04_tranges_6-7_20-30_runid_*_hard_houtput.root
	fi
	_slist=$(ls h_jet_ch_R04_tranges_6-7_20-30_runid_*_inel_houtput.root)
	if [ ! -z "${_slist}" ]; then
		hadd -f hjet_inel.root h_jet_ch_R04_tranges_6-7_20-30_runid_*_inel_houtput.root
	fi
	_slist=$(ls h_jet_ch_R04_tranges_6-7_20-30_runid_*_pthatmin_6.0_houtput.root)
	if [ ! -z "${_slist}" ]; then
		hadd -f hjet_pthatmin6.root h_jet_ch_R04_tranges_6-7_20-30_runid_*_pthatmin_6.0_houtput.root
	fi
	for pth in ${pthats}
	do
		# hadd -f hjet_pthatmin_${pth}_inel.root h_jet_ch_R04_tranges_6-7_20-30_runid_*_pthatmin_${pth}.0_inel.root
		_slist=$(ls h_jet_ch_R04_tranges_6-7_20-30_runid_*_pthatmin_${pth}.0_hard_houtput.root)
		if [ ! -z "${_slist}" ]; then
			hadd -f hjet_pthatmin_${pth}_hard.root ${_slist}
		fi
	done
fi
