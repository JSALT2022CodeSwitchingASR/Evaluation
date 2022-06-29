#!/usr/bin/bash
# sclite scoring with glm
# 
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

decode_dir=$1			# Espnet1 decoding directory
BW=$2					# whether to convert from Buckwalter to utf8 
# generate decoding utf8 files
if [ $BW == true ]; then
  python sclite_BW_to_utf8.py $decode_dir
  cp $decode_dir/ref_utf8.trn .
  cp $decode_dir/hyp_utf8.trn .
else 
  echo "UTF8"
  cp $decode_dir/ref.wrd.trn ref_utf8.trn
  cp $decode_dir/hyp.wrd.trn hyp_utf8.trn
fi

python trn2ctm.py hyp_utf8.trn hyp2.ctm
python trn2stm.py ref_utf8.trn ref2.stm

hubscr=${KALDI_ROOT}/tools/sctk/bin/hubscr.pl

${hubscr} -d -V -f ctm -F stm -l arabic -h hub5 -g test.glm -r ref2.stm hyp2.ctm