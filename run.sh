#! /bin/bash

BASEDIR=$(readlink -f ../labAHLT)
export PYTHONPATH=$BASEDIR/util

if [[ "$*" == *"parse"* ]]; then
   $BASEDIR/util/corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
   sleep 1

   python3 parse_data.py $BASEDIR/data/train train
   python3 parse_data.py $BASEDIR/data/devel devel
   kill `cat /tmp/corenlp-server.running`
fi

if [[ "$*" == *"train"* ]]; then
    rm -rf model model.idx
    python3 train.py train.pck devel.pck model
fi

if [[ "$*" == *"predict"* ]]; then
   rm -f devel.stats devel.out
   python3 predict.py model devel.pck devel.out 
   python3 $BASEDIR/util/evaluator.py DDI $BASEDIR/data/devel devel.out | tee devel.stats
fi


