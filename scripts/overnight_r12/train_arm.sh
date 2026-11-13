#!/bin/bash
# usage: train_arm.sh <arm>   (spec in runs/overnight_r12_20261109/arms/<arm>.json)
set -u
ECM=/media/fulei/fuleivla/ecm_vla; R=$ECM/runs/overnight_r12_20261109; ARM=$1
POLICY=/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning
PY=/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
GPUS=$(python3 -c "import json;print(','.join(map(str,json.load(open('$R/arms/$ARM.json'))['gpus'])))")
cd $POLICY
CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OPENPI_DATA_HOME=/media/fulei/fuleivla/acm_vla/cache/openpi \
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$POLICY/src:$POLICY/packages/openpi-client/src \
exec $PY -u $ECM/scripts/finetune_event_keyframes.py --spec $R/arms/$ARM.json
