#!/bin/bash
# Shows the state of all background downloads.
cd "$(dirname "$0")/.." || exit 1
echo "== running download processes =="
pgrep -fa "download_(data|models)\.sh|download_chexpert_redivis\.py" | grep -v pgrep || echo "none running"
echo; echo "== ReXVal + IU X-ray ==";  tail -n 4 logs/downloads/data_master.log 2>/dev/null || echo "not started"
echo; echo "== models ==";             tail -n 5 logs/downloads/models_master.log 2>/dev/null || echo "not started"
echo; echo "== CheXpert Plus ==";      tail -n 4 logs/downloads/chexpert_redivis.log 2>/dev/null | tr '\r' '\n' | tail -n 4 || echo "not started"
echo; echo "== completed steps =="
ls logs/downloads/*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.done$//' || echo "none yet"
echo; echo "== sizes on disk =="
du -sh data/rexval data/iu-xray data/chexpert-plus models/qwen2.5-vl-7b-instruct models/chexagent-2-3b \
       models/green-radllama2-7b models/huatuogpt-vision-7b 2>/dev/null
echo; echo "== free disk =="; df -h /localstorage | tail -1
