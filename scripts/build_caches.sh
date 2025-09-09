#!/usr/bin/env
set -euo pipefail

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-meta-llama/Llama-3.1-70B-Instruct}

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Input: $IN"

python teacher_farm/make_topk_cache.py \        #---|
    --model "$TEACHER" \                        #   |
    --input_jsonl "$IN" \                       #    ====== RB top-k
    --out_dir data/topk_k16/ \                  #   |
    --k 16                                      #---|

python teacher_farm/make_hidden_cache.py \     #----|
    --model "$TEACHER" \                       #    |
    --input_jsonl "$IN" \                      #    ===== FB Hidden (L22)
    --out_dir data/fb_hints_L22/ \             #    |
    --layers 22                                #----|

python teacher_farm/make_embed_cache.py \      #----|
    --model "$TEACHER" \                       #    |    
    --input_jsonl "$IN" \                      #    ===== RELB pooled embs
    --out_dir data/relb_embeds/                #----|



## TODO: Insert the system needs for slurm.