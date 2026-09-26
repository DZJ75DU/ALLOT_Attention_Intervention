#!/usr/bin/env bash
#SBATCH --job-name=allot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-24%4
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
DROOT=${DROOT:-/dataset}
MODEL_ROOT=${MODEL_ROOT:-/models}
MODEL=${MODEL:-}
OUT=${OUT:-$ROOT/Experiment_result}

if [[ -z "$MODEL" ]]; then
  echo "ERROR: set MODEL to a model directory name under MODEL_ROOT." >&2
  echo "Example: sbatch --export=ALL,MODEL=Qwen3-8B run.slurm" >&2
  exit 2
fi
mkdir -p "$OUT" "$ROOT/logs"

SEEDS_spm05=${SEEDS_spm05:-"0 1 2 3 4"}
SEEDS_spm07=${SEEDS_spm07:-"0 1 2 3 4"}
SEEDS_spm09=${SEEDS_spm09:-"0 1 2 3 4"}
SEEDS_mutag=${SEEDS_mutag:-"0 1 2 3 4"}
SEEDS_sst=${SEEDS_sst:-"0 1 2 3 4"}

manifest() {
  for seed in $SEEDS_spm05; do echo "SPMotif-0.5 $seed"; done
  for seed in $SEEDS_spm07; do echo "SPMotif-0.7 $seed"; done
  for seed in $SEEDS_spm09; do echo "SPMotif-0.9 $seed"; done
  for seed in $SEEDS_mutag; do echo "mutag $seed"; done
  for seed in $SEEDS_sst; do echo "Graph-SST2 $seed"; done
}

CELLS=$(manifest)
NCELL=$(printf '%s\n' "$CELLS" | wc -l | tr -d ' ')
TID=${SLURM_ARRAY_TASK_ID:-${TID:-0}}
if (( TID >= NCELL )); then
  echo "ERROR: array index $TID exceeds the $NCELL configured cells." >&2
  exit 2
fi
LINE=$(printf '%s\n' "$CELLS" | sed -n "$((TID + 1))p")
read -r DS SEED <<< "$LINE"

COMMON=(
  --model "$MODEL" --model_root "$MODEL_ROOT" --dataset_root "$DROOT"
  --extractor gsat --gnn_type GCN --gnn_layers 3 --gnn_hidden_dim 128
  --target_ratio 0.15 --gsat_kl_lambda 3.0 --gsat_temperature 1.0
  --use_attention_modify --attention_mod_mode pre_softmax_bias --per_token_norm
  --classifier_type linear --num_samples_per_class -1
  --lr 1e-3 --epochs 100 --use_early_stopping --early_stop_patience 20
  --llm_device_map auto --attn_implementation eager --dtype bfloat16
)

OUTDIR=$OUT/$DS/s$SEED
DONE=$OUTDIR/$DS/$MODEL/train_result.json
if [[ -f "$DONE" ]]; then
  echo ">>> [skip] finished: $DONE"
  exit 0
fi
mkdir -p "$OUTDIR"

echo ">>> dataset=$DS seed=$SEED model=$MODEL"
command -v nvidia-smi >/dev/null 2>&1 && \
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

case "$DS" in
  SPMotif-0.5|SPMotif-0.7|SPMotif-0.9)
    DATA_ARGS=(
      --dataset_name "$DS"
      --llm_max_seq_len 2048
      --attention_gamma 5.5 --batch_size 1
    )
    ;;
  mutag)
    DATA_ARGS=(
      --dataset_name mutag --prompt_mode atom_pair
      --llm_max_seq_len 2048
      --attention_gamma 7.5 --batch_size 1
    )
    ;;
  Graph-SST2)
    DATA_ARGS=(
      --dataset_name Graph-SST2 --prompt_mode word_pair
      --llm_max_seq_len 2048
      --attention_gamma 4.0 --batch_size 1
    )
    ;;
  *)
    echo "ERROR: unsupported dataset $DS" >&2
    exit 2
    ;;
esac

srun "$PYTHON" "$ROOT/train.py" "${COMMON[@]}" "${DATA_ARGS[@]}" \
  --eval_metric acc --seed "$SEED" --output_dir "$OUTDIR"
rc=$?

if (( rc == 0 )) && [[ -f "$DONE" ]]; then
  echo ">>> [done] $DS seed=$SEED -> $DONE"
else
  present=no
  [[ -f "$DONE" ]] && present=yes
  echo ">>> [FAILED] $DS seed=$SEED (rc=$rc, result present: $present)"
fi
exit "$rc"
