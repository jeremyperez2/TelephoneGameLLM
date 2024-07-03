#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=00:29:29
#SBATCH --gres=gpu:2
#SBATCH --array=0-99 # n_story x n_seeds -> 20x5  -> 0-99
#SBATCH -o slurm_logs/sb_log_%A_%a.out
#SBATCH -e slurm_logs/sb_log_%A_%a.err
##SBATCH --qos=qos_gpu-dev

model_name="$1"
model=$HF_HOME"/"$model_name

prompt="$2"


seed_list=(1 2 3 4 5)
initial_stories=('Positivity-0.9738' 'Positivity-0.5106' 'Positivity-0.0653' 'Positivity0.5994' 'Positivity0.9019' 'Difficulty11.57' 'Difficulty14.78' 'Difficulty22.9' 'Difficulty24.83' 'Difficulty32.24' 'Toxicity0.00113261' 'Toxicity0.1864273' 'Toxicity0.46547398' 'Toxicity0.8434329' 'Toxicity0.9934216' 'Length518.0' 'Length1635.0' 'Length2752.0' 'Length3869.0' 'Length4986.0')



seed_list_len=${#seed_list[@]}


story_i=$(( SLURM_ARRAY_TASK_ID / $seed_list_len ))

seed_i=$(( SLURM_ARRAY_TASK_ID % $seed_list_len ))

story="${initial_stories[$story_i]}"
seed="${seed_list[$seed_i]}"


echo "ID:"$SLURM_ARRAY_TASK_ID
echo "Prompt:"$prompt
echo "Seed:"$seed
echo "Model:"$model
echo "Story":$story
# Define the output folder
##########################################################
SUBDIR=$model_name"/"$prompt"/"$story"/"seed_$seed

SAVE_DIR="Results/"$SUBDIR
LOG_DIR="logs/"$SUBDIR

echo "save_dir:"$SAVE_DIR
# ## Start the server
# ##########################################################
# python server.py

# access_url="http://127.0.0.1:5000"
access_url='https://heaven-impressed-victoria-totals.trycloudflare.com'
# Start the experiment
##########################################################
mkdir -p $LOG_DIR

source $HOME/.bashrc
ray start --head --num-cpus=32
## define the conda env to use
#conda activate LLM-Culture

# Other params
##########################################################

n_agents=50
n_timesteps=50
prompt_init=$prompt
prompt_update=$prompt
network_structure_type="sequence"
n_seeds=1
n_edges=1
personality_list="Empty"
n_cliques=1

# Run the experiment

VLLM_SWAP_SPACE=32 python -u main_simulation.py \
    --n_agents $n_agents \
    --n_timesteps $n_timesteps \
    --prompt_init $prompt_init \
    --prompt_update $prompt_update \
    --format_prompt "Empty" \
    --start_flag None \
    --end_flag None \
    --network_structure $network_structure_type \
    --n_seeds $n_seeds \
    --n_edges $n_edges \
    --personality_list $personality_list \
    --output_dir $SAVE_DIR \
    --access_url $access_url \
    --n_cliques $n_cliques \
    --initial_story $story \
    --use_vllm 'True' \
    --model $model