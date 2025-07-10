# Specify which GPUs to use
GPUS=(0 1 2 3 4 5)  # Modify this array to specify which GPUs to use
SEEDS=(0 1 2 3 4)
NUM_EACH_GPU=3
PARALLEL=$((NUM_EACH_GPU * ${#GPUS[@]}))

TASKS=(
    # "dmc-acrobot-swingup"
    # "dmc-ball_in_cup-catch"
    # "dmc-cartpole-balance"
    # "dmc-cartpole-swingup"
    # "dmc-cheetah-run"
    # "dmc-dog-run"
    # "dmc-dog-stand"
    # "dmc-dog-trot"
    # "dmc-dog-walk"
    # "dmc-finger-spin"
    # "dmc-finger-turn_easy"
    # "dmc-finger-turn_hard"
    # "dmc-fish-swim"
    # "dmc-hopper-hop"
    # "dmc-hopper-stand"
    # "dmc-humanoid-run"
    # "dmc-humanoid-stand"
    "dmc-humanoid-walk"
    # "dmc-pendulum-swingup_dense"
    # "dmc-quadruped-run"
    # "dmc-quadruped-walk"
    # "dmc-reacher-easy"
    # "dmc-reacher-hard"
    # "dmc-walker-run"
    # "dmc-walker-stand"
    # "dmc-walker-walk"
    # "dmc-cartpole-balance_sparse"
    # "dmc-cartpole-swingup_sparse"
    # "dmc-pendulum-swingup"
)


SHARED_ARGS=(
    "obs=rgb"
)

run_task() {
    task=$1
    seed=$2
    slot=$3
    # Calculate device index based on available GPUs
    num_gpus=${#GPUS[@]}
    device_idx=$((slot % num_gpus))
    device=${GPUS[$device_idx]}
    echo "Running $env $seed on GPU $device"
    export CUDA_VISIBLE_DEVICES=$device
    command="python3 train.py task=$task seed=$seed ${SHARED_ARGS[@]}"
    if [ -n "$DRY_RUN" ]; then
        echo $command
    else
        echo $command
        $command
    fi
}

. env_parallel.bash
if [ -n "$DRY_RUN" ]; then
    env_parallel -P${PARALLEL} run_task {1} {2} {%} ::: ${TASKS[@]} ::: ${SEEDS[@]}
else
    env_parallel --bar --results log/parallel/$name -P${PARALLEL} run_task {1} {2} {%} ::: ${TASKS[@]} ::: ${SEEDS[@]}
fi
