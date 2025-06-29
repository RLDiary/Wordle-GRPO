

import sys
sys.path.append('/home/ubuntu/research_nfs/ramya/tau-bench/')
import os
from tau_bench.envs.retail.env import MockRetailDomainEnv as MockRetailDomainEnv

import verifiers as vf
import os
from trl import GRPOConfig
import datetime

os.environ["WANDB_PROJECT"] = "GRPO-TauBench"
DEBUG = True
if DEBUG:
    print("************************************************Libraries Imported Successfully.************************************************")

# model_name = "/home/ubuntu/research_nfs/ramya/tau-bench/VLLM-Models/Llama3.1-8B-Instruct"
# model_name = "/home/ubuntu/research_nfs/ramya/tau-bench/VLLM-Models/FinalSFT_TauBench_Llama_3.1_8B_Run1_8bit-V2"
# model_name = "/home/ubuntu/research_nfs/ramya/tau-bench/RL/verifiers/verifiers/outputs/SFT-GRPO-Llama3.1-8B-Instruct-Retail_V0_2025-06-14_01-17-06/checkpoint-60"
model_name = "/home/ubuntu/research_nfs/ramya/tau-bench/RL/verifiers/verifiers/outputs/RFT-SFT-GRPO-Llama3.1-8B-Instruct-Retail_V0_2025-06-14_13-06-05/checkpoint-105"
run_name = f"RFT-SFT-GRPO-Llama3.1-8B-Instruct-Retail_V0_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"



model, tokenizer = vf.get_model_and_tokenizer(model_name)
# tokenizer.pad_token = tokenizer.eos_token

# Initialize Training Arguments
output_dir = f"/home/ubuntu/research_nfs/ramya/tau-bench/RL/verifiers/verifiers/outputs/{run_name}"
EVAL_ON_START = False
NUM_GPUS = 4
NUM_GENERATIONS = 9 # 3
NUM_EPOCHS = 1

EPSILON = 0.2
LEARNING_RATE = 1e-6 # 5e-7
SCHEDULER_TYPE = "constant_with_warmup"
MULTIPLIER_TYPE = "complexity"
SCALE_REWARDS = False

EVAL_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 8 # 25
EVAL_STEPS = 8
SAVE_STEPS = 16 # 25

MAX_GRAD_NORM = 3.0 # 1.0
NUM_ITERATIONS = 2 # 2
GRADIENT_ACCUMULATION_STEPS = 3
PER_DEVICE_BATCH_SIZE = 3 # 1
PER_DEVICE_EVAL_BATCH_SIZE = 3 # 4

# batch_size = per_device_train_batch_size * num_processes * gradient_accumulation_steps

MAX_TURNS_PER_TASK = 62 # 55
MAX_COMPLETION_LENGTH = 1000 # Maximum tokens per completion
VLLM_MAX_MODEL_LEN = 22000
VLLM_TEMPERATURE = 0.7


BETA = 0.004
UPDATE_REF_MODEL = True
REF_MODEL_MIXUP_ALPHA = 0.8
REF_MODEL_SYNC_STEPS = 20

vf_env = vf.TauBenchEnv(
    domain="retail",
    max_turns_per_task=MAX_TURNS_PER_TASK
)


train_dataset = vf_env.get_dataset(samples=100)
eval_dataset = vf_env.get_eval_dataset(samples=115)
rubric = vf_env.get_rubric()

# Specify Training Arguments
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=NUM_GPUS
)

training_args.output_dir = output_dir
training_args.overwrite_output_dir = True

training_args.num_train_epochs = NUM_EPOCHS
training_args.num_generations = NUM_GENERATIONS
training_args.max_completion_length = MAX_COMPLETION_LENGTH
training_args.report_to = "wandb"

training_args.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
training_args.gradient_checkpointing = True
training_args.num_iterations = NUM_ITERATIONS
training_args.beta = BETA
training_args.max_grad_norm = MAX_GRAD_NORM
training_args.epsilon = EPSILON
training_args.loss_type = "dapo"
training_args.scale_rewards = SCALE_REWARDS
training_args.multiplier_type = MULTIPLIER_TYPE

training_args.learning_rate = LEARNING_RATE
training_args.lr_scheduler_type = SCHEDULER_TYPE
training_args.warmup_steps = WARMUP_STEPS


training_args.use_vllm = True


training_args.vllm_server_host = "0.0.0.0"
training_args.vllm_server_port = 8000
training_args.vllm_max_model_len = VLLM_MAX_MODEL_LEN
training_args.vllm_temperature = VLLM_TEMPERATURE
training_args.vllm_gpu_memory_utilization = 0.7

training_args.eval_strategy = "steps"
training_args.eval_on_start = EVAL_ON_START
training_args.eval_steps = EVAL_STEPS
training_args.eval_accumulation_steps = EVAL_ACCUMULATION_STEPS

training_args.save_strategy = "steps"
training_args.save_steps = SAVE_STEPS

training_args.sync_ref_model = UPDATE_REF_MODEL
training_args.ref_model_mixup_alpha = REF_MODEL_MIXUP_ALPHA
training_args.ref_model_sync_steps = REF_MODEL_SYNC_STEPS

training_args.per_device_train_batch_size = PER_DEVICE_BATCH_SIZE
training_args.per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH_SIZE
training_args.eval_accumulation_steps = EVAL_ACCUMULATION_STEPS
training_args.reward_weights=vf_env.get_reward_weights()

# Initialize Trainer
trainer = vf.GRPOMultiTurnTrainer(
    run_name=run_name,
    model=model,
    processing_class=tokenizer,
    reward_funcs = rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    debug=DEBUG,
    test_hypothesis_clip_advantage=False
)

# Train
print("************************************************Initializing training...************************************************")
trainer.train()
print("************************************************Training complete.************************************************")

