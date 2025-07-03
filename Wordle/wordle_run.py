
import Wordle as W
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

accelerator = Accelerator()

import os
os.environ["WANDB_PROJECT"] = "Wordle-GRPO"

import datetime
date = datetime.datetime.now().strftime("%d-%m")
time = datetime.datetime.now().strftime("%H-%M")

def shared_dataset(env, split: str, n_games: int):
    """
    • Rank-0 samples the games           (any RNG is confined to that process)
    • The list is broadcast to all ranks (deterministic without touching seeds)
    """
    if accelerator.is_main_process:           # rank-0
        data = env.get_dataset(split, n_games)
        payload = [data]
    else:                                     # all other ranks
        payload = [None]

    broadcast_object_list(payload, from_process=0)
    return payload[0]

def main():
    model_name = 'vigneshR/Qwen2.5-3B-WORDLE-FineTune'
    run_name = f'Initial-A5000-TestRuns-{date}-{time}'
    
    print('Loading Model...')
    model, tokenizer = W.get_model_and_tokenizer(model_name)
    print('Model Loaded')
    
    # Initialize Environment and Get Dataset
    env = W.WordleEnv()
    train_dataset = shared_dataset(env, 'all', 1000)
    print('Now waiting for everyone to finish...')
    accelerator.wait_for_everyone()

    # Initialize Training Arguments
    training_args = W.grpo_defaults(run_name=run_name)

    # Saving Config
    training_args.save_strategy = "steps"
    training_args.save_steps = 300
    training_args.output_dir = f"/workspace/Wordle-GRPO/Saved-Models/{run_name}"
    training_args.overwrite_output_dir = True
    training_args.save_total_limit = 5

    # Evaluation Config
    training_args.eval_strategy = "no"
    # training_args.per_device_eval_batch_size = 4

    # Learning Rate, Beta and KL Divergence
    training_args.learning_rate = 1e-6
    training_args.beta = 0.0
    training_args.sync_ref_model = False
    training_args.ref_model_mixup_alpha = 0.0
    
    # Training Config
    training_args.num_iterations=2
    training_args.num_generations=6

    # Batch Size Parameters
    training_args.per_device_train_batch_size = 3
    training_args.steps_per_generation = 1
    training_args.max_steps=1000

    
    # Max Length Parameters
    training_args.max_prompt_length=4096
    training_args.max_completion_length=4096
    
    

    # VLLM Config
    training_args.use_vllm = True
    training_args.vllm_mode = 'colocate'
    training_args.vllm_gpu_memory_utilization = 0.3
    training_args.vllm_tensor_parallel_size = 1
    
    # Loss Config
    training_args.loss_type = 'bnpo'
    training_args.scale_rewards = False
    training_args.use_liger_loss = True
    training_args.bf16_full_eval = True
    training_args.bf16 = True
    
    # Wandb Config
    training_args.report_to = 'wandb'
    training_args.logging_steps = 1
    training_args.run_name = run_name
    
    # Reward Config
    rubric = W.WordleRubric()
    reward_funcs = rubric.get_reward_funcs()
    training_args.reward_weights = rubric.get_reward_weights()

    # Generation Config
    training_args.generation_kwargs = {
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "min_p": 0.5,
        "top_p": 0.95,
        "top_k": 100,
        "max_tokens": 1024
    }

    # Printing Completions
    training_args.num_completions_to_print = 1

    trainer = W.GRPOMultiTurnTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()



if __name__ == "__main__":
    main()

