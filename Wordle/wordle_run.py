
import Wordle as W

def main():
    model_name = '/home/ubuntu/TauBench/tau-bench/Model Tuning/Qwen2.5-0.5B'
    run_name = 'Test'
    model, tokenizer = W.get_model_and_tokenizer(model_name)
    env = W.WordleEnv()
    train_dataset = env.get_dataset(dataset='train', number_of_games=100)
    eval_dataset = env.get_dataset(dataset='test', number_of_games=100)

    training_args = W.grpo_defaults(run_name=run_name)
    training_args.report_to = 'none'
    training_args.use_vllm = True
    training_args.num_iterations=1
    training_args.per_device_train_batch_size=6
    training_args.num_generations=12
    training_args.gradient_accumulation_steps=4
    training_args.max_prompt_length=1024
    training_args.max_completion_length=4096
    training_args.max_steps=100
    training_args.vllm_mode = 'colocate'
    training_args.vllm_gpu_memory_utilization = 0.3
    training_args.vllm_tensor_parallel_size = 1
    training_args.use_liger_loss = True
    rubric = W.WordleRubric()
    reward_funcs = rubric.get_reward_funcs()
    training_args.reward_weights = rubric.get_reward_weights()

    trainer = W.GRPOMultiTurnTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


main()
# if __name__ == "__main__":
    # main()

