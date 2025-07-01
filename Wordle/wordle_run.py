
import Wordle as W

def main():
    model_name = '/workspace/Models/Qwen2.5-3B-Instruct'
    run_name = 'Test'
    model, tokenizer = W.get_model_and_tokenizer(model_name)
    env = W.WordleEnv()
    train_dataset = env.get_dataset(dataset='train', number_of_games=100)
    eval_dataset = env.get_dataset(dataset='test', number_of_games=100)

    training_args = W.grpo_defaults(run_name=run_name)
    training_args.report_to = 'none'
    training_args.beta = 0.0
    training_args.use_vllm = True
    training_args.num_iterations=2
    training_args.num_generations=4
    training_args.max_prompt_length=4096
    training_args.max_completion_length=4096
    training_args.max_steps=100
    training_args.vllm_mode = 'colocate'
    training_args.vllm_gpu_memory_utilization = 0.5
    training_args.vllm_tensor_parallel_size = 1
    training_args.per_device_train_batch_size = 2
    training_args.generation_batch_size = 4
    training_args.per_device_eval_batch_size = 2
    training_args.use_liger_loss = True
    training_args.bf16_full_eval = False
    training_args.bf16 = False
    rubric = W.WordleRubric()
    reward_funcs = rubric.get_reward_funcs()
    training_args.reward_weights = rubric.get_reward_weights()

    training_args.generation_kwargs = {
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "min_p": 0.5,
        "top_p": 0.95,
        "top_k": 100,
        "max_tokens": 1024
    }

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



if __name__ == "__main__":
    main()

