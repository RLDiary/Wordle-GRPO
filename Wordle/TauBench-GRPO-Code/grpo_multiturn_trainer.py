from typing import Callable, Optional, Union, Any, List
import time
import json
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from collections.abc import Sized
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
from trl.extras.profiling import profiling_decorator
import copy
from verifiers.envs.taubenchretail_env import TauBenchEnv
from verifiers.envs.environment import Environment

from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.tools.bfcl_tools import INVOLVED_CLASS_TO_FUNC_DOC_PATH


from tau_bench.envs.base import Env
from tau_bench.envs import get_env

# from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
# from tau_bench.envs.retail.data import load_data
# from tau_bench.envs.retail.rules import RULES
from tau_bench.envs.retail.tools import ALL_TOOLS
from tau_bench.envs.retail.wiki import WIKI
from tau_bench.agents.learning_agent import LearningAgent
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from tau_bench.agents.learning_agent import LearningAgent
from typing import Dict, Optional, List, Any
import os
import datetime
import copy

if is_peft_available():
    from peft import PeftConfig # type: ignore



RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])





class GRPOMultiTurnTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            debug: bool = False,
            run_name: str = "",
            model_name: str = "",
            test_hypothesis_clip_advantage: bool = False,
            scale_rewards: bool = False,
            loss_type: str = "dapo",
            multiplier_type: Optional[str] = None,
            **kwargs,
    ):
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        
        self.debug = debug

        # self.processing_class.pad_token_id = self.processing_class.eos_token_id
        
        self.env = env
        self.agent = LearningAgent(
            tools_info=[tool.get_info() for tool in ALL_TOOLS],
            wiki=WIKI,
            debug=self.debug,
        )
        self._eval_started = False
        self._train_started = False
        self.scale_rewards = scale_rewards
        self.loss_type = loss_type

        self.model_name = model_name
        self._initial_eval = True
        self.run_name = run_name
        self.test_hypothesis_clip_advantage = test_hypothesis_clip_advantage
        self.multiplier_type = multiplier_type
        if train_dataset is not None:
            dataset_hash = train_dataset._fingerprint
        else:
            dataset_hash = "N/A"
        metadata_dict = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "prompt_func": "N/A",
            "parse_func": "N/A",
            "model_name": model_name,
            "run_hash": run_name,
            "batch_mode": False,
            "response_format": "N/A",
        }

        # if os.environ["CURATOR_VIEWER"] == "1":
        #     self._curator_viewer_client = Client()
        #     self._curator_session_id = self._curator_viewer_client.create_session(metadata_dict)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        if self.model.training:
            print('@@@@@@@@@@@@@@@@@Model is in training mode@@@@@@@@@@@@@@@@@', " Device: ", device)
        else:
            print('✓✓✓✓✓✓✓✓✓✓✓✓✓Model is in evaluation mode✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓', " Device: ", device)
        
        task_idxs = [x["task_idxs"] for x in inputs]

        # if self.debug:
        print('************************************************')
        print('Now performing Generation and Scoring...')
        print('The current task idxs on device {} are: {}'.format(device, task_idxs))
        print('************************************************')

        tasks = []
        prompts_text = []

        for task_idx in task_idxs:
            task_split = 'train' if self.model.training else 'test'
            env = get_env(
                    'retail',
                    user_strategy='llm',
                    user_model='gpt-4.1-mini',
                    task_split=task_split,
                    user_provider='openai',
                    task_index=task_idx
                )
            response = env.reset(task_index=task_idx)
            prompt = [
                        {"role": "system", "content": self.agent.prompt},
                        {"role": "user", "content": response.observation},
                        ]
            prompts_text.append(maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"])
            tasks.append({
                "task_idx": task_idx,
                "env": env,
                "prompt": prompt,
                "n_prompt_messages": len(prompt),
                "trajectory": copy.deepcopy(prompt),
                "multi_turn_history": copy.deepcopy(prompt),
                "prompt_ids": [],
                "completion_ids": [],
                "completion_mask": [],
                "completed": False,
                "num_turns": len(prompt),
                "complexity_score": env.complexity_score if env.complexity_score is not None else 0.0,
                "num_actions": env.num_actions if env.num_actions is not None else 0.0,
                "task_completion_reward": 0,
                "total_reward": 0,
            })
            
        prompt_input = self.processing_class(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_input = Trainer._prepare_inputs(self, prompt_input)
        prompt_ids, prompt_mask = prompt_input["input_ids"], prompt_input["attention_mask"]
        
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the all task indexes across all processes
        all_tasks = gather_object(tasks)

        if self.accelerator.is_main_process:
            print("Main Process: Generating completions...")
            env_result = self.env.generate(
                tasks=all_tasks,
                llm=self.llm,
                tokenizer=self.processing_class,
                sampling_params=self.sampling_params,
                debug=self.debug,
                training=self.model.training
            )
            
            self.env.logger.log(all_tasks, self.model.training)
            
            completion_ids = env_result['ids']
            completion_messages = env_result['trajectory_sans_prompt']
            completion_mask = env_result['mask']
            tasks = env_result['tasks']
        else:
            completion_ids = [None] * len(all_tasks)
            completion_messages = [None] * len(all_tasks)
            completion_mask = [None] * len(all_tasks)
            tasks = [None] * len(all_tasks)

        
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        tasks = broadcast_object_list(tasks, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(task_idxs),
            (self.accelerator.process_index + 1) * len(task_idxs),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        tasks = tasks[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id, padding_side='right') # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0, padding_side='right')
            
        # if self.debug:
        #     print('Shape of prompt_ids: ', prompt_ids.shape, " Device: ", device)
        #     print('Shape of completion_ids: ', completion_ids.shape, " Device: ", device)
        #     print('Shape of prompt_mask: ', prompt_mask.shape, " Device: ", device)
        #     print('Shape of completion_mask: ', completion_mask.shape, " Device: ", device)
        
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)
        
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                # if self.debug:
                #     print("Getting old per token log probabilities...", " Device: ", device)
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                # if self.debug:
                #     print("Getting ref model per token log probabilities...", " Device: ", device)
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        
        # Compute Rewards for each Reward Function
        completions = completion_messages
        rewards_per_func = torch.zeros(len(task_idxs), len(self.reward_funcs), device=device)
        
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(tasks=tasks)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        


        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore

        

        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        if self.test_hypothesis_clip_advantage:
            advantages = torch.clip(advantages, min=0) # Clip the advantages to be all positive
            # advantages = torch.clip(advantages, max=0) # Clip the advantages to be all negative
            assert (advantages >= 0).all(), f"Advantages: {advantages}"
            # assert (advantages <= 0).all(), f"Advantages: {advantages}"


        if self.debug:
            print(f"Rewards: {rewards}", "Device: ", device)
        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(task_idxs),
            (self.accelerator.process_index + 1) * len(task_idxs),
        )
        advantages = advantages[process_slice]
        
        
        
        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"
        mode = "train" if self.model.training else "eval"

        # NOTE: Log Rewards
        reward_per_func_mean = rewards_per_func.mean(0) # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # average number of “completion” tokens per example in the current batch
        avg_completion_len = (
            completion_mask          # 0/1 mask
            .sum(dim=-1)             # → per-example lengths, shape [batch_size]
            .float()                 # make sure we’re in floating-point
            .mean()                  # → scalar mean length
            .clamp(min=1.0)          # keep the metric ≥ 1 so log/ratio metrics don’t break
        )

        # record the scalar (as a Python float) in the metrics buffer
        self._metrics[mode]["completion_length"].append(avg_completion_len.item())
        
        total_num_turns = 0

        for task in tasks:
            total_num_turns += task['num_turns']       
        avg_num_turns = total_num_turns / len(tasks)
        self._metrics[mode]['average_num_turns'].append(avg_num_turns)
        
        avg_complexity_score = 0.0
        avg_num_actions = 0.0

        if self.model.training:
            total_complexity_score = 0.0
            total_num_actions = 0
            for task in tasks:
                total_complexity_score += task['complexity_score']
                total_num_actions += task['num_actions']

            avg_complexity_score = total_complexity_score / len(tasks)
            avg_num_actions = total_num_actions / len(tasks)
            self._metrics[mode]["complexity_score"].append(avg_complexity_score)
            self._metrics[mode]["num_customer_asks"].append(avg_num_actions)
        

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "avg_complexity_score": avg_complexity_score,
            "avg_num_actions": avg_num_actions,
        }

    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        if self.model.training:
            avg_complexity_score = inputs["avg_complexity_score"]
            avg_num_actions = inputs["avg_num_actions"]
        else:
            avg_complexity_score = 0.0
            avg_num_actions = 0.0

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "dapo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        
        # Multiple loss by the 1 + Average complexity of the tasks in this current batch
        if self.multiplier_type == "complexity":
            loss = loss * (1 + avg_complexity_score/100)
        elif self.multiplier_type == "num_actions":
            loss = loss * (1 + avg_num_actions/100)
        else:
            pass


        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["Per_token_kl_mean"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A step by step reasoning to process the context and inform the decision making.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

To respond to the user, always use the 'respond' action as shown above.

Try to be helpful and always follow the policy.
"""