import json
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.envs.environment import Environment

from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
from typing import List, Dict, Any
from tau_bench.agents.base import Agent
from verifiers.rubrics import TauBenchRubric
import re

import datetime
from langfuse import Langfuse
import os
from dotenv import load_dotenv
from copy import deepcopy
from litellm import completion

from pydantic import BaseModel
from ..imports import LLM, SamplingParams
from typing import Sequence
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


load_dotenv()

class Output(BaseModel):
    text: str
    token_ids: List[Any]

class AgentResponse(BaseModel):
    outputs: List[Output]
    prompt_token_ids: List[Any]

class Logger:
    def __init__(self):
        self.langfuse = self.initialize_langfuse()
        self.rubric = TauBenchRubric()
    
    def initialize_langfuse(self):
        return Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
    
    def log_langfuse(self, task: Dict[str, Any], training: bool = True):
        
        total_reward = 0.0
        for i, reward_func in enumerate(self.rubric.reward_funcs):
            total_reward += reward_func(tasks=[task])[0] * self.rubric.reward_weights[i]
        
        trace = self.langfuse.trace(
            name=f"Train_Task-ID-{task.get('task_idx')}" if training else f"Eval_Task-ID-{task.get('task_idx')}",
            input=task.get("prompt"),
            output=task.get("trajectory")[2:],
            metadata={"completion_reward": task.get("task_completion_reward"),
                      "total_reward": total_reward,
                      "num_turns": task.get("num_turns"),
                      "complexity_score": task.get("complexity_score"),
                      "num_actions": task.get("num_actions")}
        )
        
        
        for turn, i in enumerate(range(1, len(task.get("trajectory")), 2), start=1):
            user_msg = task.get("trajectory")[i - 1]
            assistant_msg = task.get("trajectory")[i]

            trace.generation(
                name=f"turn-{turn}",
                input=[user_msg],
                output=[assistant_msg]
            )
        

        self.langfuse.flush()
        
    
    def log(self, tasks: List[Dict[str, Any]], training: bool = True):
        for task in tasks:
            self.log_langfuse(task, training)
        print(f"----LOGGING COMPLETED: Task-IDs-{', '.join([str(task.get('task_idx')) for task in tasks])} to Langfuse----")

class TauBenchEnv(Environment):
    def __init__(self, domain: str, max_turns_per_task: int = 50, mask_env_response: bool = True):
        super().__init__()
        self.domain = domain
        self.max_turns_per_task = max_turns_per_task
        self.rubric = TauBenchRubric()       
        self.env_mask = 0 if mask_env_response else 1
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
            "include_stop_str_in_output": False,
            "truncate_prompt_tokens": 19500,
        }         
        self.logger = Logger()
        self.max_workers = 10
        self._ACTION_RE = re.compile(r"(?is)Action:\s*(\{.*?\}\})")

        self.supervisor_model = "claude-3-7-sonnet-latest"
        self.supervisor_model_provider = "anthropic"
        
    
    def get_dataset(self, samples: int):
        sorted_tasks = sorted(TASKS_TRAIN, key=lambda x: (x.complexity_score, x.num_tasks))
        task_idxs = [TASKS_TRAIN.index(task) for task in sorted_tasks]
        return Dataset.from_list([{"task_idxs": idx} for idx in task_idxs[-samples:]])

    def get_eval_dataset(self, samples: int):
        task_idxs = list(range(samples))
        return Dataset.from_list([{"task_idxs": idx} for idx in task_idxs])

    def get_reward_funcs(self):
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self):
        return self.rubric.get_reward_weights()

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def _supervisor_completion(self, trajectory: List[Dict[str, Any]], llm: LLM, tokenizer: Any, sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        agent_response = AgentResponse(outputs=[], prompt_token_ids=[])
        res = completion(model=self.supervisor_model, custom_llm_provider=self.supervisor_model_provider, messages=trajectory)
        text = res.choices[0].message.content
        formatted_text = text + "<|eot_id|>"
        formatted_token_ids = tokenizer(formatted_text, add_special_tokens=False)["input_ids"]
        prompt_token_ids = llm.chat([trajectory], sampling_params=sampling_params, use_tqdm=False)[0].prompt_token_ids

        agent_response.outputs = [Output(text=text, token_ids=formatted_token_ids)]
        agent_response.prompt_token_ids = prompt_token_ids
        return [agent_response]
    
    def step(self, tasks: List[Dict[str, Any]], llm: LLM, tokenizer: Any, sampling_params: SamplingParams, assist: bool = False, training: bool = True) -> List[Dict[str, Any]]:
        live_indices = [i for i, t in enumerate(tasks) if not t["completed"]]
        messages_to_step = [tasks[i]["trajectory"] for i in live_indices]

        if assist:
            agent_responses = self._supervisor_completion(messages_to_step[0], llm, tokenizer, sampling_params)
        else:
            agent_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False)

        def update_task(j, agent_response):
            # Isolate the Agent Response
            agent_response_text = agent_response.outputs[0].text
            task = deepcopy(tasks[j])

            if len(task["prompt_ids"]) == 0:
                task["prompt_ids"] = list(agent_response.prompt_token_ids)
            task["trajectory"].append({"role": "assistant", "content": agent_response_text})
            
            # This code is for the future, if we want to use reasoning in the multi-turn history
            if "<reasoning>" in agent_response_text and "</reasoning>" in agent_response_text:
                agent_response_without_reasoning = agent_response_text.split("</reasoning>")[1]
                task["multi_turn_history"].append({"role": "assistant", "content": agent_response_without_reasoning})
            else:
                task["multi_turn_history"].append({"role": "assistant", "content": agent_response_text})
            
            
            # Update Completion IDs and Completion Masks
            total_prev_len = len(task["prompt_ids"]) + len(task["completion_ids"])
            env_response_len  = len(list(agent_response.prompt_token_ids)) - total_prev_len # type: ignore
            new_completion_len = len(agent_response.outputs[0].token_ids)
            
            task["completion_mask"].extend([self.env_mask] * env_response_len)
            task["completion_mask"].extend([1] * new_completion_len)

            task["completion_ids"] = list(agent_response.prompt_token_ids) # type: ignore
            task["completion_ids"].extend(list(agent_response.outputs[0].token_ids))
            task["completion_ids"] = task["completion_ids"][len(task["prompt_ids"]):]

            # Check to see if the last token is the eos token, if not, append it to the completion_ids and completion_mask
            if task["completion_ids"][-1] != tokenizer.eos_token_id:
                task["completion_ids"].append(tokenizer.eos_token_id)
                task["completion_mask"].append(1)

            
            # Handling mismatch between completion_ids and completion_mask length; This is a hack, in theory
            # there should not be a mismatch between the two.
            if len(task["completion_ids"]) > len(task["completion_mask"]): # type: ignore
                task["completion_mask"].extend([1] * (len(task["completion_ids"]) - len(task["completion_mask"]))) # type: ignore
            if len(task["completion_mask"]) > len(task["completion_ids"]): # type: ignore
                task["completion_mask"] = task["completion_mask"][:len(task["completion_ids"])] # type: ignore

            # Parse the Agent Response to extract the Action
            # --- new extraction block --------------------------------------------
            match = self._ACTION_RE.search(agent_response_text)
            if match:                         # found an “Action:” JSON block
                payload = match.group(1).strip()
            else:                             # no Action header → treat whole text as a reply
                payload = agent_response_text.strip()
            # ----------------------------------------------------------------------
            
            parsed = None
            
            try:
                parsed: Dict[str, Any] = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise TypeError("JSON must decode to an object (dict)")
            except (json.JSONDecodeError, TypeError):
                # Valid JSON not found, so complete the task and give a penalty
                if training and not assist:
                    task["completed"] = True
                    task["task_completion_reward"] = 0
                    return j, task
                else:
                    parsed = {
                        "name": RESPOND_ACTION_NAME,
                        "arguments": {
                            "content": payload
                        }
                    }

            # Take the Action and observe the Environment
            if parsed is not None:
                try:
                    action = Action(
                                name=parsed.get("name"),
                                kwargs=parsed.get("arguments"),
                                )
                except:
                    if training and not assist:
                        task["completed"] = True
                        task["task_completion_reward"] = 0
                        return j, task
                    else:
                        action = Action(
                            name=RESPOND_ACTION_NAME,
                            kwargs={"content": payload}
                        )
                
                env_response = task["env"].step(action)
                obs = env_response.observation
            
                if action.name != RESPOND_ACTION_NAME:
                    obs = "API output: " + obs
            
                # Update the Trajectory and Multi-Turn History with the Environment Response
                task["trajectory"].append({"role": "user", "content": obs})
                task["multi_turn_history"].append({"role": "user", "content": obs})
                
                # Check if the task is completed
                task["num_turns"] = len(task["trajectory"])
                if task["num_turns"] > self.max_turns_per_task or env_response.done:
                    task["completed"] = True
                    task["task_completion_reward"] = env_response.reward
                    
                    # This is specific to TauBench where the conversation ends with the end user
                    # This is specific to LLAMA 3.1 Format
                    final_str = f"<|start_header_id|>user<|end_header_id|>\n\n{obs}<|eot_id|>"
                    final_completion_ids = tokenizer(final_str, add_special_tokens=False)["input_ids"]
                    task["completion_ids"].extend(final_completion_ids)
                    task["completion_mask"].extend([self.env_mask] * len(final_completion_ids))

                
            
            return j, task
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_task(*args),
                [(j, agent_responses[i]) for i, j in enumerate(live_indices)]
            ))

        for j, task in results:
            tasks[j] = task
        
        return tasks

                
    def generate(self,
                        tasks: List[Dict[str, Any]],
                        llm: LLM,
                        tokenizer: Any,
                        sampling_params: SamplingParams,
                        debug: bool = True,
                        training: bool = True,
                        **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
    
        custom_sp = sampling_params.clone()
        supervisor_task_1 = deepcopy(tasks[-1])
        supervisor_task_2 = deepcopy(tasks[-2])
        
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        all_completed = False
        assert len(tasks[0]['prompt']) == 2
        
        while not all_completed:
            tasks = self.step(tasks, llm, tokenizer, custom_sp, assist=False, training=training)
            all_completed = all(task["completed"] for task in tasks)

        if training and self.all_failed(tasks):
            print("All tasks failed, first attempt at using supervisor to complete the task {}".format(supervisor_task_1["task_idx"]))
            while not supervisor_task_1["completed"]:
                supervisor_task_1 = self.step([supervisor_task_1], llm, tokenizer, custom_sp, assist=True, training=training)[0]
            tasks[-1] = supervisor_task_1
            if supervisor_task_1["task_completion_reward"] > 0:
                print("Supervisor completed the task in the first attempt that the model wasn't able to complete for the following task: {}".format(supervisor_task_1["task_idx"]))

        if training and self.all_failed(tasks):
            print("All tasks failed, second attempt at using supervisor to complete the task {}".format(supervisor_task_2["task_idx"]))
            while not supervisor_task_2["completed"]:
                supervisor_task_2 = self.step([supervisor_task_2], llm, tokenizer, custom_sp, assist=True, training=training)[0]
            tasks[-2] = supervisor_task_2
            if supervisor_task_2["task_completion_reward"] > 0:
                print("Supervisor completed the task in the second attempt that the model wasn't able to complete for the following task: {}".format(supervisor_task_2["task_idx"]))

        if training and self.all_failed(tasks):
            print("Supervisor failed on both tasks for the following tasks: {}".format(supervisor_task_2["task_idx"]))
        
        completion_messages = [t["trajectory"][t["n_prompt_messages"]:] for t in tasks]
        completion_ids = [t["completion_ids"] for t in tasks]
        completion_mask = [t["completion_mask"] for t in tasks]
        
        output = {
            "ids": completion_ids,
            "trajectory_sans_prompt": completion_messages,
            "mask": completion_mask,
            "tasks": tasks,
        }

        return output


    def all_failed(self, tasks: List[Dict[str, Any]]) -> bool:
        for task in tasks:
            if task["task_completion_reward"] > 0:
                return False
        return True