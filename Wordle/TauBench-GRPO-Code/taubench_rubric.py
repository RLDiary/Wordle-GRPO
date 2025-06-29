
from verifiers.rubrics import Rubric
from typing import List, Dict, Any, Tuple, Union
from tau_bench.envs.tool import Tool
from tau_bench.envs.retail.tools import ALL_TOOLS
from tau_bench.types import Action
import json
import re
from tau_bench.types import RESPOND_ACTION_NAME

class TauBenchRubric(Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_funcs = [
            self.task_completion_reward,
            # self.repetitive_function_call_penalty,
            self.format_error_penalty
            ]
        
        self.reward_weights = [
            5.0,
            # 1.0,
            3.0
            ]

        self.tools_map = {t.get_info()["function"]["name"].lower(): t for t in ALL_TOOLS}
        self.tools_map[RESPOND_ACTION_NAME.lower()] = None
        
        self.penalty_score = -1.0
        self.reward_score = 1.0
        self.ParameterErrorPhrase = "invoke() got an unexpected keyword argument"
        self.FunctionNameErrorPhrase = "API output: Unknown action"
        self._ACTION_RE = re.compile(r"(?is)Action:\s*(\{.*?\}\})")
    
    def format_error_penalty(self, tasks: List[Dict[str, Any]]) -> List[float]:
        """
        Penalize tasks that contain format errors.
        """
        rewards = []
        for task in tasks:
            reward = 0.0
            for message in task.get("trajectory", []):
                role = message.get("role")
                content = (message.get("content") or "").strip()

                if (role == "user" and (self.ParameterErrorPhrase.lower() in content.lower() or self.FunctionNameErrorPhrase.lower() in content.lower())):
                    reward = self.penalty_score
                    break
                if role == 'assistant':
                    match = self._ACTION_RE.search(content)
                    if not match:
                        reward = self.penalty_score
                        break
                    try:
                        action_json = json.loads(match.group(1))
                        if "name" not in action_json or "arguments" not in action_json:
                            reward = self.penalty_score
                            break
                        tool_name = action_json.get("name", '').lower()
                        if tool_name not in self.tools_map:
                            reward = self.penalty_score
                            break
                        
                    except json.JSONDecodeError:
                        reward = self.penalty_score
                        break      
            rewards.append(reward)
        return rewards
    
    def task_completion_reward(self, tasks: List[Dict[str, Any]]) -> List[float]:
        """
        Reward Function for Task Completion
        """
        rewards = []
        for task in tasks:
            reward = 0.0
            if task["completed"]:
                if task["task_completion_reward"] > 0.0:
                    reward = self.reward_score
            rewards.append(reward)
        return rewards

    def repetitive_function_call_penalty(self, tasks: List[Dict[str, Any]]) -> List[float]:
        """
        Return one reward/penalty for each task.
        A task is penalised (duplicate_penalty) if it contains the same
        function call (same .name and identical kwargs) more than once;
        otherwise it receives 0.0.
        """
        rewards: List[float] = []

        def make_hashable(obj: Any) -> Union[Tuple, Any]:
            """Recursively convert dicts/lists to tuples so they become hashable."""
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(x) for x in obj)
            else:
                return obj

        for task in tasks:
            seen = set()
            reward = 0.0
            
            for action in task['env'].actions:
                # Convert the arguments to a frozenset of key-value tuples (sorted) to ensure hashability
                key = (action.name, make_hashable(action.kwargs))
                if key in seen:
                    reward = self.penalty_score
                    break
                else:
                    seen.add(key)

            rewards.append(reward)
        return rewards


