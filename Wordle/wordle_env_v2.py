
import sys
sys.path.append('/home/ubuntu/')
from vllm import LLM, SamplingParams
import copy
from typing import List, Optional, Any, Dict
from Wordle.type import Word, Trajectory
from Wordle import TRAIN_WORDS, ALL_WORDS

import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from jinja2 import Environment, PackageLoader, select_autoescape
import json
from pathlib import Path
from datasets import Dataset
from nltk.corpus import words
import enchant
import re
from tqdm import tqdm
import hashlib



def hash_word(s: str, bits=16):
    hash_obj = hashlib.sha256(s.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16) % (2 ** bits)

def parser(text):
    reasoning = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    guess = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    if reasoning and guess:
        return reasoning.group(1).strip(), guess.group(1).strip()
    elif not reasoning and guess:
        return None, guess.group(1).strip()
    elif reasoning and not guess:
        return reasoning.group(1).strip(), None
    else:
        return None, None
    

class WordleRubric:
    def __init__(self, **kwargs):
        self.reward_funcs = [
            self.game_completion_reward,
            self.format_error_penalty
            ]
        
        self.reward_weights = [5.0,3.0]
        
        self.penalty_score = -1.0
        self.reward_score = 1.0
    
    def game_completion_reward(self, trajectories: List[Trajectory]) -> List[float]:
        """
        Reward Function for Game Completion
        """
        rewards = []
        for trajectory in trajectories:
            reward = 0.0
            if trajectory.solved:
                reward = self.reward_score
            rewards.append(reward)
        return rewards

    def format_error_penalty(self, trajectories: List[Trajectory]) -> List[float]:
        """
        Penalize trajectories that do not contain the <think> and <answer> tags
        """
        rewards = []
        for trajectory in trajectories:
            if '<think>' not in trajectory.messages[-1]['content'] or '</think>' not in trajectory.messages[-1]['content']:
                reward = self.penalty_score
            elif '<answer>' not in trajectory.messages[-1]['content'] or '</answer>' not in trajectory.messages[-1]['content']:
                reward = self.penalty_score
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards

    def get_reward_funcs(self):
        return self.reward_funcs
    
    def get_reward_weights(self):
        return self.reward_weights


class WordleEnv:
    def __init__(self):
        self.max_turns = 6
        self.games_won = 0
        self.games_played = 0
        self.env_mask = 0
        self.total_cost = 0.0
        self.lock = threading.Lock()  # Lock for synchronized writing
        self.jinja_env = Environment(
            loader=PackageLoader('Wordle'),
            autoescape=select_autoescape()
        )
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
            "include_stop_str_in_output": False,
            "truncate_prompt_tokens": 8000,
        }   
        broker = enchant.Broker()
        self.d = broker.request_dict("en_US")
        self.local_dictionary = list(json.load(open('Wordle/data/English Words Dictionary.json')).keys())
        self.nltk_words = words.words()
        self.system_prompt_template = self.jinja_env.get_template('system_prompt.txt')
        self.messages_template = [
                                    {'role': 'system', 'content': self.system_prompt_template.render()},
                                    {'role': 'user', 'content': 'Your first guess is?'}
                                ]
        
    
    def get_feedback(self, guess: str, word: str):
        """
        trajectory: list of tuples (word, result)
        word: string
        guess: string
        """
        # if not self.d.check(guess) and guess not in self.nltk_words:
        if guess not in self.nltk_words and guess not in self.local_dictionary and not self.d.check(guess):
            return "Invalid word, not a valid English word"
        
        if len(guess) != 5:
            return "Invalid word, not a 5 letter word"
        
        response = ""
        response_text = ""
        for i, alphabet in enumerate(guess):
            if word[i] == alphabet:
                response += "G"
                response_text += f'{alphabet} is in the word and in the correct position.\n'
            elif alphabet in word:
                response += "Y"
                response_text += f'{alphabet} is in the word but in the wrong position.\n'
            else:
                response += "B"
                response_text += f'{alphabet} is not in the word.\n'
        return f"{guess} -> {response}"
    
    def get_dataset(self, dataset: str = 'train', number_of_games: Optional[int] = None):
        if dataset == 'train':
            word_dicts = random.sample([word.model_dump() for word in TRAIN_WORDS], number_of_games)
            return Dataset.from_list(word_dicts)
        word_dicts = random.sample([word.model_dump() for word in ALL_WORDS], number_of_games)
        return Dataset.from_list(word_dicts)

    def generate_response(self, trajectories: List[Trajectory], llm: LLM, sampling_params: SamplingParams):
        messages = [trajectory.messages for trajectory in trajectories]
        agent_responses = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        return agent_responses

    def play(self, tokenizer: Any, trajectories: List[Trajectory], llm: LLM, sampling_params: SamplingParams, training: bool = True):
        # Force load the nltk words
        live_indices = [i for i, trajectory in enumerate(trajectories) if not trajectory.game_completed]
        if len(live_indices) == 0:
            return trajectories
        
        agent_responses = self.generate_response(trajectories, llm, sampling_params)

        def update_task(j, agent_response):
            # Isolate the Agent Response
            
            agent_response_text = agent_response.outputs[0].text
            trajectory = copy.deepcopy(trajectories[j])
            trajectory.num_turns += 1

            if len(trajectory.prompt_ids) == 0:
                trajectory.prompt_ids = list(agent_response.prompt_token_ids)
            trajectory.messages.append({"role": "assistant", "content": agent_response_text})
            
            
            # Update Completion IDs and Completion Masks
            total_prev_len = len(trajectory["prompt_ids"]) + len(trajectory["completion_ids"])
            env_response_len  = len(list(agent_response.prompt_token_ids)) - total_prev_len # type: ignore
            new_completion_len = len(agent_response.outputs[0].token_ids)
            
            trajectory.completion_mask.extend([self.env_mask] * env_response_len)
            trajectory.completion_mask.extend([1] * new_completion_len)

            trajectory.completion_ids = list(agent_response.prompt_token_ids) # type: ignore
            trajectory.completion_ids.extend(list(agent_response.outputs[0].token_ids))
            trajectory.completion_ids = trajectory.completion_ids[len(trajectory.prompt_ids):]

            # Check to see if the last token is the eos token, if not, append it to the completion_ids and completion_mask
            if trajectory.completion_ids[-1] != tokenizer.eos_token_id:
                trajectory.completion_ids.append(tokenizer.eos_token_id)
                trajectory.completion_mask.append(1)

            
            # Handling mismatch between completion_ids and completion_mask length; This is a hack, in theory
            # there should not be a mismatch between the two.
            if len(task["completion_ids"]) > len(task["completion_mask"]): # type: ignore
                task["completion_mask"].extend([1] * (len(task["completion_ids"]) - len(task["completion_mask"]))) # type: ignore
            if len(task["completion_mask"]) > len(task["completion_ids"]): # type: ignore
                task["completion_mask"] = task["completion_mask"][:len(task["completion_ids"])] # type: ignore

            trajectory.messages.append({'role': 'assistant', 'content': agent_response_text})
            # Parse the Agent Response to extract the Guess
            # --- new extraction block --------------------------------------------
            _, guess = parser(agent_response_text)
            feedback = None
            if guess:
                feedback = self.get_feedback(guess, trajectory.word)
                trajectory.messages.append({'role': 'user', 'content': feedback})
            else:
                trajectory.messages.append({'role': 'user', 'content': f'<think>Invalid format, stick to the <think>, </think> and <answer>, </answer> tags provided in the system prompt'})
            # ----------------------------------------------------------------------
            if feedback.split('->')[-1].strip() == 'GGGGG':
                trajectory.solved = True
                trajectory.messages[-1]['content'] = f'Success! You found the word {trajectory.word} in {trajectory.num_turns} turns.'
                trajectory.game_completed = True
                
            if trajectory.num_turns == self.max_turns:
                trajectory.game_completed = True
                    
            # This is specific to Qwen2.5 Format
            if trajectory.game_completed:
                final_str = f"<|im_start|>user\nSuccess! You found the word {trajectory.word} in {trajectory.num_turns} turns.<|im_end|>\n"
                final_completion_ids = tokenizer(final_str, return_tensors=None, add_special_tokens=False)["input_ids"]
                trajectory.completion_ids.extend(final_completion_ids)
                trajectory.completion_mask.extend([self.env_mask] * len(final_completion_ids))
                return j, trajectory
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_task(*args),
                [(j, agent_responses[i]) for i, j in enumerate(live_indices)]
            ))

        for j, trajectory in results:
            trajectories[j] = trajectory
        
        return trajectories
    
    def solve(self, trajectories: List[Trajectory], llm: LLM, sampling_params: SamplingParams, training: bool = True):
        _ = words.words()
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        
        all_games_completed = False
        while not all_games_completed:
            trajectories = self.play(trajectories, llm, custom_sp, training)
            all_games_completed = all(trajectory.game_completed for trajectory in trajectories)

        completion_messages = [t["trajectory"][2:] for t in trajectories]
        completion_ids = [t["completion_ids"] for t in trajectories]
        completion_mask = [t["completion_mask"] for t in trajectories]
        
        output = {
            "ids": completion_ids,
            "trajectory_sans_prompt": completion_messages,
            "mask": completion_mask,
            "trajectories": trajectories,
        }

        return output