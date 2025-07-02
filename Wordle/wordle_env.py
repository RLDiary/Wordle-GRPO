import os
import sys
sys.path.append('/home/ubuntu/')
import copy
from typing import List, Optional, Any
from Wordle.type import Word, Trajectory
from Wordle.Data.train import TRAIN_WORDS
from Wordle.Data.all_words import ALL_WORDS
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
import litellm
from litellm import completion, batch_completion
# litellm._turn_on_debug()
import hashlib
from dotenv import load_dotenv
import requests

# Replace with your vLLM server URL
VLLM_SERVER_URL = "http://localhost:8000"

load_dotenv()


def get_model_id():
    try:
        response = requests.get(f"{VLLM_SERVER_URL}/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            print("No models are currently loaded.")
        else:
            print("Loaded models:")
            for model in models:
                print(f"- ID: {model.get('id')} | Owned by: {model.get('owned_by', 'unknown')}")
            return models[0].get('id')
    except requests.RequestException as e:
        print(f"Error querying vLLM server: {e}")

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

class GameState:
    def __init__(self, W: Word, messages: List[dict], model_config: dict = None):
        
        if model_config is None:
            self.model = 'o3'
            self.custom_llm_provider = 'openai'
            self.api_base = "https://api.openai.com/v1"
        else:
            self.model = model_config['model']
            self.custom_llm_provider = model_config['custom_llm_provider']
            self.api_base = model_config['api_base']
        self.W = Word(**W)
        self.trajectory = Trajectory(
                                    word=self.W.word,
                                    word_hash=self.W.hash,
                                    messages=messages,
                                    completed_by=self.model,
                                    completion_cost=0.0,
                                    solved=False
                                )
        self.max_turns = 6
        self.current_turn = 0
        self.current_guess = None

        broker = enchant.Broker()
        self.d = broker.request_dict("en_US")
        self.local_dictionary = list(json.load(open('/home/ubuntu/Wordle/English Words Dictionary.json')).keys())
        
        
        self.nltk_words = words.words()

    def get_feedback(self, guess: str):
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
            if self.W.word[i] == alphabet:
                response += "G"
                response_text += f'{alphabet} is in the word and in the correct position.\n'
            elif alphabet in self.W.word:
                response += "Y"
                response_text += f'{alphabet} is in the word but in the wrong position.\n'
            else:
                response += "B"
                response_text += f'{alphabet} is not in the word.\n'
        return f"{guess} -> {response}"
    
    def generate_response(self):
        if self.custom_llm_provider == 'openai':
            response = completion(model=self.model, custom_llm_provider=self.custom_llm_provider, messages=self.trajectory.messages)
        elif self.custom_llm_provider == 'vllm':
            response = completion(model=self.model, api_base=self.api_base, messages=self.trajectory.messages)
        else:
            raise ValueError(f"Invalid custom llm provider")

        if response._hidden_params["response_cost"] is not None:
            self.trajectory.completion_cost += response._hidden_params["response_cost"]
        
        return response
    
    def step(self):
        response = self.generate_response()
        
        answer = response.choices[0].message.content
        # Get the guess from the answer and get feedback
        reasoning, guess = parser(answer)

        if 'reasoning_content' in vars(response.choices[0].message):
            reasoning = response.choices[0].message.reasoning_content
            self.trajectory.messages.append({'role': 'assistant', 'content': f'<think>{reasoning}</think><answer>{guess}</answer>'})
        else:
            self.trajectory.messages.append({'role': 'assistant', 'content': answer})
        
        feedback = None
        if guess:
            feedback = self.get_feedback(guess)
            self.trajectory.messages.append({'role': 'user', 'content': feedback})
        else:
            self.trajectory.messages.append({'role': 'user', 'content': f'<think>Invalid format, stick to the <think>, </think> and <answer>, </answer> tags provided in the system prompt'})
        
        # If the feedback is 'GGGGG', break
        print(f'{feedback}')
        if feedback.split('->')[-1].strip() == 'GGGGG':
            print(f'Word {self.W.word} found in {self.current_turn+1} turns')
            self.trajectory.solved = True
            self.trajectory.messages[-1]['content'] = f'Success! You found the word {self.W.word} in {self.current_turn+1} turns.'
            return True
        
        return False
    
    def solve(self):
        while self.current_turn < self.max_turns and not self.trajectory.solved:
            self.step()
            self.current_turn += 1
        return self.trajectory

class WordleEnv:
    def __init__(self, dataset: Optional[str] = None, number_of_games: Optional[int] = None, model_config: dict = None):
        self.num_workers = 2
        self.number_of_games = len(self.dataset) if number_of_games is None else number_of_games
        
        if dataset is not None:
            self.dataset = self.get_dataset(dataset, number_of_games)
        else:
            self.dataset = self.get_dataset(number_of_games=number_of_games)
        self.model_config = model_config
        self.games_won = 0
        
        self.total_cost = 0.0
        self.lock = threading.Lock()  # Lock for synchronized writing
        self.jinja_env = Environment(
            loader=PackageLoader('Wordle'),
            autoescape=select_autoescape()
        )
        

        self.trajectory_output_dir = Path('/home/ubuntu/Wordle/Results')
        self.filename = 'trajectories.json'
        self.trajectory_output_path = self.trajectory_output_dir / self.filename
        self.reset()
        
        # Ensure the output directory exists
        self.trajectory_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset(self, dataset: str = 'train', number_of_games: Optional[int] = None):
        if dataset == 'train':
            word_dicts = random.sample([word.model_dump() for word in TRAIN_WORDS], number_of_games)
            return Dataset.from_list(word_dicts)
        word_dicts = random.sample([word.model_dump() for word in ALL_WORDS], number_of_games)
        return Dataset.from_list(word_dicts)
    
    def reset(self):
        self.system_prompt_template = self.jinja_env.get_template('system_prompt.txt')
        self.messages_template = [
                                    {'role': 'system', 'content': self.system_prompt_template.render()},
                                    {'role': 'user', 'content': 'Your first guess is?'}
                                ]

    def play(self):
        # Force load the nltk words
        _ = words.words()
        def play_game(i):
            print(f'Playing game {i}')
            messages = copy.deepcopy(self.messages_template)
            G = GameState(W=copy.deepcopy(self.dataset[i]), messages=messages, model_config=self.model_config)
            G.solve()
            print(f'Game {i} solved')
            
            
            # Append to file safely
            with self.lock:
                self.total_cost += G.trajectory.completion_cost
                if G.trajectory.solved:
                    self.games_won += 1
                # 1️⃣ Load whatever is already on disk (may be empty/invalid)
                trajectories: list[dict] = []
                if self.trajectory_output_path.exists() and self.trajectory_output_path.stat().st_size:
                    try:
                        with self.trajectory_output_path.open("r", encoding="utf-8") as f:
                            trajectories = json.load(f)          # expect list-or-dict; default to []
                            if not isinstance(trajectories, list):
                                trajectories = [trajectories]
                    except json.JSONDecodeError:
                        # file existed but had garbage / partial write – start fresh
                        trajectories = []

                # 2️⃣ Add the latest trajectory
                trajectories.append(G.trajectory.model_dump())

                # 3️⃣ Persist the updated list (truncate + write)
                with self.trajectory_output_path.open("w", encoding="utf-8") as f:
                    json.dump(trajectories, f, indent=2)
                                

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(play_game, i): i for i in range(self.number_of_games)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in game {i}: {e}")
        
        print(f'Total games played: {self.number_of_games}')
        print(f'Games won: {self.games_won}')
        print(f'Total cost: {self.total_cost}')
        
if __name__ == '__main__':
    model_config = {
        'model': 'hosted_vllm/'+get_model_id(),
        'custom_llm_provider': 'vllm',
        'api_base': "http://localhost:8000/v1"
    }
    env = WordleEnv(dataset='all', number_of_games=100, model_config=model_config)

    env.play()