from pydantic import BaseModel
from typing import Optional

class Trajectory(BaseModel):
    word: str
    word_hash: int
    messages: list[dict]
    prompt_ids: Optional[list[int]] = []
    completion_ids: Optional[list[int]] = []
    completion_mask: Optional[list[int]] = []
    completed_by: Optional[str] = ''
    completion_cost: Optional[float] = 0.0
    solved: Optional[bool] = False
    game_completed: Optional[bool] = False
    num_turns: Optional[int] = 0
    guesses: Optional[list[str]] = []
    feedback: Optional[list[str]] = []

class Word(BaseModel):
    df_index: Optional[int] = None
    hash: int
    word: str
    occurrence: Optional[float] = 0.0
    linked_trajectories: Optional[list[int]] = []
    games_won: Optional[int] = 0
    games_total: Optional[int] = 0