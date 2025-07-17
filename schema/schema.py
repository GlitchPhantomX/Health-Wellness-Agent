from typing import Literal
from pydantic import BaseModel

class ParsedGoal(BaseModel):
    goal_type: Literal['weight_loss', 'weight_gain', 'muscle_gain', 'general_fitness']
    quantity: float
    metric: str
    duration: str