from agents import function_tool, Agent, Runner, RunContextWrapper
from typing import List
from pydantic import BaseModel
from context.session_context import UserSessionContext
from config import run_config


class MealPlanner(BaseModel):
    days: List[str]



@function_tool
async def meal_planner(ctx: RunContextWrapper[UserSessionContext], diet_preference: str) -> MealPlanner:
    """
    Generates a 7-day meal plan according to user diet_preference.
    """
    
    if diet_preference.lower() not in ["vegetarian", "non-vegetarian", "keto", "vegan"]:
        raise ValueError("Please choose a valid diet preference: vegetarian, non-vegetarian, keto, or vegan.")


    instructions = (
        f"Create a 7-day {diet_preference.lower()} meal plan in list format (one string per day). "
        f"Make sure to follow the {diet_preference.lower()} diet strictly."
        f"Respond ONLY with a list of 7 meal suggestio ns."
    )

    dynamic_agent = Agent(
        name="Meal Planner Agent",
        instructions= instructions,
        output_type=MealPlanner
    )

    result = await Runner.run(
        dynamic_agent,
        input=diet_preference,
        context=ctx.context,
        run_config=run_config
    )

    ctx.context.meal_plan = result.final_output

    return result.final_output