from agents import function_tool, Agent, Runner, RunContextWrapper
from pydantic import BaseModel
from context.session_context import UserSessionContext
from config import run_config


class WorkoutPlan(BaseModel):
    monday: str
    tuesday: str
    wednesday: str
    thursday: str
    friday: str
    saturday: str
    sunday: str


workout_agent = Agent(
    name="Workout Recommender Agent",
    instructions=(
        "Create a 7-day workout plan for the user based on their goal and experience level. "
        "Return your response in EXACTLY this JSON format:\n"
        '{\n'
        '  "monday": "Workout type",\n'
        '  "tuesday": "Workout type",\n'
        '  "wednesday": "Workout type",\n'
        '  "thursday": "Workout type",\n'
        '  "friday": "Workout type",\n'
        '  "saturday": "Workout type",\n'
        '  "sunday": "Workout type"\n'
        '}\n'
        "Only return the JSON — no extra text or explanation."
    ),
    output_type=WorkoutPlan,
)


@function_tool
async def workout_recommend(ctx: RunContextWrapper[UserSessionContext], experience_level: str) -> str:
    """
    Generates a 7-day workout plan based on the user's goal and experience level.
    """

    valid_levels = ["beginner", "intermediate", "advanced"]
    if experience_level.lower() not in valid_levels:
        return f"❌ Invalid experience level. Please choose one of: {', '.join(valid_levels)}"

    user_goal = ctx.context.goal or {"goal_type": "general_fitness"}
    prompt = f"User goal: {user_goal}. Experience level: {experience_level}."

    try:
        result = await Runner.run(
            starting_agent=workout_agent,
            input=prompt,
            context=ctx.context,
            run_config=run_config
        )

        ctx.context.workout_plan = result.final_output

        return "✅ Your 7-day personalized workout plan has been created! 💪 Let me know if you'd like to adjust it."

    except Exception as e:
        return f"❌ Something went wrong while generating your workout plan. Error: {str(e)}"