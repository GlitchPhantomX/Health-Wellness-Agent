from agents.tool import function_tool
from agents import RunContextWrapper, Runner, Agent
from pydantic import BaseModel
from context.session_context import UserSessionContext
from config import run_config
from guardrails.guardrails import goal_output_guardrail
from schema.schema import ParsedGoal


goal_parser_agent = Agent(
    name="Goal Parser Agent",
    instructions="""
You analyze a user's health-related goal and return a clean, structured JSON object.
Only respond with the valid JSON format like:
{
  "goal_type": "weight_loss",
  "quantity": 5.0,
  "metric": "kg",
  "duration": "2 months"
}
""",
    output_type=ParsedGoal,
    output_guardrails=[goal_output_guardrail]
)


@function_tool
async def analyze_goal(ctx: RunContextWrapper[UserSessionContext], goal_description: str) -> str:
    """
    Uses LLM to analyze and convert a goal like 'I want to lose 5kg in 2 months'
    into a structured ParsedGoal object, but returns user-friendly text.
    """

    result = await Runner.run(
        starting_agent=goal_parser_agent,
        input=goal_description,
        context=ctx.context,
        run_config=run_config
    )

    parsed_goal: ParsedGoal = result.final_output

    goal_type_map = {
        "weight_loss": "lose",
        "weight_gain": "gain weight",
        "muscle_gain": "build muscle",
        "general_fitness": "improve fitness"
    }

    summary = (
        f"✅ Goal confirmed: You want to {goal_type_map.get(parsed_goal.goal_type, 'work on your goal')} "
        f"{parsed_goal.quantity} {parsed_goal.metric} in {parsed_goal.duration}."
    )

    return summary