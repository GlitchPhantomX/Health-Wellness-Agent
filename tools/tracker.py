from agents import function_tool, Agent, Runner, RunContextWrapper
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
from context.session_context import UserSessionContext
from config import run_config

class ProgressUpdate(BaseModel):
    update_type: Literal["weight_loss", "diet_follow", "workout_done", "missed_workout", "general"]
    message: str
    timestamp: str


progress_tracker_agent = Agent(
    name="Progress Tracker Agent",
    instructions=(
        "The user will give updates about their progress (e.g., 'I lost 1kg', 'I missed my workout'). "
        "Classify the update into one of the types: weight_loss, diet_follow, workout_done, missed_workout, general. "
        "Return the result as JSON like:\n"
        '{\n  "update_type": "workout_done",\n  "message": "Completed full workout today",\n  "timestamp": "YYYY-MM-DD HH:MM"\n}'
    ),
    output_type=ProgressUpdate
)


@function_tool
async def track_progress(ctx: RunContextWrapper[UserSessionContext], user_update: str) -> str:
    """
    Accepts a user update and classifies + stores it. Returns a user-friendly confirmation.
    """

    prompt = f"User update: {user_update}"

    try:
        result = await Runner.run(
            starting_agent=progress_tracker_agent,
            input=prompt,
            context=ctx.context,
            run_config=run_config
        )

        result.final_output.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        ctx.context.progress_logs.append(result.final_output.dict())

        return f"✅ Got it! Your progress has been logged as **{result.final_output.update_type}**. Keep it up! 💪"

    except Exception as e:
        return f"❌ Sorry, I couldn't record that update. Error: {str(e)}"