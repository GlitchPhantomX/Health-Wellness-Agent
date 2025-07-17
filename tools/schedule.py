from typing import TypedDict
from agents import function_tool

class CheckinSchedule(TypedDict):
    frequency: str 
    day: str        
    message: str    

@function_tool
def schedule_checkin(preferred_day: str = "Friday") -> CheckinSchedule:
    """
    Schedule a weekly progress check-in reminder on user's preferred day
    """
    return {
        "frequency": "weekly",
        "day": preferred_day,
        "message": f"Don't forget to check in every {preferred_day}! Share your progress with me."
    }