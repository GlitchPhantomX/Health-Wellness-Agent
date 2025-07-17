from typing import Optional
from agents import function_tool
from datetime import datetime


@function_tool
def check_for_checkin_reminder(checkin_day: str = "Friday") -> Optional[str]:
    """
    Check if tomorrow is the user's check-in day and return a reminder message.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    today_index = datetime.now().weekday() 
    tomorrow_index = (today_index + 1) % 7
    tomorrow_day = weekdays[tomorrow_index]

    if tomorrow_day.lower() == checkin_day.lower():
        return f"📅 Reminder: Kal {checkin_day} hai. Tumhara progress check scheduled hai. Apna update dena mat bhoolna!"
    else:
        return None  