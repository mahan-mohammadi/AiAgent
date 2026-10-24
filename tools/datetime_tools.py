

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def get_current_time(timezone: str) -> str:
    """Get the current date and time in an IANA timezone.
    Example timezone: Asia/Tehran.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z")
    except (ZoneInfoNotFoundError, ValueError):
        return f"Invalid timezone: {timezone}"


@tool
def get_relative_date(days_from_today: int) -> str:
    """Get the weekday and date for today or a relative day.
    Today = 0, tomorrow = 1, yesterday = -1.
    """
    today = datetime.now().astimezone().date()
    target = today + timedelta(days=days_from_today)
    return target.strftime("%A, %Y-%m-%d")


@tool
def time_until(target_time: str, timezone: str) -> str:
    """Calculate the time until the next occurrence of a clock time.
    target_time must use HH:MM format, e.g. 11:30.
    Use an IANA timezone, e.g. Asia/Tehran.
    """
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        parsed_time = datetime.strptime(target_time, "%H:%M").time()
        target = datetime.combine(now.date(), parsed_time, tzinfo=tz)

        if target <= now:
            target += timedelta(days=1)

        seconds = int((target - now).total_seconds())
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return (
            f"Time until {target_time} in {timezone}: "
            f"{hours} hours, {minutes} minutes, {seconds} seconds."
        )
    except (ValueError, ZoneInfoNotFoundError):
        return "Invalid time or timezone. Use HH:MM and a valid IANA timezone."
