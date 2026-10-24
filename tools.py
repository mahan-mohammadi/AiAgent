

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import ast
import operator
import re
import math

from langchain_core.tools import tool


# ---------- Calculator ----------

@tool
def calculator(numbers: list[float], operation: str) -> float:
    """Perform arithmetic on a list of numbers.

    Operations: add, subtract, multiply, divide, average,
    min, max, power, modulo.
    Subtraction and division work from left to right.
    Power uses exactly two numbers: base and exponent.
    """

    if not numbers:
        raise ValueError("Provide at least one number")

    if operation == "add":
        return sum(numbers)

    if operation == "subtract":
        if len(numbers) < 2:
            raise ValueError("Subtraction needs at least two numbers")
        result = numbers[0]
        for n in numbers[1:]:
            result -= n
        return result

    if operation == "multiply":
        if len(numbers) < 2:
            raise ValueError("Multiplication needs at least two numbers")
        return math.prod(numbers)

    if operation == "divide":
        if len(numbers) < 2:
            raise ValueError("Division needs at least two numbers")
        result = numbers[0]
        for n in numbers[1:]:
            if n == 0:
                raise ValueError("Cannot divide by zero")
            result /= n
        return result

    if operation == "average":
        return sum(numbers) / len(numbers)

    if operation == "min":
        return min(numbers)

    if operation == "max":
        return max(numbers)

    if operation == "power":
        if len(numbers) != 2:
            raise ValueError("Power needs a base and exponent")
        return numbers[0] ** numbers[1]

    if operation == "modulo":
        if len(numbers) != 2:
            raise ValueError("Modulo needs exactly two numbers")
        if numbers[1] == 0:
            raise ValueError("Cannot use modulo with zero")
        return numbers[0] % numbers[1]

    raise ValueError(f"Unknown operation: {operation}")


# ---------- Date and time ----------

@tool
def get_current_time(timezone: str) -> str:
    """Get the current date and time in an IANA timezone.
    Examples: Europe/Berlin, Asia/Tokyo, Asia/Baku.
    Never translate timezone identifiers.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except (ZoneInfoNotFoundError, ValueError):
        return f"Unknown timezone: {timezone}"


@tool
def get_relative_date(days_from_today: int) -> str:
    """Return the actual weekday and date for a relative day.

    Use this tool whenever the user asks what day or date it is
    today, tomorrow, yesterday, or a certain number of days from
    today. Examples:
    - tomorrow: days_from_today=1
    - yesterday: days_from_today=-1
    - today: days_from_today=0

    Return the tool's result to the user; do not merely describe
    how this function could be used.
    """
    today = datetime.now().astimezone().date()
    target = today + timedelta(days=days_from_today)
    return target.strftime("%A, %Y-%m-%d")



@tool
def add_days(date: str, days: int) -> str:
    """Add or subtract a number of days from a YYYY-MM-DD date."""
    result = datetime.strptime(date, "%Y-%m-%d").date()
    return (result + timedelta(days=days)).isoformat()


# ---------- Text utilities ----------

@tool
def count_words(text: str) -> dict:
    """Count words, characters, and lines in text."""
    return {
        "words": len(text.split()),
        "characters": len(text),
        "lines": len(text.splitlines()) if text else 0,
    }


@tool
def reverse_text(text: str) -> str:
    """Reverse a string character by character."""
    return text[::-1]


@tool
def find_text(text: str, query: str) -> dict:
    """Find case-insensitive occurrences of a phrase in text."""
    return {
        "query": query,
        "occurrences": len(re.findall(re.escape(query), text, re.I)),
        "found": query.casefold() in text.casefold(),
    }


# ---------- Unit conversion ----------

@tool
def convert_units(value: float, category: str, from_unit: str, to_unit: str) -> float:
    """Convert units within a supported category.

    Categories and units:
    length: m, km, cm, mm, mi, ft, in
    mass: kg, g, mg, lb, oz
    temperature: C, F, K
    """

    factors = {
        "length": {
            "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001,
            "mi": 1609.344, "ft": 0.3048, "in": 0.0254,
        },
        "mass": {
            "kg": 1, "g": 0.001, "mg": 0.000001,
            "lb": 0.45359237, "oz": 0.028349523125,
        },
    }

    if category in factors:
        units = factors[category]
        if from_unit not in units or to_unit not in units:
            raise ValueError("Unsupported unit for this category")
        return value * units[from_unit] / units[to_unit]

    if category == "temperature":
        def to_celsius(v, unit):
            if unit == "C":
                return v
            if unit == "F":
                return (v - 32) * 5 / 9
            if unit == "K":
                return v - 273.15
            raise ValueError("Temperature units must be C, F, or K")

        celsius = to_celsius(value, from_unit)
        if to_unit == "C":
            return celsius
        if to_unit == "F":
            return celsius * 9 / 5 + 32
        if to_unit == "K":
            return celsius + 273.15
        raise ValueError("Temperature units must be C, F, or K")

    raise ValueError("Category must be length, mass, or temperature")
