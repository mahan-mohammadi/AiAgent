import math

from langchain_core.tools import tool


@tool
def calculator(numbers: list[float], operation: str) -> float:
    """Calculate using a list of plain numeric values.

    Arguments:
    - numbers: A list of numbers, such as [555, 10].
    Each item MUST be a number, never a dictionary or object.
    - operation: One of add, subtract, multiply, divide,
    average, min, max, power.

    For multiplication, include EVERY operand in numbers.
    Example: 555 multiplied by 10 means
    numbers=[555, 10], operation="multiply".

    Subtraction and division work left to right.
    Power requires exactly two numbers: base and exponent.
    """
    if not numbers:

        raise ValueError("Provide at least one number.")

    if operation == "add":
        return sum(numbers)

    if operation == "subtract":
        if len(numbers) < 2:
            raise ValueError("Subtraction requires at least two numbers.")
        result = numbers[0]
        for number in numbers[1:]:
            result -= number
        return result

    if operation == "multiply":
        return math.prod(numbers)

    if operation == "divide":
        if len(numbers) < 2:
            raise ValueError("Division requires at least two numbers.")
        result = numbers[0]
        for number in numbers[1:]:
            if number == 0:
                raise ValueError("Cannot divide by zero.")
            result /= number
        return result

    if operation == "average":
        return sum(numbers) / len(numbers)

    if operation == "min":
        return min(numbers)

    if operation == "max":
        return max(numbers)

    if operation == "power":
        if len(numbers) != 2:
            raise ValueError("Power requires exactly two numbers.")
        return numbers[0] ** numbers[1]

    raise ValueError(f"Unknown operation: {operation}")
