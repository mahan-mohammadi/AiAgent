

from langchain_core.tools import tool


UNITS = {
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "mi": 1609.344,
        "ft": 0.3048,
        "in": 0.0254,
    },
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 0.000001,
        "lb": 0.45359237,
        "oz": 0.028349523125,
    },
}


@tool
def convert_units(
    value: float,
    category: str,
    from_unit: str,
    to_unit: str,
) -> str:
    """Convert length, mass, or temperature units.

    Length: m, km, cm, mm, mi, ft, in.
    Mass: kg, g, mg, lb, oz.
    Temperature: C, F, K.
    """
    if category == "temperature":
        unit_map = {"C": "C", "F": "F", "K": "K"}
        if from_unit not in unit_map or to_unit not in unit_map:
            raise ValueError("Temperature units must be C, F, or K.")

        if from_unit == "F":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "K":
            celsius = value - 273.15
        else:
            celsius = value

        if to_unit == "F":
            result = celsius * 9 / 5 + 32
        elif to_unit == "K":
            result = celsius + 273.15
        else:
            result = celsius

        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    if category not in UNITS:
        raise ValueError("Category must be length, mass, or temperature.")

    units = UNITS[category]
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Unsupported {category} units.")

    result = value * units[from_unit] / units[to_unit]
    return f"{value} {from_unit} = {result:.4f} {to_unit}"
