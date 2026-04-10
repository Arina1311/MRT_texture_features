def map_classname_to_numeric(classname: str) -> int:
    return 0 if classname.startswith("AS") else 1


def map_numeric_to_classname(numeric: int) -> str:
    return "AS" if numeric == 0 else "CO"
