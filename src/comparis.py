import os
import json
import math
from typing import Any, List, Tuple

# Папки для сравнения
FOLDER1 = "/Users/arinaperzu/Desktop/MRT2/features"
FOLDER2 = "/Users/arinaperzu/Desktop/MRT2/src/features"

# Допуск для сравнения чисел с плавающей точкой (0 — строгое равенство)
EPSILON = 0.0

# Цвета ANSI (если не хочешь цвета — поставь USE_COLOR = False)
USE_COLOR = True
RED = "\033[31m" if USE_COLOR else ""
GREEN = "\033[32m" if USE_COLOR else ""
YELLOW = "\033[33m" if USE_COLOR else ""
CYAN = "\033[36m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{RED}Ошибка чтения {path}: {e}{RESET}")
        return None

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def equal_with_eps(a: Any, b: Any, eps: float) -> bool:
    if eps == 0:
        return a == b
    if is_number(a) and is_number(b):
        # одинаковые типы чисел сравним по допуску
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=eps)
    return a == b

def compare_json(a: Any, b: Any, path: str = "root") -> Tuple[int, List[str]]:
    """
    Возвращает:
      matches_count: количество совпавших атомарных значений,
      diffs: список строк с описанием отличий (с путём)
    """
    diffs: List[str] = []
    matches = 0

    # Оба dict
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            p = f"{path}.{k}"
            if k not in a:
                diffs.append(f"{YELLOW}{p}{RESET}: ключ отсутствует в {FOLDER1}, значение во 2-й: {repr(b[k])}")
            elif k not in b:
                diffs.append(f"{YELLOW}{p}{RESET}: ключ отсутствует в {FOLDER2}, значение в 1-й: {repr(a[k])}")
            else:
                m, d = compare_json(a[k], b[k], p)
                matches += m
                diffs.extend(d)
        return matches, diffs

    # Оба list
    if isinstance(a, list) and isinstance(b, list):
        n = max(len(a), len(b))
        for i in range(n):
            p = f"{path}[{i}]"
            if i >= len(a):
                diffs.append(f"{YELLOW}{p}{RESET}: элемент отсутствует в {FOLDER1}, значение во 2-й: {repr(b[i])}")
            elif i >= len(b):
                diffs.append(f"{YELLOW}{p}{RESET}: элемент отсутствует в {FOLDER2}, значение в 1-й: {repr(a[i])}")
            else:
                m, d = compare_json(a[i], b[i], p)
                matches += m
                diffs.extend(d)
        return matches, diffs

    # Типы различаются
    if type(a) is not type(b):
        diffs.append(f"{RED}{path}{RESET}: разные типы ({type(a).__name__} vs {type(b).__name__}); "
                     f"{FOLDER1}={repr(a)}, {FOLDER2}={repr(b)}")
        return matches, diffs

    # Атомарные значения
    if equal_with_eps(a, b, EPSILON):
        return matches + 1, diffs
    else:
        # для чисел покажем разницу
        if is_number(a) and is_number(b):
            diffs.append(f"{RED}{path}{RESET}: различие чисел "
                         f"{FOLDER1}={a}, {FOLDER2}={b}, Δ={abs(float(a)-float(b))}")
        else:
            diffs.append(f"{RED}{path}{RESET}: различие значений "
                         f"{FOLDER1}={repr(a)}, {FOLDER2}={repr(b)}")
        return matches, diffs

def main():
    files1 = set(f for f in os.listdir(FOLDER1) if f.endswith(".json"))
    files2 = set(f for f in os.listdir(FOLDER2) if f.endswith(".json"))
    common = sorted(files1 & files2)

    if not common:
        print(f"{YELLOW}Нет одинаковых JSON-файлов в двух папках{RESET}")
        return

    for fname in common:
        p1 = os.path.join(FOLDER1, fname)
        p2 = os.path.join(FOLDER2, fname)
        j1 = load_json(p1)
        j2 = load_json(p2)
        if j1 is None or j2 is None:
            continue

        matches, diffs = compare_json(j1, j2, "root")

        print(f"\n{CYAN}=== Файл: {fname} ==={RESET}")
        print(f"{GREEN}Совпадений атомарных значений: {matches}{RESET}")
        print(f"{RED}Отличий: {len(diffs)}{RESET}")

        if diffs:
            print(f"{RED}Отличия (путь → детали):{RESET}")
            for d in diffs:
                print(" - " + d)

if __name__ == "__main__":
    main()
