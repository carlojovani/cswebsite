import re

# маркеры битого текста (типичные "Рџ...", "СЃ...", "вЂ—", "Ð", "Ñ", "Â")
BAD_RE = re.compile(r"(Р[А-Яа-яЁё]|С[А-Яа-яЁё]|вЂ|Ð|Ñ|Â)", re.U)
CYR_RE = re.compile(r"[А-Яа-яЁё]", re.U)

def fix_mojibake(s: str) -> str:
    """
    Чинит строку целиком (без "кусков").
    Пробует типичные случаи:
      - UTF-8 байты были прочитаны как cp1251
      - UTF-8 байты были прочитаны как latin1/cp1252
    Выбирает лучший вариант по эвристике.
    """
    if not isinstance(s, str) or not s:
        return s

    # если маркеров нет — не трогаем
    if not BAD_RE.search(s):
        return s

    def score(txt: str) -> tuple[int, int, int]:
        # больше кириллицы = лучше
        cyr = len(CYR_RE.findall(txt))
        # меньше "плохих маркеров" = лучше
        bad = len(BAD_RE.findall(txt))
        # меньше replacement символов = лучше
        repl = txt.count("�")
        return (cyr, -bad, -repl)

    candidates: list[str] = [s]

    # 1) самый частый случай: UTF-8 -> cp1251
    try:
        candidates.append(s.encode("cp1251", errors="strict").decode("utf-8", errors="strict"))
    except Exception:
        pass
    try:
        candidates.append(s.encode("cp1251", errors="ignore").decode("utf-8", errors="ignore"))
    except Exception:
        pass

    # 2) latin1/cp1252 варианты
    try:
        candidates.append(s.encode("latin1", errors="strict").decode("utf-8", errors="strict"))
    except Exception:
        pass
    try:
        candidates.append(s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore"))
    except Exception:
        pass

    try:
        candidates.append(s.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore"))
    except Exception:
        pass

    best = max(candidates, key=score)

    # страховка: не ухудшаем строку (по bad-маркерам)
    if len(BAD_RE.findall(best)) > len(BAD_RE.findall(s)):
        return s

    return best


def deep_fix(obj):
    if isinstance(obj, str):
        return fix_mojibake(obj)
    if isinstance(obj, list):
        return [deep_fix(x) for x in obj]
    if isinstance(obj, dict):
        return {deep_fix(k) if isinstance(k, str) else k: deep_fix(v) for k, v in obj.items()}
    return obj
