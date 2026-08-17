from __future__ import annotations

# ruff: noqa: E501

import contextvars
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

_CURRENT_SCENARIO: contextvars.ContextVar["ScenarioContext | None"] = contextvars.ContextVar("pdfrag_smart_scenario", default=None)

_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}
_NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_STATE_PATTERN = r"(?:fail(?:ed|ure)?|isolat(?:ed|ion)|defect(?:ive|ed)?|inoperative|operative|not\s+working|out\s+of\s+service|open|closed|available|unavailable|normal|abnormal|raised|lowered|tripped|healthy)"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CONTEXT_BLOCK_RE = re.compile(
    r"\[PDF CHUNK CONTEXT\](.*?)\[/PDF CHUNK CONTEXT\]",
    re.IGNORECASE | re.DOTALL,
)

_NOISE = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "do",
    "does", "for", "from", "has", "have", "if", "in", "is", "it", "of", "on",
    "or", "not", "shall", "should", "than", "that", "the", "then", "this", "to", "was",
    "were", "when", "with", "would",
}

_SYNONYM_STEMS = {
    "brakes": "brake",
    "braking": "brake",
    "failed": "fail",
    "failure": "fail",
    "failures": "fail",
    "failing": "fail",
    "isolated": "isolate",
    "isolation": "isolate",
    "isolations": "isolate",
    "defective": "defect",
    "defected": "defect",
    "doors": "door",
    "coaches": "coach",
    "cars": "car",
    "trains": "train",
    "indications": "indication",
    "indicators": "indicator",
    "controllers": "controller",
    "stations": "station",
}


@dataclass(frozen=True, slots=True)
class NumericFact:
    tokens: frozenset[str]
    value: float
    unit: str
    raw: str


@dataclass(frozen=True, slots=True)
class NumericRule:
    tokens: frozenset[str]
    operator: str
    threshold: float
    unit: str
    raw: str


@dataclass(slots=True)
class ScenarioContext:
    original_question: str
    canonical_question: str
    numeric_facts: list[NumericFact] = field(default_factory=list)
    states: dict[str, str] = field(default_factory=dict)
    inferred_states: dict[str, str] = field(default_factory=dict)
    terminology: list[str] = field(default_factory=list)

    @property
    def is_situational(self) -> bool:
        lowered = self.canonical_question.casefold()
        return bool(
            self.numeric_facts
            or self.states
            or self.inferred_states
            or re.search(
                r"\b(?:if|when|stuck|stopped|unable|cannot|can't|won't|failed|failure|fault|what\s+(?:should|shall|do)|what\s+to\s+do)\b",
                lowered,
            )
        )


def compile_scenario(question: str, terminology_hints: Iterable[str] = ()) -> ScenarioContext:
    hints = [" ".join(str(value).split()) for value in terminology_hints if str(value).strip()]
    expansions: list[str] = []
    for hint in hints:
        if "=" not in hint or "ambiguous corpus meaning" in hint.casefold():
            continue
        left, right = hint.split("=", 1)
        alias = left.strip()
        canonical = right.split("—", 1)[0].split("--", 1)[0].strip()
        if alias and canonical:
            expansions.append(f"{alias} means {canonical}")

    canonical_question = " ".join(question.split())
    if expansions:
        canonical_question = f"{canonical_question}. Terminology: {'; '.join(expansions[:8])}."

    context = ScenarioContext(
        original_question=question,
        canonical_question=canonical_question,
        terminology=hints,
    )
    context.numeric_facts.extend(extract_numeric_facts(canonical_question))
    context.states.update(extract_states(canonical_question))

    lowered = canonical_question.casefold()
    unable_to_move = bool(
        re.search(
            r"\b(?:unable\s+to\s+(?:move|proceed)|cannot\s+(?:move|proceed)|can't\s+(?:move|proceed)|won't\s+(?:move|take\s+traction)|not\s+taking\s+traction|no\s+traction|immobili[sz]ed|stuck\s+between)\b",
            lowered,
        )
    )
    if unable_to_move:
        context.inferred_states["train_movement"] = "unable_to_proceed"
    if re.search(r"\bbetween\s+(?:two|2)\s+stations?\b|\bbetween\s+stations?\b", lowered):
        context.inferred_states["location"] = "between_stations"
    return context


def extract_numeric_facts(text: str) -> list[NumericFact]:
    facts: list[NumericFact] = []
    seen: set[tuple[frozenset[str], float, str]] = set()

    count_patterns = [
        re.compile(
            rf"\b(?P<num>{_NUMBER_PATTERN})\s+(?P<subject>[A-Za-z][A-Za-z0-9/-]*(?:\s+[A-Za-z][A-Za-z0-9/-]*){{0,3}}?)\s+(?:are\s+|is\s+|have\s+|has\s+)?(?P<state>{_STATE_PATTERN})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<num>{_NUMBER_PATTERN})\s+(?P<state>{_STATE_PATTERN})\s+(?P<subject>[A-Za-z][A-Za-z0-9/-]*(?:\s+[A-Za-z][A-Za-z0-9/-]*){{0,3}})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:failure|failures|isolation|isolations)\s+of\s+(?P<num>{_NUMBER_PATTERN})\s+(?P<subject>[A-Za-z][A-Za-z0-9/-]*(?:\s+[A-Za-z][A-Za-z0-9/-]*){{0,3}})\b",
            re.IGNORECASE,
        ),
    ]
    for pattern in count_patterns:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 24) : match.start()]
            if re.search(r"\bline\s*[-_:]?\s*$", prefix, re.IGNORECASE):
                continue
            if re.search(r"\b(?:SC|SM|SOP|JPO|INST(?:RUCTION)?|MRGR)\s*[-_/ ]\s*$", prefix, re.IGNORECASE):
                continue
            value = _number(match.group("num"))
            if value is None:
                continue
            subject = match.group("subject")
            state = match.groupdict().get("state") or match.group(0).split()[0]
            tokens = _concept_tokens(f"{subject} {_state_value(state)}")
            if not tokens:
                continue
            key = (tokens, value, "count")
            if key in seen:
                continue
            seen.add(key)
            facts.append(NumericFact(tokens=tokens, value=value, unit="count", raw=match.group(0)))

    measurement_pattern = re.compile(
        rf"\b(?P<subject>(?:brake\s+)?pressure|speed|velocity|voltage|line\s+voltage|temperature|distance|duration|time)\s*(?:is|=|of|at|around|approximately|about)?\s*(?P<num>{_NUMBER_PATTERN})\s*(?P<unit>km/?h|kph|bar|psi|kv|v|volts?|c|degc|seconds?|secs?|minutes?|mins?|hours?|hrs?|m|mm|cm|km)?\b",
        re.IGNORECASE,
    )
    for match in measurement_pattern.finditer(text):
        value = _number(match.group("num"))
        if value is None:
            continue
        unit = _normalize_unit(match.group("unit") or "")
        tokens = _concept_tokens(match.group("subject"))
        key = (tokens, value, unit)
        if key in seen:
            continue
        seen.add(key)
        facts.append(NumericFact(tokens=tokens, value=value, unit=unit, raw=match.group(0)))
    return facts


def extract_numeric_rules(text: str) -> list[NumericRule]:
    body = strip_chunk_context(text)
    rules: list[NumericRule] = []
    seen: set[tuple[frozenset[str], str, float, str]] = set()

    patterns: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                rf"\b(?:failure|failures|isolation|isolations)\s+of\s+(?P<num>{_NUMBER_PATTERN})\s+(?:or\s+more|or\s+above|and\s+above|or\s+greater)\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,50}}?)(?=(?:\.|,|;|\n|\s+(?:shall|must|should|will|is|are|has|have|fail|failed|isolat|defect)))",
                re.IGNORECASE,
            ),
            ">=",
        ),
        (
            re.compile(
                rf"\b(?:number|count)\s+of\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,35}}?)\s+(?:is|are|becomes?|reaches?)\s+(?P<num>{_NUMBER_PATTERN})\s+(?:or\s+more|or\s+above|and\s+above|or\s+greater)\b",
                re.IGNORECASE,
            ),
            ">=",
        ),
        (
            re.compile(
                rf"\b(?P<num>{_NUMBER_PATTERN})\s+(?:or\s+more|or\s+above|and\s+above|or\s+greater)\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,50}}?)(?=(?:\.|,|;|\n|\s+(?:shall|must|should|will|is|are|has|have|fail|failed|isolat|defect)))",
                re.IGNORECASE,
            ),
            ">=",
        ),
        (
            re.compile(
                rf"\b(?:at\s+least|not\s+less\s+than)\s+(?P<num>{_NUMBER_PATTERN})\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,50}}?)(?=(?:\.|,|;|\n|\s+(?:shall|must|should|will|is|are|has|have|fail|failed|isolat|defect)))",
                re.IGNORECASE,
            ),
            ">=",
        ),
        (
            re.compile(
                rf"\b(?:more\s+than|greater\s+than|above|exceed(?:ing|s)?)\s+(?P<num>{_NUMBER_PATTERN})\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,50}}?)(?=(?:\.|,|;|\n|\s+(?:shall|must|should|will|is|are|has|have|fail|failed|isolat|defect)))",
                re.IGNORECASE,
            ),
            ">",
        ),
        (
            re.compile(
                rf"\b(?P<num>{_NUMBER_PATTERN})\s+(?:or\s+less|or\s+fewer|or\s+below)\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,50}}?)(?=(?:\.|,|;|\n|\s+(?:shall|must|should|will|is|are|has|have|fail|failed|isolat|defect)))",
                re.IGNORECASE,
            ),
            "<=",
        ),
        (
            re.compile(
                rf"\b(?:less\s+than|below|under)\s+(?P<num>{_NUMBER_PATTERN})\s+(?P<subject>[A-Za-z][A-Za-z0-9 /_-]{{0,50}}?)(?=(?:\.|,|;|\n|\s+(?:shall|must|should|will|is|are|has|have|fail|failed|isolat|defect)))",
                re.IGNORECASE,
            ),
            "<",
        ),
    ]

    state_tail = re.compile(rf"\b(?P<state>{_STATE_PATTERN})\b", re.IGNORECASE)
    for pattern, operator in patterns:
        for match in pattern.finditer(body):
            value = _number(match.group("num"))
            if value is None:
                continue
            subject = " ".join(match.group("subject").split())
            prefix = body[max(0, match.start() - 20) : match.start() + 35].casefold()
            if re.search(r"\bfail(?:ure|ures)?\s+of\b", prefix):
                subject = f"{subject} failed"
            elif re.search(r"\bisolat(?:ion|ions)?\s+of\b", prefix):
                subject = f"{subject} isolated"
            # A state word is often just after the captured noun phrase.
            tail = body[match.end() : match.end() + 50]
            state_match = state_tail.search(tail)
            if state_match and len(subject.split()) <= 6:
                subject = f"{subject} {_state_value(state_match.group('state'))}"
            tokens = _concept_tokens(subject)
            if not tokens:
                continue
            rule = NumericRule(
                tokens=tokens,
                operator=operator,
                threshold=value,
                unit="count",
                raw=_sentence_excerpt(body, match.start(), match.end()),
            )
            key = (tokens, operator, value, "count")
            if key not in seen:
                seen.add(key)
                rules.append(rule)

    measurement_patterns: list[tuple[re.Pattern[str], str]] = []
    for phrase, operator in [
        (r"at\s+least|not\s+less\s+than", ">="),
        (r"more\s+than|greater\s+than|above|exceed(?:ing|s)?", ">"),
        (r"not\s+more\s+than|up\s+to|at\s+most", "<="),
        (r"less\s+than|below|under", "<"),
    ]:
        measurement_patterns.append(
            (
                re.compile(
                    rf"\b(?P<subject>(?:brake\s+)?pressure|speed|velocity|voltage|line\s+voltage|temperature|distance|duration|time)\s*(?:is|shall\s+be|must\s+be|of)?\s*(?:{phrase})\s*(?P<num>{_NUMBER_PATTERN})\s*(?P<unit>km/?h|kph|bar|psi|kv|v|volts?|c|degc|seconds?|secs?|minutes?|mins?|hours?|hrs?|m|mm|cm|km)?\b",
                    re.IGNORECASE,
                ),
                operator,
            )
        )
    for pattern, operator in measurement_patterns:
        for match in pattern.finditer(body):
            value = _number(match.group("num"))
            if value is None:
                continue
            unit = _normalize_unit(match.group("unit") or "")
            tokens = _concept_tokens(match.group("subject"))
            key = (tokens, operator, value, unit)
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                NumericRule(
                    tokens=tokens,
                    operator=operator,
                    threshold=value,
                    unit=unit,
                    raw=_sentence_excerpt(body, match.start(), match.end()),
                )
            )
    return rules


def extract_states(text: str) -> dict[str, str]:
    states: dict[str, str] = {}
    patterns = {
        "vcb": r"\bVCB\b\s*(?:is|=|remains?)?\s*(open|closed|tripped)",
        "pantograph": r"\bpantograph\b\s*(?:is|=|remains?)?\s*(raised|lowered|up|down)",
        "line_voltage": r"\b(?:line\s+)?voltage\b\s*(?:is|=|shows?|showing)?\s*(available|unavailable|normal|abnormal|present|absent)",
        "traction": r"\btraction\b\s*(?:is|=|remains?)?\s*(available|unavailable|normal|failed|failure|not\s+available)",
        "brake": r"\bbrakes?\b\s*(?:are|is|=)?\s*(normal|healthy|failed|isolated|defective|inoperative)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            states[key] = _state_value(match.group(1))
    return states


def logical_match_score(scenario: ScenarioContext, source_text: str) -> tuple[float, list[str]]:
    if not scenario.is_situational:
        return 0.0, []
    notes: list[str] = []
    boost = 0.0

    for rule in extract_numeric_rules(source_text):
        best: tuple[float, NumericFact] | None = None
        for fact in scenario.numeric_facts:
            similarity = _token_similarity(rule.tokens, fact.tokens)
            if rule.unit and fact.unit and rule.unit != fact.unit and "count" not in {rule.unit, fact.unit}:
                similarity *= 0.25
            if best is None or similarity > best[0]:
                best = (similarity, fact)
        if best is None or best[0] < 0.42:
            continue
        fact = best[1]
        applies = _evaluate(fact.value, rule.operator, rule.threshold)
        concept = " ".join(sorted(rule.tokens)) or "condition"
        if applies:
            boost += 0.32
            notes.append(
                f"User reports {fact.value:g} for {concept}; source condition is {rule.operator} {rule.threshold:g}. "
                f"Deterministic evaluation: {fact.value:g} {rule.operator} {rule.threshold:g} is true."
            )
        else:
            boost -= 0.18
            notes.append(
                f"User reports {fact.value:g} for {concept}; source condition is {rule.operator} {rule.threshold:g}. "
                f"Deterministic evaluation is false, so this threshold does not match."
            )

    source_states = extract_states(strip_chunk_context(source_text))
    for key, user_state in scenario.states.items():
        source_state = source_states.get(key)
        if not source_state:
            continue
        if source_state == user_state:
            boost += 0.06
        elif {source_state, user_state} <= {"available", "unavailable"} or {source_state, user_state} <= {"open", "closed"}:
            boost -= 0.10

    return max(-0.35, min(0.45, boost)), notes[:4]


def scenario_relaxed_query(scenario: ScenarioContext) -> str:
    text = scenario.canonical_question
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)
    for word in _NUMBER_WORDS:
        text = re.sub(rf"\b{word}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,;")
    additions: list[str] = []
    for fact in scenario.numeric_facts[:4]:
        additions.extend(sorted(fact.tokens))
    for key, value in list(scenario.states.items())[:6]:
        additions.extend([key.replace("_", " "), value])
    for key, value in list(scenario.inferred_states.items())[:4]:
        additions.extend([key.replace("_", " "), value.replace("_", " ")])
    return " ".join([text, *additions]).strip()


def scenario_relevance_question(scenario: ScenarioContext) -> str:
    """Remove user numeric values that should be evaluated, not literally matched.

    The legacy relevance layer treats numbers as hard anchors.  For a threshold
    question that would incorrectly require the source to contain the user's value
    (4) instead of the rule threshold (2).  Scope identifiers such as ``Line 4``
    and document codes such as ``SC-04`` are preserved.
    """
    text = scenario.canonical_question
    removable: list[str] = []
    word_by_number = {int(value): word for word, value in _NUMBER_WORDS.items() if float(value).is_integer()}
    for fact in scenario.numeric_facts:
        value = fact.value
        rendered = f"{value:g}"
        # Preserve a number when the same value is explicitly an applicability or
        # document identifier in the user's wording.
        escaped = re.escape(rendered)
        if re.search(rf"\bline\s*[-_:]?\s*{escaped}\b", scenario.original_question, re.IGNORECASE):
            continue
        if re.search(rf"\b(?:SC|SM|SOP|JPO|INST(?:RUCTION)?|MRGR)\s*[-_/ ]\s*{escaped}\b", scenario.original_question, re.IGNORECASE):
            continue
        removable.append(rendered)
        if value.is_integer() and int(value) in word_by_number:
            removable.append(word_by_number[int(value)])
    for token in sorted(set(removable), key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" .,;")


def scenario_logic_value_terms(scenario: ScenarioContext) -> set[str]:
    """Return normalized literal value tokens safe to drop from relevance metadata."""
    result: set[str] = set()
    for fact in scenario.numeric_facts:
        rendered = f"{fact.value:g}".casefold()
        if not re.search(rf"\bline\s*[-_:]?\s*{re.escape(rendered)}\b", scenario.original_question, re.IGNORECASE):
            result.add(rendered)
            if fact.value.is_integer():
                for word, value in _NUMBER_WORDS.items():
                    if value == fact.value:
                        result.add(word)
    return result

def parse_chunk_context(text: str) -> dict[str, str]:
    match = _CONTEXT_BLOCK_RE.search(text)
    if not match:
        return {}
    result: dict[str, str] = {}
    prefixes = {
        "Pages:": "pages",
        "Section path:": "section",
        "Rolling stock / train context:": "stock",
        "Procedure context:": "procedure",
        "Important tags:": "tags",
    }
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        for prefix, key in prefixes.items():
            if line.casefold().startswith(prefix.casefold()):
                result[key] = line[len(prefix) :].strip()
                break
    return result


def strip_chunk_context(text: str) -> str:
    return _CONTEXT_BLOCK_RE.sub("", text or "").strip()


def _number(value: str) -> float | None:
    cleaned = value.casefold().strip()
    if cleaned in _NUMBER_WORDS:
        return _NUMBER_WORDS[cleaned]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _concept_tokens(value: str) -> frozenset[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(value.casefold()):
        if raw in _NOISE or raw.isdigit():
            continue
        token = _SYNONYM_STEMS.get(raw, raw)
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed") and not token.endswith("eed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        token = _SYNONYM_STEMS.get(token, token)
        if len(token) >= 2:
            tokens.add(token)
    return frozenset(tokens)


_CONDITION_STATE_TOKENS = {"fail", "isolate", "operative", "open", "closed", "available", "unavailable"}


def _token_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    shared = left & right
    if not shared:
        return 0.0
    left_states = left & _CONDITION_STATE_TOKENS
    right_states = right & _CONDITION_STATE_TOKENS
    left_entities = left - _CONDITION_STATE_TOKENS
    right_entities = right - _CONDITION_STATE_TOKENS
    # Matching only on a generic condition word (for example "failed") is unsafe:
    # four failed brakes must not satisfy a threshold about two failed doors.
    if left_entities and right_entities and not (left_entities & right_entities):
        return 0.0
    score = len(shared) / max(1, min(len(left), len(right)))
    if left_states and right_states and not (left_states & right_states):
        score *= 0.20
    return score


def _evaluate(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        return value == threshold
    return False


def _normalize_unit(unit: str) -> str:
    value = unit.casefold().replace(" ", "")
    aliases = {
        "km/h": "km/h",
        "kmh": "km/h",
        "kph": "km/h",
        "volt": "v",
        "volts": "v",
        "sec": "s",
        "secs": "s",
        "second": "s",
        "seconds": "s",
        "min": "min",
        "mins": "min",
        "minute": "min",
        "minutes": "min",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
    }
    return aliases.get(value, value)


def _state_value(value: str) -> str:
    cleaned = " ".join(value.casefold().split())
    aliases = {
        "not available": "unavailable",
        "fail": "failed",
        "failure": "failed",
        "defect": "failed",
        "defective": "failed",
        "defected": "failed",
        "inoperative": "failed",
        "not working": "failed",
        "out of service": "failed",
        "present": "available",
        "absent": "unavailable",
        "up": "raised",
        "down": "lowered",
        "healthy": "normal",
    }
    return aliases.get(cleaned, cleaned)


def _sentence_excerpt(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    right_dot = text.find(".", end)
    right_nl = text.find("\n", end)
    candidates = [value for value in (right_dot, right_nl) if value >= 0]
    right = min(candidates) if candidates else min(len(text), end + 220)
    left = 0 if left < 0 else left + 1
    return " ".join(text[left : right + 1].split())[:420]


def set_current_scenario(value: ScenarioContext | None) -> None:
    _CURRENT_SCENARIO.set(value)


def current_scenario() -> ScenarioContext | None:
    return _CURRENT_SCENARIO.get()
