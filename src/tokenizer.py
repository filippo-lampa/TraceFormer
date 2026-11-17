
import ast
import json
import re
import math
from collections import Counter

import pandas as pd


# ============================================================
# NORMALIZATION PATTERNS
# ============================================================

_ADDR_0X = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ADDR_NO0X = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_0X = re.compile(r"^0x[0-9a-fA-F]+$")
_HEX32_0X = re.compile(r"^0x[0-9a-fA-F]{64}$")
_HEX32_BARE = re.compile(r"^[0-9a-fA-F]{64}$")

_ISO_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:[Zz]|[+-]\d{2}:?\d{2})$"
)

_SCI_NOTATION = re.compile(
    r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$"
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ABI_ARRAY = re.compile(
    r"^(?P<base>[A-Za-z_][A-Za-z_0-9]*)(?:\[\d*\])+$"
)
_ABI_SIMPLE = re.compile(
    r"^(?:u?int(?:\d{0,3})|bytes(?:\d{1,2})?|"
    r"address|bool|string|function|tuple)$"
)

_URL = re.compile(r"(?:https?://|www\.)", re.I)
_IPFS = re.compile(r"^(?:ipfs://|/ipfs/)", re.I)
_ENS = re.compile(r"^(?:[a-z0-9-]+\.)+eth$", re.I)
_HASH_BARE_QUOTED = re.compile(
    r"^[\"'][0-9a-fA-F]{64}[\"']$"
)
_PROSE = re.compile(r"(?:\s|[,:;.!?])")


# ============================================================
# RESERVED OUTPUT TOKENS
# ============================================================

_RESERVED = {
    "[START]", "[END]",
    "[INsSTART]", "[INsEND]",
    "[INXsSTART]", "[INXsEND]",
    "[CALLSTART]", "[CALL_END]",
    "[CALLS_START]", "[CALLS_END]",
    "[CALLS_CHILD_START]", "CALLS_CHILD_END",
    "[TIMESTAMP_ISO]", "[TXHASH]",
    "[ADDR]", "[HEX]", "[HEX32]", "[MULTI_HEX]",
    "[DATE]", "[NA]", "[NONE]", "[EMPTY_STR]",
    "[FLOAT]", "[SCI_NOTATION]", "[NUM_STR]",
    "[BOOL_TRUE]", "[BOOL_FALSE]",
    "[INT_0]", "[INT_1]",
    "[INT_LT10]", "[INT_LT100]",
    "[INT_LT1K]", "[INT_LT10K]",
    "[INT_LT1M]", "[INT_LT1B]",
    "[INT_BILLION]", "[INT_TRILLION]",
    "[INT_QUADRILLION]", "[INT_HUGE]",
    "[TEXT]", "[IPFS_URI]", "[URI]", "[ENS_NAME]",
    "[ABI_TUPLE]", "[ABI_TUPLE_ARRAY]",
    "[UNPARSED_INPUT]",
    "[UNPARSED_CALLS]",
    "[UNPARSED_INTERNAL_TX]",
    "[CONTRACT_NAME]",
    "[FUNCTION_NAME]",
    "[ACTIVITY_NAME]",
    "[PARAM_NAME]",
    "[EMPTY_BYTES]",
    "[DICT]", "[SET]",
}

_LIST_BUCKET_MARKERS = (
    "[LIST_0]",
    "[LIST_1]",
    "[LIST_2]",
    "[LIST_3_4]",
    "[LIST_5_9]",
    "[LIST_10_19]",
    "[LIST_20_49]",
    "[LIST_50_99]",
    "[LIST_100_999]",
    "[LIST_GE1000]",
)

_RESERVED.update(_LIST_BUCKET_MARKERS)
_RESERVED.update(
    f"[INT_{i}D]" for i in range(10, 13)
)
_RESERVED.update(
    f"[ABI_ARRAY_{family}]"
    for family in (
        "UINT", "INT", "BYTES", "ADDRESS",
        "BOOL", "STRING", "FUNCTION", "TUPLE"
    )
)

RESERVED_OUTPUT_TOKENS = tuple(sorted(_RESERVED))

_FREE_TEXT_FIELDS = {
    "description", "reason", "message",
    "comment", "comments", "rationale",
    "proposaldescription"
}

_TEXT_LENGTH_THRESHOLD = 120

_SMALL_INT_BUCKETS = [
    (0, "[INT_0]"),
    (1, "[INT_1]"),
    (10, "[INT_LT10]"),
    (100, "[INT_LT100]"),
    (1000, "[INT_LT1K]"),
    (10_000, "[INT_LT10K]"),
    (1_000_000, "[INT_LT1M]"),
    (1_000_000_000, "[INT_LT1B]"),
]


# ============================================================
# NUMERIC NORMALIZATION
# ============================================================

def _safe_int_token(n: int) -> str:
    absn = abs(n)

    for bound, tok in _SMALL_INT_BUCKETS:
        if absn <= bound:
            return tok

    if absn > 1_000_000_000:
        if absn < 10**12:
            return "[INT_BILLION]"
        elif absn < 10**15:
            return "[INT_TRILLION]"
        elif absn < 10**18:
            return "[INT_QUADRILLION]"
        else:
            return "[INT_HUGE]"

    return str(n)


def _abi_type_token(value):
    if not isinstance(value, str):
        return None

    s = value.strip().replace(" ", "")

    if s.startswith("(") and ")" in s:
        depth = 0
        tuple_end = -1

        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return None

                if depth == 0:
                    tuple_end = i
                    break

        if (
            tuple_end > 0
            and depth == 0
            and re.fullmatch(
                r"(?:\[\d*\])*", s[tuple_end + 1:]
            )
        ):
            if tuple_end + 1 < len(s):
                return "[ABI_TUPLE_ARRAY]"
            return "[ABI_TUPLE]"

    match = _ABI_ARRAY.fullmatch(s)

    if match and _ABI_SIMPLE.fullmatch(match.group("base")):
        base = match.group("base")

        if base.startswith("uint"):
            family = "UINT"
        elif base.startswith("int"):
            family = "INT"
        elif base.startswith("bytes"):
            family = "BYTES"
        else:
            family = base.upper()

        return f"[ABI_ARRAY_{family}]"

    return None


# ============================================================
# MAIN VALUE NORMALIZATION
# ============================================================

def normalize_token(value, field_name=None):

    if isinstance(value, bool):
        return (
            "[BOOL_TRUE]" if value else "[BOOL_FALSE]"
        )

    if isinstance(value, int):
        return _safe_int_token(value)

    if isinstance(value, float):
        if pd.isna(value):
            return "[NA]"
        return "[FLOAT]"

    if value is None:
        return "[NONE]"

    if isinstance(value, str):
        s = value.strip()

        if not s:
            return "[EMPTY_STR]"

        if s.lower() == "0x":
            return "[EMPTY_BYTES]"

        field_lower = (
            field_name.casefold()
            if isinstance(field_name, str)
            else ""
        )

        # Name abstraction (v3/v4 experiment)
        if field_lower in {
            "contractcalledname", "contractname"
        }:
            return "[CONTRACT_NAME]"

        if field_lower in {
            "functionname", "call_function_name"
        }:
            return "[FUNCTION_NAME]"

        if field_lower == "activity":
            return "[ACTIVITY_NAME]"

        if field_lower in {
            "inputname", "abi_parameter_name"
        }:
            return "[PARAM_NAME]"

        if field_lower in _FREE_TEXT_FIELDS:
            return "[TEXT]"

        if _IPFS.match(s):
            return "[IPFS_URI]"

        if _URL.search(s):
            return "[URI]"

        if _ENS.fullmatch(s):
            return "[ENS_NAME]"

        if _HASH_BARE_QUOTED.fullmatch(s):
            return "[HEX32]"

        if (
            len(s) > _TEXT_LENGTH_THRESHOLD
            and any(c.isspace() for c in s)
        ):
            return "[TEXT]"

        if (
            field_lower in {
                "inputvalue", "output", "value"
            }
            and " " in s
            and len(s) >= 24
            and _PROSE.search(s)
        ):
            return "[TEXT]"

        # Preserve scalar ABI types,
        # canonicalize composite ABI types.
        if field_name == "type":
            abi_type = _abi_type_token(s)
            if abi_type is not None:
                return abi_type

        if _SCI_NOTATION.fullmatch(s):
            return "[SCI_NOTATION]"

        if _ISO_TIMESTAMP.fullmatch(s):
            return "[TIMESTAMP_ISO]"

        if "," in s and "0x" in s:
            hex_parts = s.split(",")
            if (
                len(hex_parts) > 1
                and all(
                    part.strip().startswith("0x")
                    for part in hex_parts
                )
            ):
                return "[MULTI_HEX]"

        if (
            field_name == "transactionHash"
            and (
                _HEX32_0X.fullmatch(s)
                or _HEX32_BARE.fullmatch(s)
            )
        ):
            return "[TXHASH]"

        if _ADDR_0X.match(s) or _ADDR_NO0X.match(s):
            return "[ADDR]"

        if (
            _HEX32_0X.fullmatch(s)
            or _HEX32_BARE.fullmatch(s)
        ):
            return "[HEX32]"

        if _HEX_0X.match(s):
            return "[HEX]"

        if (
            s.isdigit()
            or (s[0] == "-" and s[1:].isdigit())
        ):
            try:
                num = int(s)

                if abs(num) > 1_000_000_000:
                    digit_count = len(s.lstrip("-"))

                    if digit_count <= 12:
                        return f"[INT_{digit_count}D]"

                    return "[INT_HUGE]"

                return _safe_int_token(num)

            except (ValueError, OverflowError):
                return "[NUM_STR]"

        if _DATE.fullmatch(s):
            return "[DATE]"

        return s

    # v4 list-size buckets
    if isinstance(value, (list, tuple)):
        n = len(value)

        if n == 0:
            return "[LIST_0]"
        if n == 1:
            return "[LIST_1]"
        if n == 2:
            return "[LIST_2]"
        if n <= 4:
            return "[LIST_3_4]"
        if n <= 9:
            return "[LIST_5_9]"
        if n <= 19:
            return "[LIST_10_19]"
        if n <= 49:
            return "[LIST_20_49]"
        if n <= 99:
            return "[LIST_50_99]"
        if n <= 999:
            return "[LIST_100_999]"

        return "[LIST_GE1000]"

    if isinstance(value, dict):
        return "[DICT]"

    if isinstance(value, set):
        return "[SET]"

    return "[NONE]"


# ============================================================
# PARSING HELPERS
# ============================================================

def _is_missing(value):
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    try:
        return value is pd.NA
    except AttributeError:
        return False


def _maybe_parse_literal(
    x, *, audit=None, location="value"
):
    if not isinstance(x, str):
        return x

    s = x.strip()

    if s in ("", "null", "None"):
        return None if s in ("null", "None") else x

    if not s.startswith(("[", "{", "(")):
        return x

    # Don't parse an existing canonical marker.
    if (
        s.startswith("[")
        and s.endswith("]")
        and re.fullmatch(
            r"\[[A-Z][A-Z_0-9]*\]", s
        )
    ):
        return x

    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(s)
        except (
            ValueError, SyntaxError,
            TypeError, RecursionError
        ):
            pass

    if audit is not None:
        audit[f"parse_failure:{location}"] += 1

    return x


def _as_list(value):
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _input_value_field(input_dict):
    name = input_dict.get("inputName")

    if (
        isinstance(name, str)
        and name.strip().casefold()
        in _FREE_TEXT_FIELDS
    ):
        return name.strip()

    return "inputValue"


# ============================================================
# SOURCE TRACKING FOR CSV EXPORTS
# ============================================================

class _SourcedToken(str):
    """
    Behaves like a normal string, but keeps the
    original JSON value for diagnostic exports.
    """

    def __new__(
        cls, normalized, original, field_name
    ):
        obj = super().__new__(cls, str(normalized))
        obj.raw_source_value = original
        obj.source_field = field_name
        return obj


def _source_token(value, field_name):
    return _SourcedToken(
        normalize_token(
            value, field_name=field_name
        ),
        value,
        field_name
    )


def tokenizer_with_sources(dataFrame):
    """
    One source record per normalized token.

    This output is aligned position-by-position
    with tokenizer(dataFrame).
    """
    audit = Counter()
    out = []

    for _, row in dataFrame.iterrows():
        out.extend(
            _tokenize_record(
                row.to_dict(),
                nested=False,
                audit=audit
            )
        )

    return [
        {
            "normalized_token": str(t),
            "raw_source_value": (
                t.raw_source_value
                if isinstance(t, _SourcedToken)
                else None
            ),
            "source_field": (
                t.source_field
                if isinstance(t, _SourcedToken)
                else None
            ),
            "has_source_value": isinstance(
                t, _SourcedToken
            )
        }
        for t in out
    ]


# ============================================================
# INPUT TOKENIZATION
# ============================================================

def _tokenize_inputs(value, audit):
    out = ["[INsSTART]"]

    parsed = _maybe_parse_literal(
        value,
        audit=audit,
        location="inputs"
    )

    if (
        isinstance(parsed, str)
        and parsed.strip().startswith(("[", "{"))
        and not re.fullmatch(
            r"\[[A-Z][A-Z_0-9]*\]",
            parsed.strip()
        )
    ):
        audit["unparsed_input_container"] += 1
        out.append("[UNPARSED_INPUT]")

    else:
        for item in _as_list(parsed):
            item = _maybe_parse_literal(
                item,
                audit=audit,
                location="input_item"
            )

            if isinstance(item, dict):
                abi_type = str(item.get("type", "")).strip()
            
                for key, val in item.items():
                    field = str(key)
                    out.append(field)
            
                    if field in ("value", "inputValue") and abi_type == "string":
                        out.append(
                            _SourcedToken(
                                "[STRING_VALUE]",
                                val,
                                field,
                            )
                        )
            
                    else:
                        if field == "inputValue":
                            context = _input_value_field(item)
                        elif field == "name":
                            context = "abi_parameter_name"
                        else:
                            context = field
            
                        out.append(_source_token(val, context))

            else:
                out.append(
                    _source_token(
                        item, "inputValue"
                    )
                )

    out.append("[INsEND]")
    return out


# ============================================================
# CALL TOKENIZATION
# ============================================================

def tokenize_calls(cell_value, audit=None):
    audit = (
        Counter() if audit is None else audit
    )

    calls = _maybe_parse_literal(
        cell_value,
        audit=audit,
        location="calls"
    )

    if _is_missing(calls):
        return []

    if isinstance(calls, dict):
        calls = [calls]

    if not isinstance(calls, (list, tuple)):
        audit["unparsed_calls"] += 1
        return ["[UNPARSED_CALLS]"]

    out = []

    for call in calls:
        call = _maybe_parse_literal(
            call,
            audit=audit,
            location="call_item"
        )

        if not isinstance(call, dict):
            audit["unparsed_call_item"] += 1
            out.append("[UNPARSED_CALLS]")
            continue

        out.append("[CALLSTART]")

        for key, val in call.items():
            if key in ("calls", "events"):
                continue

            if key == "inputs":
                out.extend(
                    _tokenize_inputs(val, audit)
                )

            else:
                field = str(key)

                context = (
                    "call_function_name"
                    if field == "name"
                    else field
                )

                out.extend(
                    (
                        field,
                        _source_token(
                            val, context
                        )
                    )
                )

        if (
            "calls" in call
            and not _is_missing(call["calls"])
            and call["calls"] not in (
                [], "", "[]"
            )
        ):
            out.append("[CALLS_CHILD_START]")

            out.extend(
                tokenize_calls(
                    call["calls"], audit
                )
            )

            out.append("CALLS_CHILD_END")

        out.append("[CALL_END]")

    return out


# ============================================================
# RECORD TOKENIZATION
# ============================================================

def _tokenize_record(
    record, *, nested, audit
):
    out = [] if nested else ["[START]"]

    for key, value in record.items():
        field = str(key)

        # Events intentionally excluded.
        if field == "events":
            continue

        if field == "inputs":
            out.extend(
                _tokenize_inputs(
                    value, audit
                )
            )

        elif field == "timestamp":
            ts = _maybe_parse_literal(
                value,
                audit=audit,
                location="timestamp"
            )

            if isinstance(ts, dict) and ts:
                ts = next(iter(ts.values()))

            out.extend(
                (
                    "timestamp",
                    _source_token(
                        ts, "timestamp"
                    )
                )
            )

        elif field == "internalTxs":
            out.extend(
                (
                    "internalTxs",
                    "[INXsSTART]"
                )
            )

            internal = _maybe_parse_literal(
                value,
                audit=audit,
                location="internalTxs"
            )

            for inner in _as_list(internal):
                inner = _maybe_parse_literal(
                    inner,
                    audit=audit,
                    location="internalTx_item"
                )

                if isinstance(inner, dict):
                    out.extend(
                        _tokenize_record(
                            inner,
                            nested=True,
                            audit=audit
                        )
                    )

                else:
                    audit["unparsed_internal_tx"] += 1
                    out.append(
                        "[UNPARSED_INTERNAL_TX]"
                    )

            out.append("[INXsEND]")

        elif field == "calls":
            out.append("[CALLS_START]")

            out.extend(
                tokenize_calls(
                    value, audit
                )
            )

            out.append("[CALLS_END]")

        else:
            out.extend(
                (
                    field,
                    _source_token(
                        value, field
                    )
                )
            )

    if not nested:
        out.append("[END]")

    return out


# ============================================================
# PUBLIC TOKENIZER
# ============================================================

def tokenizer(dataFrame, audit=None):
    audit = (
        Counter() if audit is None else audit
    )

    out = []

    for _, row in dataFrame.iterrows():
        out.extend(
            _tokenize_record(
                row.to_dict(),
                nested=False,
                audit=audit
            )
        )

    return [str(t) for t in out]


# ============================================================
# TREE STREAM
# ============================================================

def flatten_tokens(x):
    flat = []

    for el in x:
        if isinstance(el, list):
            flat.extend(flatten_tokens(el))
        else:
            flat.append(str(el))

    return flat


def build_tree_from_output(output):
    tokens = flatten_tokens(output)

    tree = []
    stack = ["0"]
    current = "0"

    for i, tok in enumerate(tokens):
        tree.append(current)

        if (
            i > 0
            and tokens[i - 1] == "callId"
        ):
            new_id = tok
            stack.append(new_id)
            current = new_id

        if tok == "[CALL_END]":
            if len(stack) > 1:
                stack.pop()

            current = stack[-1]

    return tokens, tree


# ============================================================
# CONTEXT STREAM
# ============================================================

def build_context_from_tokens(tokens):
    context = []

    for i, tok in enumerate(tokens):
        if i > 0 and tokens[i - 1] == "to":
            context.append("TO")

        elif i > 0 and tokens[i - 1] == "from":
            context.append("FROM")

        else:
            context.append("NONE")

    return context
