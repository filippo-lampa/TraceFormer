import ast
import re
import pandas as pd

#global variable used to avoid recursing infinitely when tokenizing internal transactions
recursing = 0

def setRecursing(r):
    global recursing
    recursing = r


# normalization helpers
_ADDR_0X = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ADDR_NO0X = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_0X = re.compile(r"^0x[0-9a-fA-F]+$")
_HEX32_0X = re.compile(r"^0x[0-9a-fA-F]{64}$")

_BIGINT_DIGITS_CUTOFF = 60
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

def normalize_token(value, field_name=None):
    """
    replace high-cardinality identifiers with special tokens to shrink vocab
    """

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
        
        # scientific notation strings
        sci_pattern = re.compile(r'^[-+]?\d*\.?\d+[eE][-+]?\d+$')
        if sci_pattern.match(s):
            return "[SCI_NOTATION]"
        
        # ISO timestamps
        iso_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
        if iso_pattern.match(s):
            return "[TIMESTAMP_ISO]"
        
        # comma-separated hex values (multiple tx data)
        if ',' in s and '0x' in s:
            hex_parts = s.split(',')
            all_hex = all(part.strip().startswith('0x') for part in hex_parts)
            if all_hex and len(hex_parts) > 1:
                return "[MULTI_HEX]"
        
        # single hex values (addresses, tx hashes, ...)
        if field_name == "transactionHash" and _HEX32_0X.match(s):
            return "[TXHASH]"
        
        if _ADDR_0X.match(s) or _ADDR_NO0X.match(s):
            return "[ADDR]"
        
        if _HEX32_0X.match(s):
            return "[HEX32]"
        
        if _HEX_0X.match(s):
            return "[HEX]"
        
        # numeric strings
        if s.isdigit() or (s[0] == '-' and s[1:].isdigit()):
            try:
                num = int(s)
                if abs(num) > 1_000_000_000:
                    digit_count = len(s.lstrip('-'))
                    if digit_count <= 12:
                        return f"[INT_{digit_count}D]"
                    else:
                        return "[INT_HUGE]"
                return _safe_int_token(num)
            except (ValueError, OverflowError):
                return "[NUM_STR]"
        
        # date-like strings without time
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        if date_pattern.match(s):
            return "[DATE]"
        
        # default: keep as is
        return s
    
    # lists
    if isinstance(value, list):
        return f"[LIST_{len(value)}]"

    # fallback for other objects: keep type but do not explode
    return f"[{type(value).__name__.upper()}]"


def _maybe_parse_literal(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return x
    return x


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return [x]


def tokenize_calls(cell_value):
    out = []
    if isinstance(cell_value, str):
        s = cell_value.strip()
        if s == "" or s == "[]":
            return out
        try:
            calls_list = ast.literal_eval(s)
        except Exception:
            calls_list = []
    else:
        calls_list = cell_value

    if not isinstance(calls_list, list):
        return out

    for call in calls_list:
        if not isinstance(call, dict):
            continue
        out.append("[CALLSTART]")

        for k, v in call.items():
            if k == "calls":
                continue

            if k == "inputs":
                out.append("[INsSTART]")
                inputs_list = v
                if isinstance(v, str):
                    try:
                        inputs_list = ast.literal_eval(v)
                    except Exception:
                        inputs_list = []

                if isinstance(inputs_list, dict):
                    inputs_list = [inputs_list]
                if not isinstance(inputs_list, list):
                    inputs_list = []

                for d in inputs_list:
                    if isinstance(d, dict):
                        for kk, vv in d.items():
                            kk_s = str(kk)
                            out.append(kk_s)
                            out.append(str(normalize_token(vv, field_name=kk_s)))
                out.append("[INsEND]")

            else:
                kk_s = str(k)
                out.append(kk_s)
                out.append(str(normalize_token(v, field_name=kk_s)))

        if "calls" in call and call["calls"]:
            out.append("[CALLS_CHILD_START]")
            out.extend(tokenize_calls(call["calls"]))
            out.append("CALLS_CHILD_END")

        out.append("[CALL_END]")
    return out


def tokenizer(dataFrame):
    global recursing
    tokenizerOutput = []

    for index, row_series in dataFrame.iterrows():
        if recursing == 0:
            tokenizerOutput.append("[START]")

        for column_name, cell_value in row_series.items():
            col = str(column_name)

            if column_name == "inputs":
                tokenizerOutput.append("[INsSTART]")
                inputs_val = _maybe_parse_literal(cell_value)
                for d in _as_list(inputs_val):
                    d = _maybe_parse_literal(d)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            kk = str(k)
                            tokenizerOutput.append(kk)
                            tokenizerOutput.append(str(normalize_token(v, field_name=kk)))
                    else:
                        tokenizerOutput.append(str(normalize_token(d, field_name=col)))
                tokenizerOutput.append("[INsEND]")

            elif column_name == "timestamp":
                ts = _maybe_parse_literal(cell_value)
                tokenizerOutput.append("timestamp")
                if isinstance(ts, dict) and len(ts) > 0:
                    tokenizerOutput.append(str(normalize_token(next(iter(ts.values())), field_name="timestamp")))
                else:
                    tokenizerOutput.append(str(normalize_token(ts, field_name="timestamp")))

            elif column_name == "internalTxs":
                tokenizerOutput.append("internalTxs")
                tokenizerOutput.append("[INXsSTART]")
                internal_val = _maybe_parse_literal(cell_value)
                internal_list = _as_list(internal_val)
                setRecursing(1)
                try:
                    tokenizerOutput.extend(tokenizer(pd.DataFrame(internal_list)))
                finally:
                    setRecursing(0)
                tokenizerOutput.append("[INXsEND]")

            elif column_name == "calls":
                tokenizerOutput.append("[CALLS_START]")
                tokenizerOutput.extend(tokenize_calls(cell_value))
                tokenizerOutput.append("[CALLS_END]")

            elif column_name == "events":
                tokenizerOutput.append("events")
                tokenizerOutput.append("[EVsSTART]")
                events_val = _maybe_parse_literal(cell_value)
                for ev in _as_list(events_val):
                    ev = _maybe_parse_literal(ev)
                    if not isinstance(ev, dict):
                        tokenizerOutput.append(str(normalize_token(ev, field_name="events")))
                        continue
                    for k, v in ev.items():
                        kk = str(k)
                        if k == "eventValues":
                            tokenizerOutput.append("eventValues")
                            ev_vals = _maybe_parse_literal(v)
                            if isinstance(ev_vals, dict):
                                for k2, v2 in ev_vals.items():
                                    k2s = str(k2)
                                    tokenizerOutput.append(k2s)
                                    tokenizerOutput.append(str(normalize_token(v2, field_name=k2s)))
                            else:
                                tokenizerOutput.append(str(normalize_token(ev_vals, field_name="eventValues")))
                        else:
                            tokenizerOutput.append(kk)
                            tokenizerOutput.append(str(normalize_token(v, field_name=kk)))
                tokenizerOutput.append("[EVsEND]")

            else:
                tokenizerOutput.append(col)
                tokenizerOutput.append(str(normalize_token(cell_value, field_name=col)))

        if recursing == 0:
            tokenizerOutput.append("[END]")
    return tokenizerOutput


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
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        tree.append(current)
        if i > 0 and tokens[i - 1] == "callId":
            new_id = tok
            stack.append(new_id)
            current = new_id
        if tok == "[CALL_END]":
            if len(stack) > 1:
                stack.pop()
            current = stack[-1]
        i += 1
    return tokens, tree


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
