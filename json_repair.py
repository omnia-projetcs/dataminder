"""Repair helpers for malformed LLM JSON (Q&A arrays and enrichment objects)."""

from __future__ import annotations

import json
import re


def _sanitize_json_string(raw_json):
    """Sanitize a JSON string that may contain literal control characters
    (newlines, tabs, etc.) inside string values, which is invalid JSON.
    This replaces unescaped control characters with their escape sequences."""
    # Replace literal control characters that break JSON parsing
    # We need to be careful: \n that is already escaped (\\n in the raw string) should stay.
    # Strategy: process character by character tracking if we're inside a JSON string value.
    
    result = []
    in_string = False
    i = 0
    while i < len(raw_json):
        ch = raw_json[i]
        
        if ch == '\\' and in_string and i + 1 < len(raw_json):
            # Escaped character inside a string — keep both characters as-is
            result.append(ch)
            result.append(raw_json[i + 1])
            i += 2
            continue
        
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        
        if in_string:
            # Replace literal control characters with their JSON escape sequences
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            elif ord(ch) < 32:
                result.append(f'\\u{ord(ch):04x}')
            else:
                result.append(ch)
        else:
            result.append(ch)
        
        i += 1
    
    return ''.join(result)


def _extract_balanced_json_array(content):
    """Return the first top-level JSON array, ignoring brackets inside strings."""
    start = content.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    i = start
    while i < len(content):
        ch = content[i]
        if ch == "\\" and in_str:
            i += 2
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1]
        i += 1
    return None


def _try_parse_json(content):
    """Try multiple strategies to extract a JSON array from LLM output."""
    
    # Strategy 1: Extract JSON from markdown code block ```json ... ```
    code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        sanitized = _sanitize_json_string(json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: First balanced [...] array (avoids swallowing later brackets)
    json_str = _extract_balanced_json_array(content)
    if json_str:
        sanitized = _sanitize_json_string(json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try direct parse of the full content
    sanitized = _sanitize_json_string(content)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Line-by-line recovery — extract individual {"question":..., "answer":...} objects
    pairs = []
    for obj_match in re.finditer(r'\{[^{}]*?"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', content, re.DOTALL):
        try:
            q = obj_match.group(1).replace('\\n', ' ').replace('\\t', ' ').strip()
            a = obj_match.group(2).replace('\\n', ' ').replace('\\t', ' ').strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
        except Exception:
            continue
    
    return pairs if pairs else None


def _try_parse_enrichment_json(content):
    """Try multiple strategies to extract a JSON object with 'input' and 'output' keys from LLM output."""

    def _attempt_parse(json_str):
        """Try parsing with sanitization and invalid-escape recovery."""
        # First try with sanitization
        sanitized = _sanitize_json_string(json_str)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        # Fallback: strip invalid backslash escapes (e.g. \S, \x without valid hex, etc.)
        cleaned = re.sub(r'\\(?!["\\\\bfnrtu])', r'\\\\', sanitized)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fallback 2: aggressively strip ALL backslash sequences that aren't standard JSON escapes
        aggressive = re.sub(r'\\(?!["\\\\bfnrtu/])', '', sanitized)
        try:
            return json.loads(aggressive)
        except json.JSONDecodeError:
            pass
        return None

    def _attempt_parse_truncated(json_str):
        """Try to repair and parse truncated JSON by closing open structures."""
        # Try the normal parse first
        result = _attempt_parse(json_str)
        if result is not None:
            return result

        # Attempt to close truncated JSON: track open braces/brackets/strings
        sanitized = _sanitize_json_string(json_str)
        # Strip trailing incomplete string values (e.g. cut mid-sentence)
        # Try progressively trimming from the end to find a parseable prefix
        for trim_target in ['\n', '.', ',', ' ']:
            last_pos = sanitized.rfind(trim_target)
            while last_pos > len(sanitized) // 2:
                candidate = sanitized[:last_pos]
                # Close any open string
                if candidate.count('"') % 2 != 0:
                    candidate += '"'
                # Close open braces/brackets
                open_braces = candidate.count('{') - candidate.count('}')
                open_brackets = candidate.count('[') - candidate.count(']')
                candidate += ']' * max(0, open_brackets)
                candidate += '}' * max(0, open_braces)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                # Also try with aggressive escape stripping
                aggressive = re.sub(r'\\(?!["\\\\bfnrtu/])', '', candidate)
                try:
                    return json.loads(aggressive)
                except json.JSONDecodeError:
                    pass
                last_pos = sanitized.rfind(trim_target, 0, last_pos)
        return None

    def _validate_enrichment(result):
        """Check that a parsed dict has usable 'input' and 'output' keys."""
        if not isinstance(result, dict):
            return None
        if "input" in result and "output" in result:
            return result
        # Sometimes the LLM nests the data one level deep (e.g. {"result": {"input": ..., "output": ...}})
        for v in result.values():
            if isinstance(v, dict) and "input" in v and "output" in v:
                return v
        return None

    # Strategy 1: Extract JSON from markdown code block ```json ... ``` (non-greedy for inner braces)
    code_block_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
    if code_block_match:
        result = _attempt_parse(code_block_match.group(1))
        validated = _validate_enrichment(result)
        if validated:
            return validated

    # Strategy 1b: Strip opening ```json fence when closing ``` is missing (truncated response)
    fence_open_match = re.search(r'```(?:json)?\s*(\{.*)', content, re.DOTALL)
    if fence_open_match:
        stripped = fence_open_match.group(1).rstrip('`').rstrip()
        result = _attempt_parse(stripped)
        validated = _validate_enrichment(result)
        if validated:
            return validated
        # Try truncated repair on the fence-stripped content
        result = _attempt_parse_truncated(stripped)
        validated = _validate_enrichment(result)
        if validated:
            return validated

    # Strategy 2: Find outermost { ... } with balanced brace matching (skip braces inside strings)
    start = content.find('{')
    if start != -1:
        depth = 0
        end = start
        in_str = False
        i = start
        while i < len(content):
            ch = content[i]
            if ch == '\\' and in_str:
                i += 2  # skip escaped char
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1
        if end > start:
            result = _attempt_parse(content[start:end])
            validated = _validate_enrichment(result)
            if validated:
                return validated

    # Strategy 3: Simple first-{-to-last-} extraction (fallback for nested issues)
    last = content.rfind('}')
    if start != -1 and last >= start:
        result = _attempt_parse(content[start:last + 1])
        validated = _validate_enrichment(result)
        if validated:
            return validated

    # Strategy 4: Regex extraction of individual fields when JSON structure is broken
    # Try to find "input" and "output" values even if the overall JSON is malformed
    input_match = re.search(r'"input"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    output_match = re.search(r'"output"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if input_match and output_match:
        try:
            inp = input_match.group(1).replace('\\n', '\n').replace('\\t', ' ')
            out = output_match.group(1).replace('\\n', '\n').replace('\\t', ' ')
            if inp.strip() and out.strip():
                return {"input": inp, "output": out}
        except Exception:
            pass

    # Strategy 5: "output" might be a nested JSON object rather than a string
    if input_match:
        output_obj_match = re.search(r'"output"\s*:\s*(\{.*)', content, re.DOTALL)
        if output_obj_match:
            obj_str = output_obj_match.group(1)
            # Find balanced braces
            depth = 0
            for i, ch in enumerate(obj_str):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        result = _attempt_parse(obj_str[:i+1])
                        if isinstance(result, dict):
                            inp = input_match.group(1).replace('\\n', '\n').replace('\\t', ' ')
                            return {"input": inp, "output": result}
                        break

    # Strategy 6: "input" is a nested JSON object (not a string), "output" may also be nested
    input_obj_match = re.search(r'"input"\s*:\s*(\{)', content, re.DOTALL)
    if input_obj_match:
        # Extract balanced input object
        obj_start = input_obj_match.start(1)
        depth = 0
        in_s = False
        inp_end = obj_start
        j = obj_start
        while j < len(content):
            ch = content[j]
            if ch == '\\' and in_s:
                j += 2
                continue
            if ch == '"':
                in_s = not in_s
            elif not in_s:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        inp_end = j + 1
                        break
            j += 1
        if inp_end > obj_start:
            inp_result = _attempt_parse(content[obj_start:inp_end])
            if isinstance(inp_result, dict):
                # Now find "output" after input
                remainder = content[inp_end:]
                out_obj_match = re.search(r'"output"\s*:\s*(\{)', remainder, re.DOTALL)
                if out_obj_match:
                    out_start = out_obj_match.start(1)
                    depth = 0
                    in_s = False
                    out_end = out_start
                    k = out_start
                    while k < len(remainder):
                        ch = remainder[k]
                        if ch == '\\' and in_s:
                            k += 2
                            continue
                        if ch == '"':
                            in_s = not in_s
                        elif not in_s:
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    out_end = k + 1
                                    break
                        k += 1
                    if out_end > out_start:
                        out_result = _attempt_parse(remainder[out_start:out_end])
                        if out_result is not None:
                            return {"input": inp_result, "output": out_result}

    # Strategy 7: Last resort — truncated JSON repair on the full content from first {
    if start != -1:
        result = _attempt_parse_truncated(content[start:])
        validated = _validate_enrichment(result)
        if validated:
            return validated

    return None
