"""
Preprocessing utilities for noisy STT transcripts.
Handles spoken numbers, symbols, and maintains character offset mapping.
"""
import re
from typing import Tuple, List


# Spoken digit mappings
SPOKEN_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "oh": "0",  # common in phone numbers
}

# Spoken symbol mappings
SPOKEN_SYMBOLS = {
    " at ": "@",
    " dot ": ".",
    " dash ": "-",
    " slash ": "/",
    " underscore ": "_",
    " hyphen ": "-",
}


def normalize_spoken_text(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Normalize spoken numbers and symbols to their written forms.
    Returns (normalized_text, char_mapping) where char_mapping[i] = original_char_index for normalized_text[i].
    
    This preserves the ability to map predictions back to original character offsets.
    """
    # Build character-level mapping
    char_map = list(range(len(text)))  # Initially 1:1
    normalized = text.lower()
    
    # First pass: normalize spoken symbols (at, dot, etc.)
    for spoken, symbol in SPOKEN_SYMBOLS.items():
        parts = []
        new_char_map = []
        pos = 0
        
        for match in re.finditer(re.escape(spoken), normalized):
            start, end = match.span()
            # Keep text before match
            parts.append(normalized[pos:start])
            new_char_map.extend(char_map[pos:start])
            
            # Replace with symbol (map all chars in symbol to first char of spoken form)
            parts.append(symbol)
            new_char_map.extend([char_map[start]] * len(symbol))
            
            pos = end
        
        if parts:  # If we found matches
            parts.append(normalized[pos:])
            new_char_map.extend(char_map[pos:])
            normalized = "".join(parts)
            char_map = new_char_map
    
    # Second pass: normalize spoken digits
    # Pattern: match sequences of spoken digits with optional spaces
    digit_pattern = r'\b(' + '|'.join(SPOKEN_DIGITS.keys()) + r')(?:\s+(' + '|'.join(SPOKEN_DIGITS.keys()) + r'))*\b'
    
    parts = []
    new_char_map = []
    pos = 0
    
    for match in re.finditer(digit_pattern, normalized):
        start, end = match.span()
        # Keep text before match
        parts.append(normalized[pos:start])
        new_char_map.extend(char_map[pos:start])
        
        # Extract all spoken digits in this match
        spoken_seq = match.group(0)
        digits = []
        digit_starts = []
        
        for word in spoken_seq.split():
            if word in SPOKEN_DIGITS:
                digits.append(SPOKEN_DIGITS[word])
                # Find position of this word in original match
                word_start = normalized.find(word, start)
                digit_starts.append(char_map[word_start] if word_start < len(char_map) else char_map[start])
        
        # Replace with digit sequence
        digit_str = "".join(digits)
        parts.append(digit_str)
        
        # Map each output digit to the start of its corresponding spoken word
        for i, digit in enumerate(digit_str):
            if i < len(digit_starts):
                new_char_map.append(digit_starts[i])
            else:
                new_char_map.append(digit_starts[-1] if digit_starts else char_map[start])
        
        pos = end
    
    parts.append(normalized[pos:])
    new_char_map.extend(char_map[pos:])
    normalized = "".join(parts)
    char_map = new_char_map
    
    return normalized, char_map


def map_span_to_original(start: int, end: int, char_mapping: List[int]) -> Tuple[int, int]:
    """
    Map a span in normalized text back to original text character offsets.
    """
    if start >= len(char_mapping):
        return (0, 0)
    
    orig_start = char_mapping[start]
    orig_end = char_mapping[min(end - 1, len(char_mapping) - 1)] + 1
    
    return orig_start, orig_end


def simple_normalize(text: str) -> str:
    """
    Simple normalization without offset tracking (for quick testing).
    """
    normalized, _ = normalize_spoken_text(text)
    return normalized
