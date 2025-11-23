"""
Quick tests for preprocessing normalization.
"""
from preprocessing import normalize_spoken_text, map_span_to_original


def test_normalization():
    print("Testing spoken number normalization:")
    
    test_cases = [
        "four two four two",
        "nine eight seven six five four three two one zero",
        "call me on nine eight seven",
        "email is john at gmail dot com",
        "card number is four two four two four two four two",
        "my phone is oh nine seven six",
    ]
    
    for text in test_cases:
        normalized, char_map = normalize_spoken_text(text)
        print(f"\nOriginal:   '{text}'")
        print(f"Normalized: '{normalized}'")
        print(f"Char map length: {len(char_map)} (normalized text length: {len(normalized)})")
        
        # Test span mapping back
        if len(normalized) >= 5:
            test_span = (0, 5)
            orig_span = map_span_to_original(*test_span, char_map)
            print(f"Test span [0:5] '{normalized[0:5]}' → original [{orig_span[0]}:{orig_span[1]}] '{text[orig_span[0]:orig_span[1]]}'")


def test_real_example():
    print("\n" + "="*60)
    print("Testing with real training example:")
    
    text = "my credit card number is 4242 4242 4242 4242 and my email is ramesh dot sharma at gmail dot com"
    normalized, char_map = normalize_spoken_text(text)
    
    print(f"Original:   '{text}'")
    print(f"Normalized: '{normalized}'")
    
    # Test mapping for "gmail dot com" → "gmail.com"
    orig_email_start = text.find("ramesh dot sharma at gmail dot com")
    orig_email_end = orig_email_start + len("ramesh dot sharma at gmail dot com")
    
    print(f"\nOriginal email span: [{orig_email_start}:{orig_email_end}]")
    print(f"Original email text: '{text[orig_email_start:orig_email_end]}'")
    
    norm_email_start = normalized.find("ramesh.sharma@gmail.com")
    norm_email_end = norm_email_start + len("ramesh.sharma@gmail.com")
    
    if norm_email_start != -1:
        print(f"Normalized email span: [{norm_email_start}:{norm_email_end}]")
        print(f"Normalized email text: '{normalized[norm_email_start:norm_email_end]}'")
        
        # Map back
        mapped_start, mapped_end = map_span_to_original(norm_email_start, norm_email_end, char_map)
        print(f"Mapped back span: [{mapped_start}:{mapped_end}]")
        print(f"Mapped back text: '{text[mapped_start:mapped_end]}'")


if __name__ == "__main__":
    test_normalization()
    test_real_example()
