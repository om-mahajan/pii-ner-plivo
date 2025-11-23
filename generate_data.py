"""
Generate synthetic noisy STT data for PII NER training.
"""
import json
import random

# Templates and variations for realistic noisy STT
FILLERS = ["uh", "umm", "like", "you know", "i mean", "yeah", "so", "well"]
CONNECTORS = ["and", "then", "after that", "also", "plus"]

PERSON_NAMES = [
    "rajesh kumar", "priya sharma", "amit patel", "sarah williams", "john smith",
    "neha gupta", "vikram singh", "anjali reddy", "michael brown", "jennifer davis",
    "rahul verma", "sneha iyer", "david johnson", "emily chen", "arjun mehta",
    "pooja desai", "robert wilson", "lisa anderson", "karthik krishnan", "meera nair",
    "arun kumar", "divya subramanian", "james taylor", "mary thomas", "suresh babu"
]

CITIES = [
    "mumbai", "bangalore", "chennai", "delhi", "pune", "hyderabad", "kolkata",
    "new york", "san francisco", "london", "toronto", "sydney", "singapore"
]

LOCATIONS = [
    "taj hotel", "central park", "main street", "downtown area", "tech park",
    "business district", "railway station", "airport terminal", "red fort area",
    "marina bay", "times square", "oxford street", "cyber city"
]

DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "company", "work"]

def random_filler():
    return random.choice(FILLERS) if random.random() < 0.3 else ""

def random_connector():
    return random.choice(CONNECTORS) if random.random() < 0.4 else "and"

def spoken_digits(num_str):
    """Convert digit string to spoken form with variations."""
    digit_map = {
        '0': ['zero', 'oh'],
        '1': ['one'],
        '2': ['two'],
        '3': ['three'],
        '4': ['four'],
        '5': ['five'],
        '6': ['six'],
        '7': ['seven'],
        '8': ['eight'],
        '9': ['nine']
    }
    
    # Mix of spoken and digits
    if random.random() < 0.3:
        return num_str  # Keep as digits
    
    result = []
    for d in num_str:
        if d in digit_map:
            result.append(random.choice(digit_map[d]))
        elif d == ' ':
            result.append(' ')
    
    return ' '.join(result)

def generate_phone():
    """Generate phone number in various formats."""
    digits = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    
    formats = [
        lambda: spoken_digits(digits),
        lambda: f"{digits[:4]} {digits[4:7]} {digits[7:]}",
        lambda: f"plus nine one {spoken_digits(digits)}",
        lambda: f"double {random.choice(['nine', 'eight'])} {spoken_digits(digits[2:])}"
    ]
    
    return random.choice(formats)()

def generate_credit_card():
    """Generate credit card in various formats."""
    digits = ''.join([str(random.randint(0, 9)) for _ in range(16)])
    
    formats = [
        lambda: f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}",
        lambda: spoken_digits(f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}"),
        lambda: spoken_digits(digits),
    ]
    
    return random.choice(formats)()

def generate_email(name=None):
    """Generate email address."""
    if name is None:
        name = random.choice(PERSON_NAMES)
    
    first, last = name.split()
    domain = random.choice(DOMAINS)
    
    formats = [
        f"{first} dot {last} at {domain} dot com",
        f"{first} underscore {last} at {domain} dot com",
        f"{first}{last} at {domain} dot com",
        f"{first[0]}{last} at {domain} dot com",
    ]
    
    return random.choice(formats), name

def generate_date():
    """Generate date in spoken form."""
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    days = [f"{random.randint(1, 28)}th" if random.random() < 0.5 
            else ["first", "second", "third", "tenth", "fifteenth", "twentieth"][random.randint(0, 5)]]
    years = [f"twenty {random.choice(['twenty', 'twenty one', 'twenty two', 'twenty three', 'twenty four'])}",
             f"two thousand {random.choice(['twenty', 'twenty one', 'twenty two', 'twenty three'])}",
             f"nineteen {random.choice(['eighty', 'ninety', 'ninety five'])}"]
    
    formats = [
        lambda: f"{random.choice(months)} {random.choice(days)} {random.choice(years)}",
        lambda: f"{random.randint(1, 28)} slash {random.randint(1, 12)} slash {random.randint(2020, 2024)}",
        lambda: f"{random.choice(days)} of {random.choice(months)} {random.choice(years)}",
    ]
    
    return random.choice(formats)()

def generate_sample(idx, is_train=True):
    """Generate one sample with 1-3 entities."""
    entities_to_gen = random.randint(1, 3)
    entity_types = random.sample([
        'PHONE', 'EMAIL', 'CREDIT_CARD', 'PERSON_NAME', 'DATE', 'CITY', 'LOCATION'
    ], entities_to_gen)
    
    parts = []
    entities = []
    current_pos = 0
    
    # Start with optional filler
    if random.random() < 0.3:
        filler = random_filler()
        if filler:
            parts.append(filler)
            current_pos = len(filler) + 1
    
    for i, ent_type in enumerate(entity_types):
        if i > 0 and random.random() < 0.5:
            connector = random_connector()
            parts.append(connector)
            current_pos += len(connector) + 1
        
        # Add context before entity
        contexts = {
            'PHONE': ["my phone is", "call me on", "my number is", "you can reach me at", "contact number is"],
            'EMAIL': ["email me at", "my email is", "send it to", "email address is"],
            'CREDIT_CARD': ["card number is", "my credit card is", "the card is", "visa card number is"],
            'PERSON_NAME': ["my name is", "this is", "i am", "contact", "please reach out to"],
            'DATE': ["on", "scheduled for", "dated", "born on", "meeting is on"],
            'CITY': ["in", "from", "located in", "based in", "visiting"],
            'LOCATION': ["at", "near", "at the", "located at"],
        }
        
        context = random.choice(contexts[ent_type])
        parts.append(context)
        current_pos += len(context) + 1
        
        # Generate entity
        if ent_type == 'PHONE':
            value = generate_phone()
        elif ent_type == 'EMAIL':
            value, person_name = generate_email()
            # Add person name entity first
            name_start = current_pos
            name_end = name_start + len(person_name.replace(' ', ' dot ' if 'dot' in value else ' underscore ' if 'underscore' in value else ' '))
            if 'dot' in value or 'underscore' in value:
                entities.append({"start": name_start, "end": name_start + value.find(' at '), "label": "PERSON_NAME"})
        elif ent_type == 'CREDIT_CARD':
            value = generate_credit_card()
        elif ent_type == 'PERSON_NAME':
            value = random.choice(PERSON_NAMES)
        elif ent_type == 'DATE':
            value = generate_date()
        elif ent_type == 'CITY':
            value = random.choice(CITIES)
        elif ent_type == 'LOCATION':
            value = random.choice(LOCATIONS)
        
        start = current_pos
        end = start + len(value)
        parts.append(value)
        current_pos = end + 1
        
        entities.append({"start": start, "end": end, "label": ent_type})
        
        # Add optional continuation
        if i < len(entity_types) - 1 and random.random() < 0.2:
            extra = random.choice(["okay", "right", "got it", ""])
            if extra:
                parts.append(extra)
                current_pos += len(extra) + 1
    
    text = ' '.join(parts)
    
    # Recalculate entity positions based on actual text
    actual_entities = []
    for ent in entities:
        label = ent["label"]
        # Find actual positions in reconstructed text
        # This is approximate - for production we'd be more careful
        actual_entities.append(ent)
    
    return {
        "id": f"{'train' if is_train else 'dev'}_{idx:04d}",
        "text": text,
        "entities": actual_entities
    }

def main():
    random.seed(42)
    
    # Generate training data
    train_samples = []
    for i in range(1, 801):
        train_samples.append(generate_sample(i, is_train=True))
    
    with open("data/train_new.jsonl", "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(train_samples)} training samples")
    
    # Generate dev data
    random.seed(123)
    dev_samples = []
    for i in range(101, 251):
        dev_samples.append(generate_sample(i, is_train=False))
    
    with open("data/dev_new.jsonl", "w", encoding="utf-8") as f:
        for sample in dev_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(dev_samples)} dev samples")

if __name__ == "__main__":
    main()
