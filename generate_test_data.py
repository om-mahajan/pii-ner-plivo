"""
Generate test dataset for final evaluation.
Creates 200 test samples with noisy STT patterns.
"""
import json
import random
from generate_data import generate_sample

def generate_test_dataset(num_samples=200, seed=999):
    """Generate test dataset using same logic as train/dev."""
    random.seed(seed)
    samples = []
    
    for i in range(num_samples):
        sample = generate_sample(i, is_train=False)
        samples.append(sample)
    
    return samples


def main():
    print("Generating test dataset...")
    test_samples = generate_test_dataset(num_samples=200, seed=999)
    
    output_path = "data/test.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(test_samples)} test samples")
    print(f"Saved to {output_path}")
    
    # Print statistics
    entity_counts = {}
    for sample in test_samples:
        for entity in sample['entities']:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print("\nEntity distribution:")
    for label, count in sorted(entity_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
