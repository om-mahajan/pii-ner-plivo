import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np

from dataset import PIIDataset, collate_batch
from labels import LABELS, PII_LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="huawei-noah/TinyBERT_General_4L_312D", help="Model name (default: TinyBERT v5 optimized)")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16 for optimal performance)")
    ap.add_argument("--epochs", type=int, default=12, help="Number of epochs (default: 12 for TinyBERT v5)")
    ap.add_argument("--lr", type=float, default=3.5e-5, help="Learning rate (default: 3.5e-5 for TinyBERT v5)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=None, help="Override model dropout rate (e.g., 0.1 for lower dropout)")
    ap.add_argument("--use_class_weights", action="store_true", help="Use class weights for PII precision")
    ap.add_argument("--pii_weight", type=float, default=2.0, help="Weight multiplier for PII classes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name, dropout=args.dropout)
    model.to(args.device)
    
    if args.dropout is not None:
        print(f"Using custom dropout rate: {args.dropout}")
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        weights = torch.ones(len(LABELS))
        for i, label in enumerate(LABELS):
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label.split("-", 1)[1]
                if entity_type in PII_LABELS:
                    weights[i] = args.pii_weight
        class_weights = weights.to(args.device)
        print(f"Using class weights (PII weight={args.pii_weight})")
    
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            if class_weights is not None:
                # Use custom weighted loss
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
