import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii, PII_LABELS
from preprocessing import normalize_spoken_text, map_span_to_original
import os


def bio_to_spans(text, offsets, label_ids, confidences=None, pii_threshold=0.0, non_pii_threshold=0.0):
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_conf = []

    for i, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        conf = confidences[i] if confidences is not None else 1.0
        
        if label == "O":
            if current_label is not None:
                avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
                threshold = pii_threshold if current_label in PII_LABELS else non_pii_threshold
                if avg_conf >= threshold:
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_conf = []
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
                threshold = pii_threshold if current_label in PII_LABELS else non_pii_threshold
                if avg_conf >= threshold:
                    spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
            current_conf = [conf]
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                current_conf.append(conf)
            else:
                if current_label is not None:
                    avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
                    threshold = pii_threshold if current_label in PII_LABELS else non_pii_threshold
                    if avg_conf >= threshold:
                        spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
                current_conf = [conf]

    if current_label is not None:
        avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
        threshold = pii_threshold if current_label in PII_LABELS else non_pii_threshold
        if avg_conf >= threshold:
            spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--pii_threshold", type=float, default=0.5, help="Confidence threshold for PII entities (precision focus)")
    ap.add_argument("--non_pii_threshold", type=float, default=0.3, help="Confidence threshold for non-PII entities")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            original_text = obj["text"]
            uid = obj["id"]
            
            # Disable normalization for now (causing span issues)
            text = original_text
            char_mapping = list(range(len(text)))

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                probs = F.softmax(logits, dim=-1)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                confidences = probs.max(dim=-1).values.cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids, confidences, 
                                args.pii_threshold, args.non_pii_threshold)
            ents = []
            for s, e, lab in spans:
                # Map back to original text offsets
                orig_s, orig_e = map_span_to_original(s, e, char_mapping)
                ents.append(
                    {
                        "start": int(orig_s),
                        "end": int(orig_e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
