"""
SemEval-2026 Task 13 - Perplexity Feature Extraction
=====================================================
Tính Perplexity (Cross-Entropy Loss) cho mỗi code snippet bằng LLM.

Yêu cầu:
- GPU với >= 4GB VRAM (hoặc CPU nhưng rất chậm)
- pip install torch transformers pandas pyarrow tqdm

Cách chạy:
    python extract_perplexity.py --input ./task_A/train.parquet --output ./task_A/train_perplexity.parquet
    python extract_perplexity.py --input ./task_A/validation.parquet --output ./task_A/val_perplexity.parquet

Tùy chọn:
    --model       Model name (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
    --max_length  Max token length (default: 512)
    --batch_size  Batch size for processing (default: 1)
    --device      Device: cuda / cpu / auto (default: auto)
    --sample      Chỉ xử lý N mẫu đầu tiên (để test)
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Perplexity features from code snippets using LLM")
    parser.add_argument("--input", type=str, default="./task_A/train.parquet",
                        help="Input parquet file path")
    parser.add_argument("--output", type=str, default="./task_A/train_perplexity.parquet",
                        help="Output parquet file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HuggingFace model name for perplexity calculation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for truncation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for parallel processing on GPU")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda / cpu / auto")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N samples (for testing)")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N samples")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: skip per-line perplexity (10-30x faster)")
    return parser.parse_args()


class PerplexityExtractor:
    """Compute code perplexity using a causal LM."""
    
    def __init__(self, model_name: str, device: str, max_length: int = 512):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model: {model_name} on {device}...")
        dtype = torch.float16 if "cuda" in device else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        logger.info(f"Model loaded successfully! dtype={dtype}, device={device}")
    
    @torch.no_grad()
    def compute_perplexity(self, code: str) -> float:
        """Compute cross-entropy loss (log-perplexity) for a code snippet."""
        if not code or not code.strip():
            return 0.0
        
        try:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = inputs.input_ids.to(self.model.device)
            
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            return outputs.loss.item()
        except Exception as e:
            logger.warning(f"Error computing perplexity: {e}")
            return 0.0
    
    @torch.no_grad()
    def compute_perplexity_batch(self, codes: list) -> list:
        """Compute cross-entropy loss for a batch of code snippets (fast path)."""
        # Replace empty strings with a single space to avoid tokenizer issues
        codes = [c if c and c.strip() else " " for c in codes]

        enc = self.tokenizer(
            codes,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        input_ids = enc.input_ids.to(self.model.device)
        attn = enc.attention_mask.to(self.model.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits  # (B, T, V)

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().float()

        # Per-token NLL
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        nll = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())  # (B, T-1)

        # Average per sample using mask
        masked_nll = nll * shift_mask
        token_counts = shift_mask.sum(dim=1).clamp(min=1)
        mean_nll = (masked_nll.sum(dim=1) / token_counts).cpu().tolist()
        return mean_nll

    @torch.no_grad()
    def compute_line_perplexities(self, code: str) -> dict:
        """
        Compute perplexity metrics at multiple granularities:
        - overall_ppl: Perplexity of the whole snippet
        - line_ppl_mean: Mean of per-line perplexities
        - line_ppl_std: Std of per-line perplexities (burstiness proxy)
        - line_ppl_max: Max per-line perplexity
        - line_ppl_min: Min per-line perplexity
        - ppl_variance: Variance of per-line perplexities
        """
        if not code or not code.strip():
            return {
                'overall_ppl': 0.0,
                'line_ppl_mean': 0.0,
                'line_ppl_std': 0.0,
                'line_ppl_max': 0.0,
                'line_ppl_min': 0.0,
                'ppl_variance': 0.0,
            }
        
        # Overall perplexity
        overall_ppl = self.compute_perplexity(code)
        
        # Per-line perplexities
        lines = [l for l in code.split('\n') if l.strip()]
        
        if len(lines) < 2:
            return {
                'overall_ppl': overall_ppl,
                'line_ppl_mean': overall_ppl,
                'line_ppl_std': 0.0,
                'line_ppl_max': overall_ppl,
                'line_ppl_min': overall_ppl,
                'ppl_variance': 0.0,
            }
        
        # Sample up to 20 lines to keep it fast
        if len(lines) > 20:
            indices = np.linspace(0, len(lines)-1, 20, dtype=int)
            sampled_lines = [lines[i] for i in indices]
        else:
            sampled_lines = lines
        
        line_ppls = []
        for line in sampled_lines:
            if len(line.strip()) > 3:  # Skip very short lines
                ppl = self.compute_perplexity(line)
                if ppl > 0:
                    line_ppls.append(ppl)
        
        if not line_ppls:
            return {
                'overall_ppl': overall_ppl,
                'line_ppl_mean': overall_ppl,
                'line_ppl_std': 0.0,
                'line_ppl_max': overall_ppl,
                'line_ppl_min': overall_ppl,
                'ppl_variance': 0.0,
            }
        
        return {
            'overall_ppl': overall_ppl,
            'line_ppl_mean': float(np.mean(line_ppls)),
            'line_ppl_std': float(np.std(line_ppls)),
            'line_ppl_max': float(np.max(line_ppls)),
            'line_ppl_min': float(np.min(line_ppls)),
            'ppl_variance': float(np.var(line_ppls)),
        }


def main():
    args = parse_args()
    
    # =========================================================================
    # 1. Setup Device
    # =========================================================================
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # =========================================================================
    # 2. Load Data
    # =========================================================================
    logger.info(f"Loading input: {args.input}")
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} samples, columns: {df.columns.tolist()}")
    
    if 'code' not in df.columns:
        logger.error("Column 'code' not found in dataframe!")
        sys.exit(1)
    
    if args.sample:
        df = df.head(args.sample)
        logger.info(f"Sampling first {args.sample} samples for testing")
    
    # =========================================================================
    # 3. Check for existing checkpoint
    # =========================================================================
    checkpoint_path = args.output + ".checkpoint.parquet"
    start_idx = 0
    results = []
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint at {checkpoint_path}")
        checkpoint_df = pd.read_parquet(checkpoint_path)
        start_idx = len(checkpoint_df)
        results = checkpoint_df.to_dict('records')
        logger.info(f"Resuming from index {start_idx}")
    
    # =========================================================================
    # 4. Initialize Extractor
    # =========================================================================
    extractor = PerplexityExtractor(
        model_name=args.model,
        device=device,
        max_length=args.max_length
    )
    
    # =========================================================================
    # 5. Extract Perplexity Features
    # =========================================================================
    logger.info(f"\nExtracting perplexity features for {len(df) - start_idx} remaining samples...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max length: {args.max_length} tokens")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Fast mode: {args.fast} (skip per-line PPL)")
    logger.info(f"Save checkpoint every {args.save_every} samples")

    start_time = time.time()
    codes_all = df['code'].tolist()

    if args.fast:
        # Fast path: batched overall perplexity only
        idx = start_idx
        pbar = tqdm(total=len(df), initial=start_idx, desc="Perplexity")
        while idx < len(df):
            batch_codes = codes_all[idx: idx + args.batch_size]
            try:
                ppls = extractor.compute_perplexity_batch(batch_codes)
            except Exception as e:
                logger.warning(f"Batch error at index {idx}: {e}. Falling back per-sample.")
                ppls = []
                for c in batch_codes:
                    try:
                        ppls.append(extractor.compute_perplexity(c))
                    except Exception:
                        ppls.append(0.0)

            for p in ppls:
                results.append({'overall_ppl': float(p)})

            idx += len(batch_codes)
            pbar.update(len(batch_codes))

            if idx % args.save_every < args.batch_size:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_parquet(checkpoint_path, index=False)
                elapsed = time.time() - start_time
                speed = (idx - start_idx) / max(elapsed, 0.001)
                remaining = (len(df) - idx) / max(speed, 0.001)
                logger.info(f"Checkpoint saved at {idx}/{len(df)} "
                           f"({speed:.1f} samples/sec, ~{remaining/60:.0f} min remaining)")
        pbar.close()
    else:
        # Original per-sample path (with per-line PPL)
        for idx in tqdm(range(start_idx, len(df)), desc="Perplexity", initial=start_idx, total=len(df)):
            code = codes_all[idx]
            try:
                ppl_features = extractor.compute_line_perplexities(code)
            except Exception as e:
                logger.warning(f"Error at index {idx}: {e}")
                ppl_features = {
                    'overall_ppl': 0.0,
                    'line_ppl_mean': 0.0,
                    'line_ppl_std': 0.0,
                    'line_ppl_max': 0.0,
                    'line_ppl_min': 0.0,
                    'ppl_variance': 0.0,
                }
            results.append(ppl_features)

            if (idx + 1) % args.save_every == 0:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_parquet(checkpoint_path, index=False)
                elapsed = time.time() - start_time
                speed = (idx - start_idx + 1) / elapsed
                remaining = (len(df) - idx - 1) / max(speed, 0.001)
                logger.info(f"Checkpoint saved at {idx+1}/{len(df)} "
                           f"({speed:.1f} samples/sec, ~{remaining/60:.0f} min remaining)")
    
    # =========================================================================
    # 6. Save Results
    # =========================================================================
    ppl_df = pd.DataFrame(results)
    
    # Merge with original dataframe
    output_df = df.reset_index(drop=True).copy()
    for col in ppl_df.columns:
        output_df[col] = ppl_df[col].values
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    # =========================================================================
    # 7. Print Summary
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"PERPLEXITY EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"Total samples: {len(output_df)}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    logger.info(f"Speed: {len(df)/max(elapsed,1):.1f} samples/sec")
    logger.info(f"\nPerplexity Stats:")
    logger.info(f"  overall_ppl  - mean: {ppl_df['overall_ppl'].mean():.4f}, std: {ppl_df['overall_ppl'].std():.4f}")
    if 'line_ppl_std' in ppl_df.columns:
        logger.info(f"  line_ppl_std - mean: {ppl_df['line_ppl_std'].mean():.4f} (burstiness proxy)")
        logger.info(f"  ppl_variance - mean: {ppl_df['ppl_variance'].mean():.4f}")

    if 'label' in output_df.columns:
        logger.info(f"\nPerplexity by Label:")
        for label in sorted(output_df['label'].unique()):
            subset = output_df[output_df['label'] == label]
            label_name = 'Human' if label == 0 else 'AI'
            extra = ""
            if 'line_ppl_std' in subset.columns:
                extra = f", line_ppl_std mean={subset['line_ppl_std'].mean():.4f}"
            logger.info(f"  {label_name}: overall_ppl mean={subset['overall_ppl'].mean():.4f}{extra}")
    
    logger.info(f"{'='*60}")
    logger.info(f"\nTo merge with your features:")
    logger.info(f"  ppl_df = pd.read_parquet('{args.output}')")
    logger.info(f"  # Columns added: {list(ppl_df.columns)}")


if __name__ == "__main__":
    main()
