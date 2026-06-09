"""
scripts/build_gene_set_mask.py
==============================
Build a binary gene-set mask [P, G] from MSigDB GMT file(s) and custom gene sets.

Usage:
    python scripts/build_gene_set_mask.py \
        --gmt_path data/h.all.v2026.1.Hs.symbols.gmt \
        --extra_gmt_paths data/c2.cp.reactome.v2023.2.Hs.symbols.gmt \
        --custom_gmt_path data/custom_gene_sets.gmt \
        --gene_names_path data/preprocessed/gene_names.txt \
        --selected_sets_path configs/selected_gene_sets_v2.txt \
        --out_mask_path data/gene_set_mask.pt \
        --out_names_path data/gene_set_names.txt \
        --dump_overlap_path data/gene_set_overlap.tsv \
        --min_overlap 5 \
        --strict \
        --upper
"""

import argparse
import os
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not available. Will use numpy .npy format instead of .pt")


def parse_gmt(gmt_path):
    """Parse MSigDB GMT file. Returns dict {gene_set_name: set(gene_symbols)}."""
    if not os.path.exists(gmt_path):
        raise FileNotFoundError(f"GMT file not found: {gmt_path}")
    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = set(g.strip() for g in parts[2:] if g.strip())
            gene_sets[name] = genes
    return gene_sets


def build_mask(gene_names_model, gene_sets_selected, min_overlap=5, strict=False):
    """Build binary mask [P, G].

    Returns:
        mask: [P, G] float32
        overlap_counts: list[int]
        overlap_symbols: list[list[str]]  # overlapping gene symbols per set
        names: list[str]
    """
    P = len(gene_sets_selected)
    G = len(gene_names_model)
    mask = np.zeros((P, G), dtype=np.float32)
    overlap_counts = []
    overlap_symbols = []

    gene_to_idx = {g: i for i, g in enumerate(gene_names_model)}
    names = []

    for p, (name, genes) in enumerate(gene_sets_selected.items()):
        matched = []
        for g in genes:
            if g in gene_to_idx:
                mask[p, gene_to_idx[g]] = 1.0
                matched.append(g)
        overlap_counts.append(len(matched))
        overlap_symbols.append(matched)
        names.append(name)

        if len(matched) < min_overlap:
            msg = f"Gene set '{name}' has only {len(matched)} overlapping genes (min={min_overlap})"
            if strict:
                raise ValueError(msg)
            else:
                print(f"[WARNING] {msg}")

    return mask, overlap_counts, overlap_symbols, names


def main():
    parser = argparse.ArgumentParser(description="Build gene-set mask from MSigDB GMT(s)")
    parser.add_argument("--gmt_path", type=str, required=True, help="Primary MSigDB GMT file")
    parser.add_argument("--extra_gmt_paths", type=str, nargs="+", default=None,
                        help="Extra GMT file(s) to search for selected gene sets")
    parser.add_argument("--custom_gmt_path", type=str, default=None,
                        help="Custom GMT file (e.g., DNA_REPAIR_SUPERSET)")
    parser.add_argument("--gene_names_path", type=str, required=True,
                        help="File with one gene symbol per line (model input order)")
    parser.add_argument("--selected_sets_path", type=str, required=True,
                        help="File with one gene set name per line")
    parser.add_argument("--out_mask_path", type=str, default="data/gene_set_mask.pt")
    parser.add_argument("--out_names_path", type=str, default="data/gene_set_names.txt")
    parser.add_argument("--dump_overlap_path", type=str, default=None,
                        help="TSV path to dump overlap gene symbols per set")
    parser.add_argument("--min_overlap", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--upper", action="store_true",
                        help="Uppercase gene names for matching")
    args = parser.parse_args()

    # Load model gene names
    with open(args.gene_names_path, "r", encoding="utf-8") as f:
        gene_names = [line.strip() for line in f if line.strip()]
    if args.upper:
        gene_names = [g.upper() for g in gene_names]
    print(f"[INFO] Loaded {len(gene_names)} model genes")

    # Load selected gene set names
    with open(args.selected_sets_path, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Selected {len(selected_names)} gene sets")

    # Parse all GMT sources
    all_gene_sets = {}

    # Primary GMT
    primary_sets = parse_gmt(args.gmt_path)
    if args.upper:
        primary_sets = {k: set(g.upper() for g in v) for k, v in primary_sets.items()}
    all_gene_sets.update(primary_sets)
    print(f"[INFO] Loaded {len(primary_sets)} sets from primary GMT: {args.gmt_path}")

    # Extra GMTs
    if args.extra_gmt_paths:
        for extra_path in args.extra_gmt_paths:
            extra_sets = parse_gmt(extra_path)
            if args.upper:
                extra_sets = {k: set(g.upper() for g in v) for k, v in extra_sets.items()}
            # Merge: extra can supplement or override primary
            for k, v in extra_sets.items():
                if k in all_gene_sets:
                    print(f"[INFO] Merging/overriding '{k}' from extra GMT: {extra_path}")
                all_gene_sets[k] = v
            print(f"[INFO] Loaded {len(extra_sets)} sets from extra GMT: {extra_path}")

    # Custom GMT
    if args.custom_gmt_path:
        if not os.path.exists(args.custom_gmt_path):
            raise FileNotFoundError(f"Custom GMT not found: {args.custom_gmt_path}")
        custom_sets = parse_gmt(args.custom_gmt_path)
        if args.upper:
            custom_sets = {k: set(g.upper() for g in v) for k, v in custom_sets.items()}
        for k, v in custom_sets.items():
            if k in all_gene_sets:
                print(f"[INFO] Overriding '{k}' with custom GMT definition")
            all_gene_sets[k] = v
        print(f"[INFO] Loaded {len(custom_sets)} sets from custom GMT: {args.custom_gmt_path}")

    # Validate selected names exist
    missing = [n for n in selected_names if n not in all_gene_sets]
    if missing:
        available = sorted(list(all_gene_sets.keys()))[:20]
        raise ValueError(
            f"Missing gene sets: {missing}\n"
            f"Available examples: {available}\n"
            f"Please check gene set names match exactly."
        )

    gene_sets_selected = {n: all_gene_sets[n] for n in selected_names}

    # Build mask
    mask, overlaps, overlap_symbols, names = build_mask(
        gene_names, gene_sets_selected, min_overlap=args.min_overlap, strict=args.strict
    )

    print(f"\n[INFO] Mask shape: {mask.shape}")
    for n, o, syms in zip(names, overlaps, overlap_symbols):
        sym_str = ", ".join(syms[:10])
        if len(syms) > 10:
            sym_str += f", ... (+{len(syms)-10} more)"
        print(f"  {n:55s}: {o:3d} genes | {sym_str}")

    print(f"\n[INFO] Total mask entries: {int(mask.sum())} / {mask.size} ({100*mask.sum()/mask.size:.2f}%)")

    # Save mask
    os.makedirs(os.path.dirname(args.out_mask_path) or ".", exist_ok=True)
    if HAS_TORCH:
        torch.save(torch.from_numpy(mask), args.out_mask_path)
    else:
        npy_path = args.out_mask_path.replace(".pt", ".npy")
        np.save(npy_path, mask)
        print(f"[INFO] Saved as numpy array: {npy_path}")
        print(f"       On server, load with:  mask = torch.from_numpy(np.load('{npy_path}'))")
    print(f"\n[SAVED] Mask -> {args.out_mask_path}")

    # Save names
    with open(args.out_names_path, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")
    print(f"[SAVED] Names -> {args.out_names_path}")

    # Dump overlap genes to TSV
    if args.dump_overlap_path:
        os.makedirs(os.path.dirname(args.dump_overlap_path) or ".", exist_ok=True)
        with open(args.dump_overlap_path, "w", encoding="utf-8") as f:
            f.write("gene_set_name\toverlap_count\toverlap_genes\n")
            for n, o, syms in zip(names, overlaps, overlap_symbols):
                f.write(f"{n}\t{o}\t{','.join(syms)}\n")
        print(f"[SAVED] Overlap dump -> {args.dump_overlap_path}")


if __name__ == "__main__":
    main()
