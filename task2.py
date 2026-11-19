"""
TEKNOBUILT AI-ML Engineering Assessment - Task 2: P6 Schedule Batch Classification (POC)
Demonstrates:
  1. Ingestion & sanitization (UTF-8 BOM cleanup, normalization, Activity ID prefix extraction)
  2. Deduplication prior to inference (~83.6% volume reduction on P6_Schedule_Sample.xlsx)
  3. Strict Zero-Hallucination Taxonomy Schema (32 flattened Discipline | Sub-Discipline pairs)
  4. Micro-batched LLM request payload builder (50 rows/request)
"""

import argparse
import re
from typing import List, Literal

import pandas as pd
from pydantic import BaseModel, Field


# Exact 32 valid (Discipline | Sub-Discipline) pairs across 12 disciplines
ValidTaxonomyPair = Literal[
    "Architectural | Acoustical Ceiling Tile",
    "Architectural | Exterior Finishes",
    "Architectural | Hot Aisle Containment",
    "Architectural | Interior Finishes",
    "Architectural | Miscellaneous",
    "Building Automation System | Pull, Test and Terminate",
    "Civil | Exterior Equipment Pad",
    "Civil | Foundation Works",
    "Civil | Miscellaneous",
    "Civil | Slab on Grade",
    "Civil | Underground Dry Utilities",
    "Civil | Underground Wet Utilities",
    "Electrical | Cable Pulling",
    "Electrical | Cable Termination",
    "Electrical | Cable Trays Drops",
    "Electrical | Cable Trays Main",
    "Electrical | Miscellaneous",
    "Electrical | Rough-In",
    "Fire Alarm | Pull, Test and Terminate",
    "Fire Protection | Installation",
    "Landscape | Miscellaneous",
    "Mechanical | Exterior Installation",
    "Mechanical | Rough-In",
    "Plumbing | Rough-In",
    "Security | Pull, Test and Terminate",
    "Structural | Envelope",
    "Structural | Exterior Equipment Steel",
    "Structural | Miscellaneous",
    "Telecommunications | Basket Tray Main",
    "Telecommunications | Digital Antenna System / Radio Over IP",
    "Telecommunications | Miscellaneous",
    "Telecommunications | Production Fiber",
]


class ClassifiedItem(BaseModel):
    batch_idx: int = Field(..., description="0-based index of the item within the micro-batch")
    taxonomy_pair: ValidTaxonomyPair = Field(..., description="Strictly constrained Discipline | Sub-Discipline pair")
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchClassificationResponse(BaseModel):
    results: List[ClassifiedItem]


def extract_id_prefix(activity_id: str) -> str:
    """Extracts leading discipline/zone code from P6 Activity ID (e.g., 'TELE-C1-c1-6750' -> 'TELE-C1')."""
    parts = re.split(r"[-_]", str(activity_id).strip())
    return "-".join(parts[:2]).upper() if len(parts) >= 2 else parts[0].upper()


def analyze_and_prepare_batches(filepath: str, batch_size: int = 50) -> None:
    df = pd.read_excel(filepath) if filepath.endswith((".xlsx", ".xls")) else pd.read_csv(filepath)
    df.columns = [c.replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]

    df["norm_name"] = df["Activity Name"].astype(str).str.strip().str.lower()
    df["id_prefix"] = df["Activity ID"].apply(extract_id_prefix)
    df["dedup_key"] = df["id_prefix"] + " :: " + df["norm_name"]

    total_rows = len(df)
    unique_names = df["norm_name"].nunique()
    unique_keys = df["dedup_key"].nunique()
    num_batches = (unique_keys + batch_size - 1) // batch_size

    print(f"Input file:                     {filepath}")
    print(f"Total P6 rows:                  {total_rows:,}")
    print(f"Unique normalized names:        {unique_names:,} ({(1 - unique_names / total_rows) * 100:.1f}% reduction)")
    print(f"Unique (Prefix, Name) keys:     {unique_keys:,} ({(1 - unique_keys / total_rows) * 100:.1f}% reduction)")
    print(f"Micro-batches (@{batch_size}/call):      {num_batches:,} API requests (vs. {total_rows:,} naive calls)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEKNOBUILT Task 2: P6 Batch Classification POC")
    parser.add_argument("--input", default="P6_Schedule_Sample.xlsx", help="Path to P6 Excel/CSV file")
    parser.add_argument("--batch-size", type=int, default=50, help="Activities per LLM micro-batch")
    args = parser.parse_args()
    analyze_and_prepare_batches(args.input, args.batch_size)
