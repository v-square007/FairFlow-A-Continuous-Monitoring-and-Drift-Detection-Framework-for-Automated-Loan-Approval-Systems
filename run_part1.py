#!/usr/bin/env python3
"""
PART 1: Complete Execution Script
==================================
Executes the complete Part 1 pipeline:
1. Train model with fairness constraints
2. Generate all baseline artifacts
3. Validate success criteria

This is the single entry point for Part 1 deliverables.

Requirements:
- Preprocessing must be complete (artifacts/splits/ must exist)
- All required packages installed

Usage:
    python run_part1.py
"""

import sys
from pathlib import Path

def check_prerequisites():
    """Check that preprocessing is complete."""
    print("Checking prerequisites...")
    
    required_files = [
        "artifacts/splits/X_train.parquet",
        "artifacts/splits/y_train.parquet",
        "artifacts/splits/X_val.parquet",
        "artifacts/splits/y_val.parquet",
        "artifacts/splits/X_test.parquet",
        "artifacts/splits/y_test.parquet",
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print("\n❌ ERROR: Missing required preprocessing files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease run preprocessing first:")
        print("   python preprocessing/run_preprocessing.py")
        return False
    
    print("✓ All prerequisites satisfied\n")
    return True


def main():
    print("="*80)
    print("PART 1: MODEL TRAINING & BASELINE SETUP")
    print("="*80)
    print()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Step 2: Train model
    print("="*80)
    print("STEP 1: TRAINING MODEL WITH FAIRNESS CONSTRAINTS")
    print("="*80)
    print()
    
    try:
        from model.train import train_model
        model, features, threshold = train_model()
        print("\n✅ Model training complete!")
    except Exception as e:
        print(f"\n❌ ERROR during model training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Generate artifacts
    print("\n" + "="*80)
    print("STEP 2: GENERATING BASELINE ARTIFACTS")
    print("="*80)
    print()
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "generate_part1_artifacts.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        print(result.stdout)
        if result.stderr and "warning" not in result.stderr.lower():
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print("\n❌ ERROR: Artifact generation failed")
            return 1
        
    except Exception as e:
        print(f"\n❌ ERROR during artifact generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("✅ PART 1 COMPLETE - ALL DELIVERABLES READY")
    print("="*80)
    print()
    print("Next steps:")
    print("  → Review artifacts/baseline/metadata.json for performance metrics")
    print("  → Review artifacts/baseline/fairness_audit_report.json for fairness analysis")
    print("  → If all criteria met, proceed to Part 2 (Monitoring)")
    print()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)