"""
run_project.py - Central Master Execution CLI for IndicSent-M
Provides clean, unified command-line controls to run:
1. Member 1: Dataset preparation (multimodal meme track or IndicSentiment track)
2. Member 2: Text sentiment classification (TF-IDF & IndicBERT transformer embeddings)
3. Member 3: Image sentiment classification (ResNet-18, ResNet-50, ViT-B-16, meme masking)
4. Member 4: Multimodal fusion (Early concat or Late fusion) & Cross-Modal Conflict Detection
5. Member 5: Integration, Streamlit interactive web application, & comprehensive test suite
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import ensure_directories
from member1.dataset_preprocessing import load_or_create_dataset, ingest_memotion_7k, ingest_indicsentiment
from member1.data_quality_report import generate_data_quality_report
from member2.text_model import train_text_pipeline
from member2.evaluate_text import evaluate_text_model
from member3.image_model import train_image_pipeline
from member3.evaluate_image import evaluate_image_model
from member4.multimodal_model import train_multimodal_pipeline, evaluate_multimodal_model
from member4.conflict_detection import analyze_test_set_conflicts, tune_conflict_threshold
from member4.compare_models import compare_all_models
from member5.results_dashboard import print_project_dashboard
from member5.testing import run_tests


def run_pipeline(
    prepare_data: bool = False,
    train_text: bool = False,
    train_image: bool = False,
    train_multimodal: bool = False,
    evaluate_all: bool = False,
    tune_threshold: bool = False,
    run_all: bool = False,
    run_test_suite: bool = False,
    launch_app: bool = False,
    track: str = "multimodal",
    memotion_dir: str = None,
    indic_file: str = None,
    fusion_method: str = "concat",
    backbone: str = "resnet18"
):
    """Execute selected components of IndicSent-M."""
    ensure_directories()

    if run_all:
        prepare_data = True
        train_text = True
        train_image = True
        train_multimodal = True
        evaluate_all = True
        tune_threshold = True

    # 1. Member 1: Dataset Preparation
    if prepare_data:
        print("\n================ [STEP 1] MEMBER 1: DATASET PREPARATION ================")
        if memotion_dir:
            print(f"Ingesting external Memotion 7K dataset from {memotion_dir}...")
            ingest_memotion_7k(memotion_dir)
        elif track == "indic":
            print("Preparing AI4Bharat IndicSentiment track benchmark...")
            ingest_indicsentiment(indic_file)
        else:
            print("Preparing multimodal paired meme dataset benchmark...")
            load_or_create_dataset()

        generate_data_quality_report()

    # 2. Member 2: Text Model Training
    if train_text:
        print("\n================ [STEP 2] MEMBER 2: TEXT SENTIMENT MODEL ===============")
        train_text_pipeline()

    # 3. Member 3: Image Model Training
    if train_image:
        print(f"\n============= [STEP 3] MEMBER 3: IMAGE SENTIMENT MODEL ({backbone.upper()}) =============")
        train_image_pipeline()

    # 4. Member 4: Multimodal Fusion Model Training
    if train_multimodal:
        print(f"\n========= [STEP 4] MEMBER 4: MULTIMODAL FUSION ({fusion_method.upper()}) =========")
        train_multimodal_pipeline(fusion_method=fusion_method)

    # 5. Evaluation & Comparison
    if evaluate_all:
        print("\n=============== [STEP 5] COMPREHENSIVE BENCHMARK EVALUATION ============")
        try:
            evaluate_text_model()
        except Exception as e:
            print(f"Warning: Text evaluation: {e}")

        try:
            evaluate_image_model()
        except Exception as e:
            print(f"Warning: Image evaluation: {e}")

        try:
            evaluate_multimodal_model()
        except Exception as e:
            print(f"Warning: Multimodal evaluation: {e}")

        print("\n============ [STEP 6] MODALITY CONFLICT & SARCASM DIAGNOSIS ============")
        analyze_test_set_conflicts()

        print("\n================= [STEP 7] SIDE-BY-SIDE MODEL COMPARISON ===============")
        compare_all_models()

        print_project_dashboard()

    # Adaptive Threshold Tuning
    if tune_threshold:
        print("\n========= [STEP 8] TASK 4.2: ADAPTIVE CONFLICT THRESHOLD TUNING ========")
        tune_conflict_threshold()

    # Integration Tests
    if run_test_suite:
        print("\n=================== [TESTS] EXECUTING INTEGRATION TESTS ================")
        success = run_tests()
        if not success:
            sys.exit(1)

    # Streamlit UI
    if launch_app:
        print("\n================= [UI] LAUNCHING STREAMLIT APPLICATION =================")
        app_file = BASE_DIR / "member5" / "app.py"
        subprocess.run(["streamlit", "run", str(app_file)])


def main():
    parser = argparse.ArgumentParser(
        description="IndicSent-M: Master CLI Runner for Multimodal Sentiment Analysis"
    )
    parser.add_argument("--all", "--run-all", dest="run_all", action="store_true", help="Execute complete pipeline end-to-end")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare dataset & data quality report (Member 1)")
    parser.add_argument("--track", choices=["multimodal", "indic"], default="multimodal", help="Dataset track: multimodal or indic")
    parser.add_argument("--memotion-dir", type=str, default=None, help="Directory of downloaded Memotion 7K dataset")
    parser.add_argument("--indic-file", type=str, default=None, help="File path to raw IndicSentiment CSV")
    parser.add_argument("--train-text", action="store_true", help="Train text-only sentiment classifier (Member 2)")
    parser.add_argument("--train-image", action="store_true", help="Train image-only sentiment classifier (Member 3)")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50", "vit_b_16"], default="resnet18", help="Vision backbone for Member 3")
    parser.add_argument("--train-multimodal", action="store_true", help="Train multimodal fusion classifier (Member 4)")
    parser.add_argument("--fusion-method", choices=["concat", "late"], default="concat", help="Early concat or decision-level late fusion (Member 4)")
    parser.add_argument("--evaluate", "--evaluate-all", dest="evaluate_all", action="store_true", help="Evaluate all models and analyze conflicts")
    parser.add_argument("--tune-threshold", action="store_true", help="Run adaptive conflict threshold ROC tuning (Task 4.2)")
    parser.add_argument("--test", "--run-tests", dest="run_tests", action="store_true", help="Run integration test suite (Member 5)")
    parser.add_argument("--app", "--run-app", dest="launch_app", action="store_true", help="Launch Streamlit web application (Member 5)")
    parser.add_argument("--dashboard", action="store_true", help="Print console summary dashboard")

    args = parser.parse_args()

    if args.dashboard:
        print_project_dashboard()
        return

    if not any(vars(args).values()) or (len(sys.argv) == 1):
        parser.print_help()
        print("\nSummary Status Check:")
        print_project_dashboard()
        return

    run_pipeline(
        prepare_data=args.prepare_data,
        train_text=args.train_text,
        train_image=args.train_image,
        train_multimodal=args.train_multimodal,
        evaluate_all=args.evaluate_all,
        tune_threshold=args.tune_threshold,
        run_all=args.run_all,
        run_test_suite=args.run_tests,
        launch_app=args.launch_app,
        track=args.track,
        memotion_dir=args.memotion_dir,
        indic_file=args.indic_file,
        fusion_method=args.fusion_method,
        backbone=args.backbone
    )


if __name__ == "__main__":
    main()
