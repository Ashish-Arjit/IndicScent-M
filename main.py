"""
main.py - Master Pipeline Orchestrator for IndicSent-M
Provides CLI commands to execute individual member modules or the end-to-end pipeline.
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
    run_all: bool = False,
    run_test_suite: bool = False,
    launch_app: bool = False,
    track: str = "multimodal",
    memotion_dir: str = None,
    indic_file: str = None,
    fusion_method: str = "concat",
    tune_threshold: bool = False
):
    """Execute selected stages of the IndicSent-M pipeline."""
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
        print("\n[STEP 1] Running Member 1 Pipeline: Dataset Preparation & Quality Report...")
        if memotion_dir:
            print(f"Ingesting external Memotion 7K dataset from {memotion_dir}...")
            ingest_memotion_7k(memotion_dir)
        elif track == "indic":
            print("Preparing AI4Bharat IndicSentiment track benchmark...")
            ingest_indicsentiment(indic_file)
        else:
            load_or_create_dataset()
        generate_data_quality_report()

    # 2. Member 2: Text Model Training
    if train_text:
        print("\n[STEP 2] Running Member 2 Pipeline: Text-Only Model Training...")
        train_text_pipeline()

    # 3. Member 3: Image Model Training
    if train_image:
        print("\n[STEP 3] Running Member 3 Pipeline: Image-Only Model Training...")
        train_image_pipeline()

    # 4. Member 4: Multimodal Model Training
    if train_multimodal:
        print(f"\n[STEP 4] Running Member 4 Pipeline: Multimodal Fusion Model Training ({fusion_method})...")
        train_multimodal_pipeline(fusion_method=fusion_method)

    # 5. Evaluation & Comparison
    if evaluate_all:
        print("\n[STEP 5] Running Comprehensive Evaluation Across All 3 Modalities...")
        try:
            evaluate_text_model()
        except Exception as e:
            print(f"Warning: Text evaluation failed: {e}")

        try:
            evaluate_image_model()
        except Exception as e:
            print(f"Warning: Image evaluation failed: {e}")

        try:
            evaluate_multimodal_model()
        except Exception as e:
            print(f"Warning: Multimodal evaluation failed: {e}")

        print("\n[STEP 6] Running Modality Conflict & Sarcasm Diagnosis...")
        analyze_test_set_conflicts()

        print("\n[STEP 7] Generating Side-by-Side Model Comparison Table...")
        compare_all_models()

        # Display dashboard
        print_project_dashboard()

    # 6. Adaptive Threshold Tuning (Task 4.2)
    if tune_threshold:
        print("\n[STEP 8] Running Adaptive Conflict Threshold Tuning...")
        tune_conflict_threshold()

    # 7. Tests
    if run_test_suite:
        print("\n[TESTS] Executing Integration and Unit Test Suite...")
        success = run_tests()
        if not success:
            sys.exit(1)

    # 8. Streamlit Interface
    if launch_app:
        print("\n[UI] Launching Streamlit Application on http://localhost:8501...")
        app_file = BASE_DIR / "member5" / "app.py"
        subprocess.run(["streamlit", "run", str(app_file)])


def main():
    parser = argparse.ArgumentParser(
        description="IndicSent-M: Multimodal Sentiment Analysis Pipeline CLI"
    )
    parser.add_argument("--prepare-data", action="store_true", help="Prepare datasets and generate quality report (Member 1)")
    parser.add_argument("--track", choices=["multimodal", "indic"], default="multimodal", help="Dataset track: multimodal or indic")
    parser.add_argument("--memotion-dir", type=str, default=None, help="Directory of downloaded Memotion 7K dataset")
    parser.add_argument("--indic-file", type=str, default=None, help="File path to raw IndicSentiment CSV")
    parser.add_argument("--train-text", action="store_true", help="Train text-only sentiment model (Member 2)")
    parser.add_argument("--train-image", action="store_true", help="Train image-only sentiment model (Member 3)")
    parser.add_argument("--train-multimodal", action="store_true", help="Train multimodal fusion model (Member 4)")
    parser.add_argument("--fusion-method", choices=["concat", "late"], default="concat", help="Early concat or decision-level late fusion")
    parser.add_argument("--evaluate-all", action="store_true", help="Evaluate all models and analyze conflicts")
    parser.add_argument("--tune-threshold", action="store_true", help="Run adaptive conflict threshold ROC tuning")
    parser.add_argument("--run-all", action="store_true", help="Run complete pipeline end-to-end")
    parser.add_argument("--run-tests", action="store_true", help="Run integration and unit tests")
    parser.add_argument("--run-app", action="store_true", help="Launch interactive Streamlit UI (Member 5)")

    args = parser.parse_args()

    # If no flags passed, display help and print dashboard
    if not any(vars(args).values()):
        parser.print_help()
        print("\nRunning summary status check:")
        print_project_dashboard()
        return

    run_pipeline(
        prepare_data=args.prepare_data,
        train_text=args.train_text,
        train_image=args.train_image,
        train_multimodal=args.train_multimodal,
        evaluate_all=args.evaluate_all,
        run_all=args.run_all,
        run_test_suite=args.run_tests,
        launch_app=args.run_app,
        track=args.track,
        memotion_dir=args.memotion_dir,
        indic_file=args.indic_file,
        fusion_method=args.fusion_method,
        tune_threshold=args.tune_threshold
    )


if __name__ == "__main__":
    main()
