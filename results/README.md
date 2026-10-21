# IndicSent-M Results Directory

This directory stores real evaluation reports, predictions, confusion matrices, and model comparison metrics.

## Key Files

- `data_quality_report.json` & `.md`: Member 1 dataset health, missing rate, and class balance analysis.
- `text_metrics.json` & `text_predictions.csv`: Member 2 text-only performance and test set predictions.
- `image_metrics.json` & `image_predictions.csv`: Member 3 image-only performance and test set predictions.
- `multimodal_metrics.json` & `multimodal_predictions.csv`: Member 4 multimodal performance and test set predictions.
- `model_comparison.csv` & `model_comparison.json`: Side-by-side comparative analysis of all 3 approaches.
- `conflict_analysis.csv`: Detailed log of samples where text and image modalities disagreed, including confidence differentials.
