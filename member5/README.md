# Member 5: Integration, Streamlit Application & Testing

**Owner:** B. Sai Krishna  
**Milestone:** End-to-End Pipeline Integration, Interactive UI, and Validation

---

## Responsibilities

1. **Integrated Pipeline:** Coordinate models from Member 2 (text), Member 3 (image), and Member 4 (multimodal & conflict detection) into a unified inference manager (`integration.py`).
2. **Streamlit Application:** Deliver a user-friendly, responsive interface with tabs for live inference, comparative benchmarks, conflict exploration, and system diagnostics (`app.py`).
3. **Results Dashboard:** Provide terminal and summary inspection of evaluation results and data quality metrics (`results_dashboard.py`).
4. **Integration Testing:** Author and run integration test suites validating interoperability across all member modules (`testing.py`).

---

## Exposed Functions & Classes

- `IndicSentPipeline`: Unified class orchestrating text, image, and multimodal sentiment inference.
- `print_project_dashboard()`: Prints summary metrics directly to terminal.
- `run_tests()`: Executes full test suite across all 5 member contributions.

---

## Usage

```bash
# Run integration tests
python member5/testing.py

# Print console dashboard
python member5/results_dashboard.py

# Launch interactive Streamlit interface
streamlit run member5/app.py
```
