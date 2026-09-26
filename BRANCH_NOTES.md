# explain-fix branch status

This branch responds to feedback on the draft report: in the deployed app,
the explanation did not explain its own recommendation. The verdict came from
the LSTM, the explanation came from fixed rules applied to indicator values,
and SHAP was applied to the Random Forest, which was not the served model.

## What's implemented and tested
- `src/explain/shap_english.py` plain-English sentences generated from the
  SHAP values of the exact prediction shown (all ten features covered).
- `src/models/confidence.py` detects collapsed models (no explanation is
  generated) and low-confidence predictions (explanation shown with a warning).
- `src/web/service.py` serves the Random Forest, so the explained model and
  the recommending model are the same.

Run the working pipeline from the command line:

    python -m src.web.service MSFT

## What's incomplete
Connecting this pipeline to the web interface produced an integration error
that could not be diagnosed before the submission deadline, so the web app on
this branch is not expected to work. The deployed version on `master` still
uses the original design. See Section 4.7 of the final report.  