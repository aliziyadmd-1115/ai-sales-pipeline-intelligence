# Model card

## Intended use

An educational, synthetic sales-opportunity prioritization prototype. The output is an estimated win probability and a threshold-based won/lost classification. It is not a validated forecast for real customers, an automated sales decision system, or proof of business impact.

## Data and target

The generator creates 3,000 opportunities with random pre-close sales signals. A hidden, noisy formula uses engagement, meetings, proposal status, competition, discount, pipeline age, segment, stage, and product to sample won/lost outcomes. Notes come from 12 templates selected using pre-close signals. Industry and timestamps are synthetic and do not represent a real acquisition process.

Six duplicate rows and formatting issues exercise cleaning. No employer/client data is used. Only email addresses are redacted; names, telephone numbers, addresses, and other PII would require additional controls.

## Model and leakage boundary

A scikit-learn pipeline combines TF-IDF unigrams/bigrams, one-hot categories, standardized numeric inputs, and unweighted logistic regression. Preprocessing is fitted on training rows only. The complete saved pipeline performs inference.

The feature allowlist excludes opportunity ID, creation timestamp, owner email, outcome, actual revenue, and close reason. Post-close explanations are used only as retrieved historical evidence. In real CRM data, a pre-close snapshot timestamp and label availability timestamp would be necessary to audit temporal leakage.

## Evaluation protocol

Opportunity IDs are sorted before fixed, stratified splits: 60% train, 15% validation, 25% test. The 1,800 training rows fit preprocessing/model parameters. Five thresholds are compared on the 450 validation rows; the highest won-F1 chooses 0.30, with a higher-threshold tie-break. The untouched 750 test rows report final performance. The fitted model is not subsequently retrained on validation/test rows.

Primary classification results use the API default threshold of 0.50. Separate test results for the validation-selected threshold are labeled explicitly. All split IDs and test predictions are exported for audit. The dataset SHA-256 and package versions are recorded in metrics.json.

The model achieves ROC-AUC 0.7902, accuracy 0.7293, macro F1 0.6999, and Brier score 0.1774. A training-prior probability baseline has Brier score 0.2333; a majority classification baseline has accuracy 0.6293. A fixed-seed, 500-resample percentile bootstrap gives a 95% ROC-AUC interval of 0.7570–0.8216. This interval covers fixed-model row-sampling variability only.

## Probability interpretation

Class balancing was removed because weighted training changes the effective class prior. Unweighted logistic regression still does not guarantee calibration. Brier score and a reliability diagram assess probability quality on this synthetic population. Additional calibration, if needed on real data, must be fitted without using the final test set.

The prior benchmark used different partitions and class weighting. Its metrics are not an apples-to-apples estimate of this change's impact.

## Limitations and next evidence needed

- Synthetic generation and repeated text make generalization to real CRM histories unknown.
- A random split is appropriate only for this synthetic demonstration. Real data needs time-based evaluation, label availability, account grouping, and a defined prediction horizon.
- The notes-only same-industry retrieval proxy is below random and is not a relevance benchmark. Build independent domain relevance judgments before evaluating semantic retrieval.
- No business uplift, causal effect, monetary ROI, subgroup fairness, latency SLA, or drift-monitoring result is claimed.
- Checking citation IDs prevents invented references, not unsupported claims about valid references. Human review remains necessary.
- Chroma client contracts and Ollama response handling are tested with substitutes. Live integrations must be tested with their actual services.
