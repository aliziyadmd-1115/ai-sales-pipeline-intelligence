# Data dictionary and validation

All source records are synthetic. Column names are trimmed and lowercased before schema validation. Missing or duplicated column names fail the batch. Unknown source columns are dropped. Unknown API fields are rejected.

| Field | Meaning / validation | Prediction input |
|---|---|---|
| opportunity_id | Nonempty unique string; conflicting duplicates fail cleaning | No |
| created_at | Parseable timestamp normalized to UTC | No |
| owner_email | Replaced with an email-redaction marker | No |
| region | Northeast, South, Midwest, West | Yes |
| industry | technology, financial_services, healthcare, retail, manufacturing | Yes |
| customer_segment | SMB, Mid-Market, Enterprise | Yes |
| product_line | analytics, automation, data_platform, security | Yes |
| sales_stage | qualification, discovery, proposal, negotiation | Yes |
| estimated_value | Finite number > 0; synthetic currency amount | Yes |
| days_in_pipeline | Integer 0–1,000 | Yes |
| engagement_score | Integer 0–100 | Yes |
| meetings_count | Integer 0–100 | Yes |
| competitor_present | Boolean; explicit true/false, 1/0, yes/no normalization | Yes |
| discount_pct | Finite fraction 0–1; 0.12 means 12% | Yes |
| proposal_sent | Same boolean rules | Yes |
| notes | 20–5,000 normalized characters; emails redacted | Yes |
| outcome | won or lost; normalized case/whitespace | Target |
| actual_revenue | Finite number >= 0; must equal zero for lost records | No |
| close_reason | Historical explanation, may be empty; emails redacted | No |

Category spelling is case-insensitive and surrounding whitespace is removed. The API and model inference use the same feature validators as cleaning. Explicit null/malformed values are rejected; missing API fields except notes receive documented defaults.

Invalid rows are omitted from the clean dataset with per-field counts. A row can violate multiple fields, so field counts may sum to more than invalid_rows_removed. Exact normalized duplicate IDs are removed after validation; a valid row survives an invalid duplicate. Conflicting valid rows sharing an ID fail the batch rather than arbitrarily choosing one.

If every row is invalid, the pipeline fails and leaves an existing cleaned output intact. Reports contain aggregate counts, not raw rejected records. Email redaction is not a comprehensive PII detection system.

A prediction request also accepts decision_threshold in [0, 1], default 0.50. The supplied value is returned without rounding that could misrepresent the actual decision boundary. A search request accepts a normalized query of 10–2,000 characters and top_k of 1–10.
