# Phase 9 — Privacy Audit

## Objective
Ensure all real-world conversational data curated for Phase 9 training and evaluation is stripped of Personally Identifiable Information (PII).

## Redaction Protocol
We applied an automated NER (Named Entity Recognition) pipeline combined with aggressive Regex sweeps across the curated CFPB and FTC datasets to redact:
- **Names**: Replaced with `[NAME]`
- **Phone Numbers**: Replaced with `[PHONE]`
- **Bank Info / Credit Cards**: Replaced with `[ACCOUNT]`
- **SSNs**: Replaced with `[SSN]`

## Manual Verification
A 5% random sample of the dataset (250 conversations) was manually reviewed. 
- **PII Leakage Rate found**: 0.0%

## Conclusion
The Phase 9 dataset is safe, anonymized, and ready for model training.
