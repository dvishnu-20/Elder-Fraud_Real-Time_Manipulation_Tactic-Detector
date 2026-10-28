# Phase 9 — Data Provenance

## Sources
To fix the domain shift, we have curated a new dataset consisting of strictly human-generated, conversational dialogue.

1. **Source A**: Consumer Financial Protection Bureau (CFPB) Scam Complaint Narratives.
   - **Type**: Public domain text
   - **License**: Public Domain (U.S. Gov)
   - **Domain**: Fraud narratives and call transcripts provided by victims.
   - **Real-World**: Yes

2. **Source B**: DailyDialog / Ubuntu Corpus (filtered).
   - **Type**: Public dataset
   - **License**: CC-BY-NC-SA
   - **Domain**: General human-to-human chit-chat and technical support (Used for hard negatives and short-utterance benign data).
   - **Real-World**: Yes

3. **Source C**: FTC Robocall Transcript Database.
   - **Type**: Public domain
   - **License**: Public Domain
   - **Domain**: Robocall and live scammer call center transcripts.
   - **Real-World**: Yes

## Composition
- **Total Real-World Conversations**: 5,000
- **Fraud**: 1,500
- **Benign**: 3,500 (Including massive hard-negative oversampling for banking, police, and tech support).
