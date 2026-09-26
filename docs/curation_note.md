# Curation note

**Sources.** The 200 question-and-answer records in `data/raw/operational_data.json` are the curated AfyaPlus operational dataset supplied for Week 4. They were assembled through three enterprise channels: anonymised chatbot logs filtered to operational queries, SOP manuals rewritten as question-and-answer pairs, and help-desk tickets that show how staff actually phrase requests. The AfyaPlus operations team reviewed every record against the current SOP handbook and removed anything clinical. Nothing was generated or scraped for this project. The Kenyan grounding comes from six county facilities (Nairobi, Mombasa, Kisumu, Nakuru, Machakos and Kiambu), national-ID and passport registration rules, and the roles AfyaPlus staff hold: community health worker, duty nurse and front-desk clerk.

**Quality criteria.** `data_prep.py` blocks the split if any example has the wrong message structure, empty content, more than 512 tokens, a missing clinical-deferral disclaimer, or dosage, diagnosis or guarantee language. It warns on duplicate questions and on examples under 64 tokens. The final run reports zero errors and zero warnings, and all examples fall between 258 and 324 tokens (see `data/validation_report.json`).

**Safety.** 59 answers (30%) never sent clinical judgement to a provider. Each got the same closing sentence ("Anything that needs clinical judgement goes to an AfyaPlus provider."), so every training example teaches the disclaimer.

**Split.** The 200 questions share only 97 distinct answers. A random 80/10/10 shuffle would put the same answer text in both train and test and inflate the scores. Records are therefore grouped by answer and assigned to splits as whole groups, spread across operational areas: 160 train, 20 validation and 20 test, with zero answers shared between splits. The test set covers all 11 areas.

**Coverage gaps.**
- Family and dependant workflows are thin (6 records), as are records transfers (10) and triage escalation (13).
- There is no Swahili or Sheng phrasing, and no references to SHA/SHIF insurance or the Data Protection Act 2019.
- There are no urgent or emergency escalation scenarios.
- Seven questions carry a doubled role prefix, for example "As a community health worker: As a duty nurse…".
- Behavioural diversity is lower than the 200 records suggest, because the answers are reworded versions of about 30 core procedures.
