# Curation note

**Sources.** The dataset has two parts:

- **200 operational records** (`data/raw/operational_data.json`): the curated AfyaPlus dataset supplied for Week 4. They were built from anonymised chatbot logs filtered to operational queries, SOP manuals rewritten as question-and-answer pairs, and real help-desk tickets. The AfyaPlus operations team reviewed each one against the current SOP handbook.
- **20 clinical-redirect records** (`data/raw/safety_refusals.json`), which I wrote for this project and which were reviewed by hand. They cover fever, chest pain, dosing, pregnancy bleeding, poisoning and similar questions. Each answer declines to advise and sends the person to the matching AfyaPlus route: book under Appointments, same-day triage, the duty nurse, or emergency care. They follow the stated AfyaPlus policy that clinical questions always go to a provider. Run 1 showed why they are needed: with no example of declining, the model answered a child's fever with a dosing schedule.

The Kenyan grounding comes from six county facilities (Nairobi, Mombasa, Kisumu, Nakuru, Machakos and Kiambu), national-ID and passport registration rules, and the roles AfyaPlus staff hold: community health worker, duty nurse and front-desk clerk.

**Quality criteria.** `data_prep.py` blocks the split if any example has the wrong message structure, empty content, more than 512 tokens, a missing clinical-deferral disclaimer, or any mention of a medicine, dose, diagnosis or guarantee. It warns on duplicate questions and on examples under 64 tokens. The final run reports zero errors and zero warnings. All 220 examples fall between 190 and 247 tokens, averaging 221, by exact LLaMA 3 chat-template count (see `data/validation_report.json`).

**Safety.** 59 operational answers (30%) never sent clinical judgement to a provider. Each got the same closing sentence, so every example now teaches the disclaimer.

**Split.** The 200 operational questions share only 97 distinct answers. Records are grouped by answer and spread across the 12 topic areas, so no validation or test answer appears in training. The split is 176 train, 22 validation and 22 test.

**Coverage gaps.**
- Family and dependant workflows are thin (6 records), as are records transfers (10) and triage escalation (13).
- There is no Swahili or Sheng phrasing, and no references to SHA/SHIF insurance or the Data Protection Act 2019.
- Only one clinical-redirect example lands in the test set.
- The operational answers are rewordings of about 30 core procedures.
