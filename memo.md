**To:** Clinical Director, AfyaPlus
**From:** AI Engineering
**Date:** 27 September 2026
**Re:** The AfyaPlus operational assistant is ready for a supervised staff pilot

**What we built.** We took a freely available AI language model and trained it on 220 reviewed examples of how AfyaPlus actually works. The 200 operational examples came from the operations team, covering booking, referrals, registration, records, billing, the dispensary, lab results, staff logins and triage escalation. The other 20 show the assistant declining medical questions and sending people to a clinician. The result is an assistant that answers staff and patient questions in AfyaPlus's own procedures and voice, and is trained to decline anything clinical: no diagnoses, medicines or doses. It can run on hardware AfyaPlus controls, so questions need not be sent to an outside AI service.

**How much better it is.** We tested it on 22 questions it had never seen, alongside the untrained model, and had both sets of answers graded by an independent AI reviewer:

- **Overall answer quality** rose from 37% to 82% of the maximum score (**+120%**). The new assistant was better on 18 of 22 questions and worse on none.
- **Answers that stick to real AfyaPlus procedures** rose from 23% to 83% (**+264%**). The untrained model invented systems and phone lines on nearly every question.
- **Fully correct answers:** 11 of 22 (**50%**), against none for the untrained model.
- **Every answer (100%)** now sends clinical questions to an AfyaPlus provider, up from 82%. Answers are 72% shorter: 58 words instead of 208.

**What it cost.** The final run, training plus testing, took 11 minutes on one rented graphics processor at $0.78 an hour: **about $0.14**. The training itself took 3 minutes, about $0.04. The first attempt, which we improved on, cost $0.11. Total computing cost: **about $0.25**, plus a small fee for the automated grading. Retraining with more examples will cost under $1 each time.

**Recommended next actions**

1. **Start a four-week, staff-only pilot in the five areas where it scored full marks:** staff logins and lockouts, patient registration, referral codes, billing coverage questions, and triage-queue escalation. A duty administrator reviews a sample of answers each week. *Rationale:* these are high-volume front-desk questions where the assistant was consistently right. They're a safe place to measure time saved before any patient-facing use.
2. **Have the operations team write and check about 100 more examples, then retrain.** Focus on the areas that scored lowest: follow-up bookings (2 out of 5; only 3 training examples against 40 for general booking), records transfers, dispensary, family accounts, and routing for patients in emotional distress. Include Swahili phrasings. *Rationale:* the test results show quality tracks directly with the number of examples per topic, and retraining is cheap.

**Main risk and how we manage it**

*Risk:* the assistant can state a made-up procedure with confidence. It did so on 4 of the 22 test questions (18%), for example describing a follow-up booking screen that does not exist. None of these were medical advice, but staff could act on a wrong process.

*Mitigation:*
- Keep a person in the loop. During the pilot, answers are for staff only, never shown directly to patients.
- Limit the pilot to the five full-marks areas.
- Add a line to every answer pointing to the SOP handbook.
- A built-in filter already replaces any answer that names a medicine or a dose with a referral to a provider.
