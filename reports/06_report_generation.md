REPORT GENERATION SYSTEM
========================

The report generation system converts the raw JSON output from the PACT compliance pipeline
into a structured, human-readable compliance report. It is entirely rule-based — no language
model is involved. All text is generated from templates and logic defined in code.


HOW IT WORKS

The pipeline produces a JSON object for each image. Report generation takes that JSON and
turns it into a text report in four steps:

1. PACT runs on an image and outputs a JSON with per-person results:
   - person_id, person_bbox
   - worn: list of PPE items confirmed as worn (assigned AND anchored correctly)
   - missing: list of required PPE items not found
   - compliance_score: worn_required_count / total_required_count (0.0 to 1.0)
   - compliant: true only if all required items are worn

2. generate_report_data() in code/reporting/generator.py reads the JSON and produces:
   - A per-person narrative (one sentence describing their compliance status)
   - An overall scene assessment (one sentence for the whole image)
   - A grouped action item list (what PPE to provide, and to which workers)
   - Metadata (timestamp, context label, rule name)

3. The output is written as a plain-text report (.txt) and/or a self-contained HTML file
   with embedded person crop images (.html).

4. The JSON itself is also saved as-is for downstream use.


COMPLIANCE SCORING

Scoring is partial — a worker gets credit for each item they are wearing, not just pass/fail.

Per-person score = number of required items worn / total required items

Examples:
  - Required: helmet, vest. Worn: helmet.        Score = 0.50 (50%)
  - Required: helmet, vest. Worn: helmet, vest.  Score = 1.00 (100%)
  - Required: coverall, mask, gloves. Worn: coverall, mask.  Score = 0.67 (67%)

Overall scene compliance rate = mean of all per-person scores.

Binary compliance (is_compliant) is also tracked — true only when score = 1.0.
Both metrics are included in the JSON output and the text report.


SEVERITY TIERS

Each PPE type is assigned a severity level. This controls the language used in the report
and the priority of action items.

| Severity | PPE Types                                                |
|----------|----------------------------------------------------------|
| CRITICAL | helmet, hard-hat                                         |
| HIGH     | vest, safety-vest                                        |
| MEDIUM   | gloves, face-guard, face_shield, face-mask-medical, mask,|
|          | goggles, glasses, earmuffs                               |
| LOW      | shoes, foot, coverall, medical-suit, safety-suit,        |
|          | hands, head, face, tools                                 |

Action items are sorted by severity (CRITICAL first, LOW last).
Items with CRITICAL or HIGH severity are labeled [URGENT], others are [ADVISORY].


DATASET-SPECIFIC COMPLIANCE RULES

Different environments require different PPE. Each dataset has its own rule defining
which items are mandatory for a worker to be considered fully compliant.

CHV — Construction Site (OSHA 1910.135 / EN 397 / ISO 20471)

  Required: helmet, vest
  Context label: Construction Site

  Rationale: head protection and high-visibility clothing are the two universal
  requirements on active construction sites. Other items (gloves, boots) are
  advisory — site-specific and task-dependent.

CPPE-5 — Medical / Healthcare (WHO PPE Guidelines / CDC Infection Control)

  Required: coverall, mask, gloves
  Context label: Medical / Healthcare

  Rationale: full-body barrier (coverall), respiratory protection (mask), and
  hand protection (gloves) are the baseline for infection control in clinical settings.
  Goggles are recommended but treated as advisory in this rule set.

SH17 — Industrial Workplace (OSHA 29 CFR 1926 / ISO 45001)

  Required: helmet, safety-vest, gloves
  Context label: Industrial Workplace

  Rationale: head protection, visibility, and hand protection cover the three most
  common injury categories in industrial environments. SH17's 17-class coverage allows
  a broader required set than CHV's 2-class setup.


OUTPUT FORMAT

Plain text report structure:

  Context: [workplace label]
  Rule: [rule name]
  Image: [file path]
  Timestamp: [YYYY-MM-DD HH:MM:SS]

  Workers detected: N
  Compliant: N
  Violations: N
  Compliance rate: NN%

  Overall assessment:
  [One sentence about scene-level compliance.]

  Workers:
    Worker 1: NON-COMPLIANT (50%)
      Worn: helmet
      Missing: vest
      Worker 1 (50% compliant) — missing: High-Visibility Vest. Worn: Safety Helmet.
    ...

  Action items:
    [URGENT] Provide Safety Helmet — Worker 2, Worker 4
    [URGENT] Provide High-Visibility Vest — Worker 1, Worker 2, Worker 3

Action items are grouped by PPE type, not repeated per-person. If 3 workers all need a vest,
there is one action item — not three separate lines.


SAMPLE OUTPUTS

CHV — Construction Site (4 workers, image: ppe_0387.jpg)

  Workers detected: 4
  Compliance rate: 50%
  Worker 1: NON-COMPLIANT (50%) — vest detected, helmet missing
  Worker 2: NON-COMPLIANT (50%) — helmet detected, vest missing
  Worker 3: NON-COMPLIANT (50%) — vest detected, helmet missing
  Worker 4: NON-COMPLIANT (50%) — helmet detected, vest missing
  Action: Provide Safety Helmet — Worker 1, Worker 3
          Provide High-Visibility Vest — Worker 2, Worker 4

CPPE-5 — Medical / Healthcare (2 persons, image: 130.png)

  Workers detected: 2
  Compliance rate: 33%
  Worker 1: NON-COMPLIANT (67%) — coverall, goggles, mask worn; gloves missing
  Worker 2: NON-COMPLIANT (0%)  — no PPE detected (patient / non-worker)
  Action: Provide Protective Gloves — Worker 1, Worker 2
          Provide Face Mask — Worker 2
          Provide Coverall Suit — Worker 2

SH17 — Industrial Workplace (2 workers, image: pexels-photo-18110372.jpeg)

  Workers detected: 2
  Compliance rate: 33%
  Worker 1: NON-COMPLIANT (33%) — helmet worn; vest and gloves missing
  Worker 2: NON-COMPLIANT (33%) — helmet worn; vest and gloves missing
  Action: Provide Safety Vest — Worker 1, Worker 2
          Provide Protective Gloves — Worker 1, Worker 2


CODE STRUCTURE

  code/reporting/generator.py    — core report logic
    generate_report_data()       — main entry point, returns structured report dict
    _person_narrative()          — generates one-sentence per-person description
    _overall_assessment()        — generates scene-level assessment sentence
    _action_items()              — produces grouped, severity-sorted action list
    SEVERITY dict                — maps PPE names to CRITICAL/HIGH/MEDIUM/LOW
    PPE_LABELS dict              — maps internal names to human-readable labels
    CONTEXT_LABELS dict          — maps rule/dataset names to environment labels

  code/reporting/renderer.py     — HTML report renderer
    render_html_report()         — produces self-contained HTML with embedded images

  code/compliance/rules.py       — required PPE per rule/dataset
  code/compliance/pipeline.py    — upstream: produces the JSON that report reads


KNOWN LIMITATIONS

- PPE not detected by the detection model cannot appear in the report as "worn",
  even if it is visually present in the image. Low detection recall (especially for vests
  in CHV and gloves in CPPE-5) directly limits report accuracy.

- The system does not distinguish between intentional non-wearing (violation) and detection
  failure (false negative). Both appear as "missing" in the report.

- Persons detected by the pose model with no PPE assigned are assumed to be workers.
  Non-workers (patients, visitors) are not filtered out and will appear as violations.

- The report is static — it does not track workers across frames or over time.
