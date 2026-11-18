# AWS Exam Gen

Generates practice questions for AWS certification exams, grounded in official AWS documentation and kept only if they survive an automated review.

## Language

### Exams

**Exam**:
One specific, versioned AWS certification exam, such as Solutions Architect – Associate version C03. Successive versions of the same certification are different Exams.
_Avoid_: Cert, certification (when the versioned exam is meant)

**Exam Code**:
The identifier of an Exam, such as `SAA-C03`.
_Avoid_: Cert code

**Certification**:
The AWS credential an Exam leads to. It is mostly a display name.

**Tier**:
The level of an Exam: Foundational, Associate, Professional or Specialty. Questions are written at their Exam's Tier.
_Avoid_: Difficulty

**Domain**:
One of an Exam's content domains, as listed in its Exam Guide. Every Question targets exactly one Domain.
_Avoid_: Domain hint, objective, topic

**Domain Weight**:
A Domain's share of an Exam's scored content, as stated in the Exam Guide.

**AWS Service**:
An AWS product that a Question tests, such as S3 or CloudFront.

### Source material

**Source Document**:
An official AWS document ingested for an Exam, such as an exam guide, whitepaper or FAQ.

**Exam Guide**:
The Source Document that defines an Exam's Domains and Domain Weights.

**Chunk**:
A retrievable passage of a Source Document.

**Grounding**:
The set of Chunks a Question was generated from.
_Avoid_: RAG context, source context

### Questions

**Question**:
One practice exam item for a single Exam and Domain.
_Avoid_: Item, MCQ

**Multiple-choice Question**:
A Question with exactly one Correct Answer among four Options.
_Avoid_: Single, single-answer

**Multiple-response Question**:
A Question with two or more Correct Answers among five Options. Its text states how many Options to select.
_Avoid_: Multiple, multiple-selection

**Option**:
One lettered answer choice of a Question.
_Avoid_: Choice, answer

**Correct Answer**:
An Option that is right.

**Distractor**:
An Option that is plausible but wrong.

**Explanation**:
The justification for why each Correct Answer is right and why each Distractor is wrong, citing AWS documentation.

### Review

**Generator**:
The role that drafts Questions from their Grounding and writes Revisions.

**AI Reviewer**:
The role that produces Critiques by judging Questions. It is distinct from the Generator, even when both use the same model.

**Review**:
The process that decides whether a Question is kept. A Review ends with the Question either Approved or Rejected.
_Avoid_: Validation

**Structural Check**:
A deterministic rule a Question must satisfy, applied as part of Review.

**Critique**:
A scored assessment of one Question, produced by the Structural Checks or the AI Reviewer. A Critique passes when its score meets the approval threshold and it has no critical Flag.

**Flag**:
One flaw identified in a Critique, with a Severity.

**Severity**:
How serious a Flag is: critical, major or minor.

**Revision**:
A rewrite of a Question made in response to a Critique.

**Approved**:
The outcome of a Review in which the Question's latest Critique passes. The Question is kept.

**Rejected**:
The outcome of a Review in which the Question still fails after the maximum number of Revisions. A Rejected Question never enters the Question Bank.

### Output

**Generation Run**:
One request to generate Questions for an Exam.

**Requested Count**:
The number of Approved Questions a Generation Run aims to export.

**Backfill**:
The extra rounds in a Generation Run that replace Questions lost to rejection or failed generation.
_Avoid_: Top-up

**Shortfall**:
The gap between the Requested Count and the number of Approved Questions when a Generation Run ends. A Shortfall is an acceptable result, not an error.

**Domain Mix**:
How a Generation Run allocates its Questions across Domains, in proportion to Domain Weight. It is balanced per Generation Run, not across the Question Bank.

**Domain Filter**:
A restriction that limits a Generation Run to a single Domain.

**Export**:
The JSON record of a Generation Run's Approved Questions. Every Generation Run that approves at least one Question writes exactly one Export.

**Rendering**:
A copy of a Generation Run's Approved Questions in a format for a learning platform, such as CSV or Moodle XML. It is written alongside the Export and is never part of the Question Bank.
_Avoid_: Export (for CSV or Moodle files)

**Question Bank**:
All Approved Questions ever exported for one Exam. Each Exam has its own Question Bank, and nothing carries over between Exam versions. The Exports are the Question Bank: no other copy exists.

**Duplicate**:
A Question whose normalized text matches a Question already in the Question Bank or earlier in the same Generation Run. Questions that describe the same scenario in different words are not Duplicates.

**Discarded**:
The fate of a Duplicate: it is dropped before Review and never receives a Critique.
_Avoid_: Rejected (for Duplicates)
