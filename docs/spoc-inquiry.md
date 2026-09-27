# SPOC Inquiry — Day 1 Submission Legality & Evaluation Mechanics
**Samsung PRISM GenAI Hackathon · Theme 1: Agentic Code Intelligence (CoIR APPS)**

---

## 1. Context & Purpose
Under Theme 1 (CoIR APPS Code Retrieval), systems are evaluated using MTEB's `AppsRetrieval` benchmark task. VERA utilizes a **Verify-First Retrieval Architecture** combining dense semantic retrieval with dynamic program verification against the problem statement's worked example inputs.

To ensure strict compliance with competition rules and eliminate any evaluation ambiguity on Day 1, the following formal inquiries are prepared for the Samsung PRISM Single Point of Contact (SPOC).

---

## 2. Inquiries for the Organizer / SPOC

### Question 1: Permissibility of MTEB v2 `SearchProtocol` (Dynamic Verification Pipeline)
- **Question**: Is the submission required to be a subclass of `mteb.AbsEncoder` (which only takes raw strings and returns dense vector embeddings), or is an implementation of MTEB v2's official `mteb.SearchProtocol` (which exposes `index(corpus)` and `search(queries, top_k)` returning custom ranked score dictionaries) fully compliant with evaluation guidelines?
- **Defensive Mitigation**: VERA implements a **dual interface** in `vera/mtebio/`. If `SearchProtocol` is permitted, VERA runs the full verification-boosted pipeline. If organizers rule that only `AbsEncoder` text-embeddings are allowed, VERA switches seamlessly to the fine-tuned chassis-only fallback path (`VERAAbsEncoder`).

### Question 2: Exact Screening Cut & Test Benchmark Submission Deadlines
- **Question**: What is the exact hard deadline for final model submission and `appsretrieval_results.json` release artifact handover? Is there an intermediate screening cut, and if so, on what date and metric?
- **Impact on Schedule**: 
  - If deadline is **SHORT (< 10 days)**: VERA adopts the accelerated fork (ships R2 fallback at M7, skips M8 corpus-wide exploration).
  - If deadline is **NORMAL ($\ge 3$ weeks)**: VERA executes the full plan (includes M8 corpus-wide gating, QB-Norm calibration, and Stage-2 behavior fingerprinting).

### Question 3: Evaluation Environment Specifications
- **Question**: What are the compute specs of the official evaluation reproduction machine (CPU core count, RAM, maximum allowed inference runtime, OS version, GPU availability if any)?
- **Defensive Mitigation**: VERA defaults to CPU-only execution with zero network dependency during inference, pinned in `requirements.lock`.

---

## 3. Email Draft for Team SPOC

```text
Subject: Samsung PRISM GenAI Hackathon - Theme 1 (CoIR APPS): Submission Interface & Timeline Clarification

Dear Samsung PRISM Hackathon Organizing Committee,

We are participating in Theme 1: Agentic Code Intelligence (CoIR APPS). To ensure our pipeline design aligns completely with your automated evaluation harness, we would appreciate clarification on the following three points:

1. Submission Class Interface:
In MTEB v2, retrieval tasks can be evaluated via either:
   (a) A standard encoder embedding model (subclassing AbsEncoder, returning dense vectors), OR
   (b) A retrieval search protocol (implementing SearchProtocol with .index() and .search(), returning ranked score dictionaries).
Could you confirm whether implementing MTEB's SearchProtocol to return final ranked results is permitted for the official submission, or if the model must strictly be an AbsEncoder?

2. Evaluation Runtime Environment:
Will the official evaluation reproduction be conducted strictly on a CPU environment, and what is the maximum recommended execution time window for evaluating the 3,765 test queries?

3. Milestone Deadlines:
Could you confirm the exact deadline dates for any intermediate screening submission as well as the final artifact submission?

Thank you for your guidance and support.

Best regards,
Theme 1 Project Team (VERA)
```
