# Redrob AI Candidate Ranker

Offline candidate ranking system for the Redrob Intelligent Candidate Discovery
challenge. It ranks the 100,000-candidate pool against the provided Senior AI
Engineer job description and emits the required top-100 submission CSV.

## Methodology

The ranker uses a hybrid offline strategy. The official path is deterministic
and does not require network access, hosted LLM calls, or a GPU. It extracts
structured evidence from each candidate and combines:

- semantic fit to the JD from titles, headlines, summaries, skills, and career descriptions;
- search/retrieval/ranking domain evidence such as semantic search, RAG, recommendation systems, vector search, and learning-to-rank;
- vector and search stack evidence such as FAISS, Elasticsearch, OpenSearch, Pinecone, Weaviate, Qdrant, and Milvus;
- production ML evidence such as shipped systems, real users, scale, monitoring, A/B tests, and evaluation metrics;
- career quality signals including years of experience, seniority, product-company exposure, education relevance, and services-only risk;
- behavioral availability signals such as open-to-work, recruiter response rate, recent activity, interview completion, offer acceptance, notice period, and verification.

Suspicious profiles receive a multiplicative penalty for expert skills with
near-zero duration, unsupported AI keyword claims, nontechnical titles claiming
deep AI expertise, inconsistent timelines, services-only history, and broad
inflated expert skill lists.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The core ranker is mostly standard-library code. `sentence-transformers` is
listed as optional for local embedding experiments only. The submitted ranking
path does not download or call any external model.

For local embedding experiments only:

```bash
pip install -r requirements-optional.txt
```

## Generate Submission

```bash
python rank.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl --job India_runs_data_and_ai_challenge/job_description.docx --out submission.csv
```

This also writes `explanations.json` next to the CSV. The JSON includes
candidate-level strengths, weaknesses, behavioral assessment, score breakdown,
and recruiter summary.

## Validate

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
python scripts/validate_local.py --submission submission.csv --candidates India_runs_data_and_ai_challenge/candidates.jsonl
```

The first command checks the official CSV format. The second adds local checks
that candidate IDs exist and reasonings are non-empty.

## Configuration

Weights live in `config/weights.yaml`.

```yaml
weights:
  semantic_fit: 0.40
  domain_fit: 0.20
  production_experience: 0.15
  behavioral_score: 0.15
  career_quality: 0.10
```

The final score is:

```text
base_score = weighted feature sum
final_score = base_score * (1 - fraud_penalty)
```

Tie-breaking is deterministic by final score, semantic fit, domain fit,
production experience, then `candidate_id` ascending.

## Assumptions and Tradeoffs

- The default run is fully offline and CPU-only, matching the challenge
  reproduction constraints.
- The system favors explicit production search/ranking/retrieval evidence in
  career history over keyword-heavy skill sections.
- Optional local embeddings can be explored later, but no remote model download
  or hosted API call is used during ranking.
- Reasoning strings are generated only from observed candidate fields to avoid
  hallucinated recruiter summaries.
