"""Audit saved prompt-B evidence offline and combine original/retry classifications.

No API calls, credentials, fuzzy similarity thresholds, or semantic judgement.
Only exact excerpts after conservative typography/case normalization, and
explicit ellipsis-separated fragments, can be recovered. Every quote must match.
Section mismatches are recorded, not silently rewritten. The most recent retry
is used regardless of label; failed retries never fall back to an earlier label.
"""
from pathlib import Path
import argparse
from difflib import SequenceMatcher
import hashlib
import importlib.util
import json
import re
import unicodedata

import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE / "prefilter_output/diagnostic_overshadowing_admission_input.parquet"
LABEL = "psychiatric_overshadowing_label"
RESULTS = "diagnostic_overshadowing_admission_results.csv"
TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
    "‐": "-", "‑": "-", "–": "-", "—": "-", "−": "-", "\u00ad": ""})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    """Preserve words, negation, numbers and punctuation other than typography."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))
                  .translate(TYPOGRAPHY).casefold()).strip()


def match_quote(quote, source):
    """Return the conservative matching rule, or None. Never approximate words."""
    if re.sub(r"\s+", " ", quote).strip() in re.sub(r"\s+", " ", source).strip():
        return "exact_whitespace_normalized"
    q, s = normalize(quote), normalize(source)
    if q in s:
        return "case_or_typography_normalized"
    # Ellipses must be explicit. Each fragment must carry at least three words,
    # remain in order, and have no more than 200 omitted characters per gap.
    fragments = [f.strip() for f in re.split(r"\.{3,}", q) if f.strip()]
    if len(fragments) < 2 or any(len(re.findall(r"\b\w+\b", f)) < 3 for f in fragments):
        return None
    starts = [m.start() for m in re.finditer(re.escape(fragments[0]), s)]
    for start in starts:
        end = start + len(fragments[0])
        for fragment in fragments[1:]:
            next_start = s.find(fragment, end)
            if next_start < 0 or next_start - end > 200:
                break
            end = next_start + len(fragment)
        else:
            return "ordered_ellipsis_fragments_max_gap_200_chars"
    return None


def locate_quote(quote, named, row, sections):
    rule = match_quote(quote, row[named])
    if rule:
        return rule, named, "named_section"
    alternatives = [(section, match_quote(quote, row[section])) for section in sections if section != named]
    alternatives = [(section, rule) for section, rule in alternatives if rule]
    if alternatives:
        return alternatives[0][1], "; ".join(section for section, _ in alternatives), "section_mismatch"
    return "unmatched", "", "unresolved"


def word_spans(text):
    """Token positions for diagnostics only; preserve decimal numbers and negations."""
    return list(re.finditer(r"\d+(?:\.\d+)*|[^\W\d_]+(?:['-][^\W\d_]+)*|_+", normalize(text)))


def closest_passage(quote, row, sections):
    """Heuristic candidate retrieval, never evidence acceptance or semantic similarity."""
    query = [m.group() for m in word_spans(quote)]
    best = {"similarity": -1, "section": "", "passage": "", "tokens": []}
    if not query:
        return best, query
    for section in sections:
        text = normalize(row[section])
        spans = word_spans(text)
        tokens = [m.group() for m in spans]
        matcher = SequenceMatcher(None, query, tokens, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        windows = set()
        for block in blocks:
            start = max(0, block.b - block.a)
            for shift in (-2, -1, 0, 1, 2):
                left = max(0, start + shift)
                for delta in (-6, -3, 0, 3, 6):
                    right = min(len(tokens), left + max(1, len(query) + delta))
                    if right > left:
                        windows.add((left, right))
        if blocks:
            windows.add((blocks[0].b, blocks[-1].b + blocks[-1].size))
        for left, right in sorted(windows):
            candidate = tokens[left:right]
            score = SequenceMatcher(None, query, candidate, autojunk=False).ratio()
            if score > best["similarity"]:
                best = {"similarity": score, "section": section,
                        "passage": text[spans[left].start():spans[right-1].end()],
                        "tokens": candidate}
    return best, query


def diagnose_quote(quote, row, sections):
    best, query = closest_passage(quote, row, sections)
    candidate = best.pop("tokens")
    changes = []
    for tag, a, b, c, d in SequenceMatcher(None, query, candidate, autojunk=False).get_opcodes():
        if tag != "equal":
            changes.append({"operation": tag, "quote_words": " ".join(query[a:b]),
                            "source_words": " ".join(candidate[c:d])})
    negation = {"no", "not", "never", "without", "denies", "denied", "deny", "negative", "neither", "nor", "n't", "cannot"}
    changed_words = [w for change in changes for field in ("quote_words", "source_words")
                     for w in change[field].split()]
    negation_difference = any(w in negation or w.endswith("n't") for w in changed_words)
    number_difference = any(re.search(r"\d", w) for w in changed_words)
    if best["similarity"] < 0.6:
        category = "no_close_candidate_under_0.60_heuristic"
    elif not changes:
        category = "same_word_tokens_punctuation_or_boundaries_differ"
    elif negation_difference or number_difference:
        category = "candidate_differs_in_negation_or_numbers"
    elif all(change["operation"] == "insert" for change in changes):
        category = "quote_omits_words_from_candidate"
    elif all(change["operation"] == "delete" for change in changes):
        category = "quote_adds_words_to_candidate"
    else:
        category = "word_substitution_or_mixed_edits"
    return {"diagnostic_category": category, "candidate_section": best["section"],
            "candidate_token_similarity": max(0, best["similarity"]),
            "candidate_passage_normalized": best["passage"],
            "candidate_has_negation_difference": negation_difference,
            "candidate_has_number_difference": number_difference,
            "token_changes": json.dumps(changes, ensure_ascii=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, default=HERE / "output_prompt_B_full_cohort")
    parser.add_argument("--retry-dir", type=Path, default=HERE / "output_prompt_B_invalid_retry")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output_prompt_B_evidence_audit")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("classifier", HERE / "02_run_diagnostic_overshadowing_admission_classifier.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sections = runner.SECTIONS
    original = pd.read_csv(args.original_dir / RESULTS, keep_default_na=False)
    retry = pd.read_csv(args.retry_dir / RESULTS, keep_default_na=False)
    manifests = [json.loads((folder / "run_manifest.json").read_text())
                 for folder in [args.original_dir, args.retry_dir]]
    for table, manifest in zip([original, retry], manifests):
        if (table.classification_input_id.duplicated().any()
                or len(table) != manifest["n_classification_requests"]
                or set(table.classification_input_id) != set(manifest["selected_input_ids"])
                or manifest["input_sha256"] != sha256(INPUT) or manifest["prompt_version"] != "B"):
            raise ValueError("Source run is incomplete, duplicated, or refers to different inputs")
    invalid_ids = set(original.loc[original.validation_status.eq("invalid"), "classification_input_id"])
    if set(retry.classification_input_id) != invalid_ids:
        raise ValueError("Retry must cover exactly the original invalid admissions")
    if manifests[1]["retry_source"]["results_sha256"] != sha256(args.original_dir / RESULTS):
        raise ValueError("Original results have changed since retry selection")
    data = pd.read_parquet(INPUT).set_index("classification_input_id", verify_integrity=True)
    lookup = retry.set_index("classification_input_id", drop=False)
    combined, audits = [], []
    for original_row in original.to_dict("records"):
        input_id = original_row["classification_input_id"]
        retried = input_id in invalid_ids
        result = lookup.loc[input_id].to_dict() if retried else original_row.copy()
        source = data.loc[input_id]
        for key in ["hadm_id", "subject_id", "pair_id"]:
            if str(result[key]) != str(source[key]):
                raise ValueError(f"Source metadata mismatch: {key}")
        decoded = {field: result[field] for field in runner.response_fields("B")}
        decoded["evidence"] = json.loads(decoded["evidence"])
        decoded[LABEL] = result["model_reported_label"]
        if decoded["overshadowing_psychiatric_illness"] == "":
            decoded["overshadowing_psychiatric_illness"] = None
        # Reuse existing non-evidence validation, then audit every original quote.
        error = ""
        try:
            runner.validate_response({**decoded, "evidence": []}, source, "B")
            if not isinstance(decoded["evidence"], list):
                raise ValueError("Invalid evidence list")
            for item in decoded["evidence"]:
                if (not isinstance(item, dict) or set(item) != {"section", "quote"}
                        or item["section"] not in sections or not isinstance(item["quote"], str)
                        or not item["quote"].strip() or len(item["quote"]) > 400):
                    raise ValueError("Invalid evidence structure, section, or quote length")
        except ValueError as exc:
            error = str(exc)
        quote_results = []
        if not error:
            for number, item in enumerate(decoded["evidence"], 1):
                rule, found, location = locate_quote(item["quote"], item["section"], source, sections)
                quote_results.append(location != "unresolved")
                audits.append({"classification_input_id": input_id, "hadm_id": result["hadm_id"],
                    "response_source": "retry" if retried else "original", "quote_number": number,
                    "model_reported_label": result["model_reported_label"], "named_section": item["section"],
                    "matched_sections": found, "match_rule": rule, "location_status": location,
                    "quote": item["quote"]})
        verified = not error and all(quote_results)
        if result["validation_status"] == "valid" and not verified:
            raise ValueError("A previously valid response failed revalidation")
        status = ("previously_valid" if result["validation_status"] == "valid" else
                  "recovered_offline" if verified else "unresolved")
        result.update({"response_source": "retry" if retried else "original",
            "offline_validation_status": status, "offline_validation_error": error,
            "n_evidence_quotes": len(quote_results), "n_matched_quotes": sum(quote_results),
            "final_label": result["model_reported_label"] if verified else ""})
        combined.append(result)
    table, audit = pd.DataFrame(combined), pd.DataFrame(audits)
    output = args.output_dir
    if output.resolve() in {args.original_dir.resolve(), args.retry_dir.resolve()}:
        raise ValueError("Audit must use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "combined_classifications.csv", index=False)
    audit.to_csv(output / "evidence_quote_audit.csv", index=False)
    audit[audit.classification_input_id.isin(table.loc[table.offline_validation_status.eq("unresolved"), "classification_input_id"])].to_csv(
        output / "unresolved_evidence_quotes.csv", index=False)
    diagnostics = []
    for quote_row in audit.loc[audit.location_status.eq("unresolved")].to_dict("records"):
        diagnostic = diagnose_quote(quote_row["quote"], data.loc[quote_row["classification_input_id"]], sections)
        diagnostics.append({**quote_row, **diagnostic,
            "candidate_section_differs_from_named": diagnostic["candidate_section"] != quote_row["named_section"],
            "automatically_accepted": False})
    diagnosis = pd.DataFrame(diagnostics)
    diagnosis.to_csv(output / "unmatched_quote_diagnosis.csv", index=False)
    category_summary = []
    if not diagnosis.empty:
        for category, group in diagnosis.groupby("diagnostic_category"):
            category_summary.append({"category": category, "n_quotes": len(group),
                                     "n_admissions": group.classification_input_id.nunique()})
    summary = {"n_admissions": len(table), "response_source_counts": table.response_source.value_counts().to_dict(),
        "offline_validation_counts": table.offline_validation_status.value_counts().to_dict(),
        "final_label_counts": table.final_label.replace("", "unresolved").value_counts().to_dict(),
        "quote_match_rules": audit.match_rule.value_counts().to_dict(),
        "quote_location_status": audit.location_status.value_counts().to_dict(),
        "unresolved_model_labels": table.loc[table.final_label.eq(""), "model_reported_label"].value_counts().to_dict(),
        "unmatched_quote_diagnosis": category_summary,
        "diagnostic_note": "Candidate passages are heuristic, not proven sources; categories do not establish clinical equivalence. Admission counts across categories can overlap. No diagnostic category changes final_label.",
        "limitations": "Evidence location verification only; not semantic or clinical validation. All quotes must match; no fuzzy word matching.",
        "source_sha256": {"input": sha256(INPUT), "original": sha256(args.original_dir / RESULTS),
                          "retry": sha256(args.retry_dir / RESULTS), "audit_script": sha256(__file__)}}
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
