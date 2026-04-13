"""
Citation Verification Agent — second-pass quality gate.

After the primary agent generates a synthesized response with citations,
this module extracts each claim-citation pair, checks whether the cited
source actually supports the claim via a dedicated Gemma 4 call, and
returns a structured verification report.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama

from models.verification import (
    ClaimCitation,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
    UncitedReference,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_NAME = "gemma4:e2b"
VERIFICATION_OPTIONS = {"temperature": 0.1}
MAX_WORKERS = 4  # Parallel verification threads

_PROMPT_PATH = Path(__file__).parent / "prompts" / "verification_system_prompt.txt"
_SYSTEM_PROMPT: str | None = None


def _load_system_prompt() -> str:
    """Load the verification system prompt from disk (cached)."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _SYSTEM_PROMPT


# ──────────────────────────────────────────────
# Claim Extraction
# ──────────────────────────────────────────────
def _split_into_sentences(text: str) -> list[str]:
    """Split text into rough sentences, handling abbreviations gracefully."""
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    # or newline, but be careful with abbreviations like "et al."
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[*•\-])", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_citations_from_text(text: str) -> list[int]:
    """Extract citation numbers from a text fragment."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]


def _strip_markdown(text: str) -> str:
    """Remove markdown references section and structural markers."""
    # Remove everything from ## References onward
    text = re.split(r"\n##\s+References\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    # Remove ## Retrieval Notes section too
    text = re.split(r"\n##\s+Retrieval Notes\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip()


def extract_claims(synthesis_text: str, reference_map: dict[int, dict]) -> list[ClaimCitation]:
    """
    Extract claim-citation pairs from the synthesis text.

    A "claim" is any sentence or bullet point that contains at least one
    inline citation like [1] or [3]. We group consecutive citations
    together (e.g., [1, 3] becomes citation_ids=[1, 3]).
    """
    clean_text = _strip_markdown(synthesis_text)
    sentences = _split_into_sentences(clean_text)

    claims: list[ClaimCitation] = []
    for sentence in sentences:
        citation_ids = _extract_citations_from_text(sentence)
        if not citation_ids:
            continue

        # Only keep citation IDs that exist in the reference map
        valid_ids = [cid for cid in citation_ids if cid in reference_map]
        if not valid_ids:
            continue

        # Clean the claim text (remove citation markers for readability)
        claim_text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", sentence).strip()
        if not claim_text or len(claim_text) < 15:
            continue

        source_titles = [
            reference_map[cid].get("title", "Unknown")
            for cid in valid_ids
        ]

        claims.append(ClaimCitation(
            claim=claim_text,
            citation_ids=valid_ids,
            source_titles=source_titles,
        ))

    return claims


# ──────────────────────────────────────────────
# Single Claim Verification
# ──────────────────────────────────────────────
def _build_source_context(citation_ids: list[int], reference_map: dict[int, dict]) -> str:
    """Build the source context string for verification from cited references."""
    parts: list[str] = []
    for cid in citation_ids:
        ref = reference_map.get(cid)
        if ref is None:
            continue
        title = ref.get("title", "Unknown Title")
        abstract = ref.get("abstract", "No abstract available")
        parts.append(f"[Source {cid}] Title: {title}\nAbstract: {abstract}")
    return "\n\n".join(parts)


def _parse_verification_response(response_text: str) -> tuple[VerificationStatus, str]:
    """Parse the LLM verification response into status + explanation."""
    text = response_text.strip()

    # Try to extract STATUS line
    status_match = re.search(
        r"STATUS:\s*(SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED)",
        text,
        re.IGNORECASE,
    )
    # Try to extract EXPLANATION line
    explanation_match = re.search(
        r"EXPLANATION:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if status_match:
        status_str = status_match.group(1).upper().strip()
        try:
            status = VerificationStatus(status_str)
        except ValueError:
            status = VerificationStatus.PARTIALLY_SUPPORTED
    else:
        # Fallback: look for keywords in the response
        upper = text.upper()
        if "NOT_SUPPORTED" in upper or "NOT SUPPORTED" in upper:
            status = VerificationStatus.NOT_SUPPORTED
        elif "PARTIALLY_SUPPORTED" in upper or "PARTIALLY SUPPORTED" in upper:
            status = VerificationStatus.PARTIALLY_SUPPORTED
        elif "SUPPORTED" in upper:
            status = VerificationStatus.SUPPORTED
        else:
            status = VerificationStatus.PARTIALLY_SUPPORTED

    explanation = explanation_match.group(1).strip() if explanation_match else text[:200]
    # Clean up explanation (take first sentence only)
    explanation = explanation.split("\n")[0].strip()
    if len(explanation) > 300:
        explanation = explanation[:297] + "..."

    return status, explanation


def verify_single_claim(
    claim: ClaimCitation,
    reference_map: dict[int, dict],
) -> VerificationResult:
    """
    Verify a single claim against its cited source(s) via an Ollama call.

    Uses a strict verification prompt at temperature 0.1 for consistency.
    """
    system_prompt = _load_system_prompt()
    source_context = _build_source_context(claim.citation_ids, reference_map)

    user_message = (
        f"CLAIM: {claim.claim}\n\n"
        f"CITED SOURCE(S):\n{source_context}"
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options=VERIFICATION_OPTIONS,
        )
        response_text = response.get("message", {}).get("content", "")
        status, explanation = _parse_verification_response(response_text)
    except Exception as exc:
        status = VerificationStatus.PARTIALLY_SUPPORTED
        explanation = f"Verification failed: {exc}"

    return VerificationResult(
        claim=claim.claim,
        citation_ids=claim.citation_ids,
        source_titles=claim.source_titles,
        status=status,
        explanation=explanation,
    )


# ──────────────────────────────────────────────
# Full Verification Pipeline
# ──────────────────────────────────────────────
def verify_all(
    synthesis_text: str,
    reference_map: dict[int, dict],
    progress_callback=None,
) -> VerificationReport:
    """
    Extract all claim-citation pairs from the synthesis and verify each one.

    Args:
        synthesis_text: The synthesized response containing inline citations.
        reference_map: Mapping of citation IDs to source metadata (must
                       include 'abstract' field for verification).
        progress_callback: Optional callable(completed, total) for progress
                          updates (e.g., Streamlit progress bar).

    Returns:
        A VerificationReport with results for each claim and an overall
        confidence score.
    """
    claims = extract_claims(synthesis_text, reference_map)

    if not claims:
        return VerificationReport(results=[])

    total = len(claims)
    results: list[VerificationResult] = []

    # Verify claims in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_claim = {
            executor.submit(verify_single_claim, claim, reference_map): claim
            for claim in claims
        }

        completed = 0
        for future in as_completed(future_to_claim):
            result = future.result()
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    # Sort results to match the original claim order
    claim_order = {claim.claim: idx for idx, claim in enumerate(claims)}
    results.sort(key=lambda r: claim_order.get(r.claim, 999))

    # Find references that were not cited
    cited_ids = set()
    for claim in claims:
        cited_ids.update(claim.citation_ids)
        
    uncited_references = []
    for cid, ref in reference_map.items():
        if cid not in cited_ids:
            uncited_references.append(UncitedReference(
                citation_id=cid,
                title=ref.get("title", "Unknown Title")
            ))
            
    uncited_references.sort(key=lambda r: r.citation_id)

    return VerificationReport(
        results=results,
        uncited_references=uncited_references,
    )
