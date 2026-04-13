"""
Pydantic models for the citation verification pipeline.

These models define the data structures for extracting claims,
recording verification results, and producing an overall report.
"""

from enum import Enum

from pydantic import BaseModel, computed_field


class VerificationStatus(str, Enum):
    """Outcome of verifying a single claim against its cited source."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class ClaimCitation(BaseModel):
    """A single claim extracted from the synthesis, paired with its citation."""

    claim: str
    citation_ids: list[int]
    source_titles: list[str]


class VerificationResult(BaseModel):
    """Result of verifying one claim against its cited source(s)."""

    claim: str
    citation_ids: list[int]
    source_titles: list[str]
    status: VerificationStatus
    explanation: str


# Scoring weights for each status
_STATUS_SCORE = {
    VerificationStatus.SUPPORTED: 1.0,
    VerificationStatus.PARTIALLY_SUPPORTED: 0.5,
    VerificationStatus.NOT_SUPPORTED: 0.0,
}


class UncitedReference(BaseModel):
    """A reference that was retrieved but not cited in the synthesis text."""

    citation_id: int
    title: str


class VerificationReport(BaseModel):
    """Aggregated verification report for a complete synthesis response."""

    results: list[VerificationResult]
    uncited_references: list[UncitedReference] = []

    @computed_field
    @property
    def confidence_score(self) -> float:
        """Percentage of claims that are SUPPORTED (partial = 0.5)."""
        if not self.results:
            return 100.0
        total = sum(_STATUS_SCORE[r.status] for r in self.results)
        return round((total / len(self.results)) * 100, 1)

    @computed_field
    @property
    def badge_color(self) -> str:
        """CSS color for the confidence badge."""
        if self.confidence_score > 90:
            return "#10b981"  # Green (Emerald)
        if self.confidence_score >= 70:
            return "#f59e0b"  # Yellow (Amber)
        return "#ef4444"  # Red

    @computed_field
    @property
    def badge_emoji(self) -> str:
        """Emoji for the confidence badge."""
        if self.confidence_score > 90:
            return "🟢"
        if self.confidence_score >= 70:
            return "🟡"
        return "🔴"

    @property
    def supported_count(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.SUPPORTED)

    @property
    def partial_count(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.PARTIALLY_SUPPORTED)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for r in self.results if r.status == VerificationStatus.NOT_SUPPORTED)
