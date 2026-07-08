"""Checkpoint history for a universe run."""

from __future__ import annotations

from dataclasses import dataclass, field

from bbsim.core.report import StageReport


@dataclass(slots=True)
class RunHistory:
    """Stores checkpoint reports for one universe run."""

    reports: list[StageReport] = field(default_factory=list)

    def add_report(self, report: StageReport) -> None:
        """Append a stage report to the run history.

        Args:
            report: Report to store.
        """

        self.reports.append(report)
