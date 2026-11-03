"""PASS / WARN / FAIL results, shared by `so101 check`, dry runs and dataset validation."""

from dataclasses import asdict, dataclass, field
from typing import Literal

Status = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class Result:
    group: str  # e.g. "follower", "front camera", "dataset"
    status: Status
    detail: str
    fix: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add(self, group: str, status: Status, detail: str, fix: str = "") -> None:
        self.results.append(Result(group, status, detail, fix))

    def ok(self, group: str, detail: str) -> None:
        self.add(group, "PASS", detail)

    def warn(self, group: str, detail: str, fix: str = "") -> None:
        self.add(group, "WARN", detail, fix)

    def fail(self, group: str, detail: str, fix: str = "") -> None:
        self.add(group, "FAIL", detail, fix)

    def extend(self, other: "Report") -> None:
        self.results.extend(other.results)

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if r.status == "FAIL"]

    @property
    def warnings(self) -> list[Result]:
        return [r for r in self.results if r.status == "WARN"]

    def print(self, title: str = "") -> None:
        if title:
            print(f"\n{title}")
        width = max((len(r.group) for r in self.results), default=0)
        for r in self.results:
            print(f"  {r.status}  {r.group:<{width}}  {r.detail}")
            if r.fix and r.status != "PASS":
                print(f"        {'':<{width}}  fix: {r.fix}")

    def summary(self) -> str:
        fails, warns = len(self.failures), len(self.warnings)
        if fails:
            return f"{fails} FAIL, {warns} WARN: fix the failures first."
        if warns:
            return f"No failures, {warns} WARN."
        return "All checks passed."

    def to_json(self) -> list[dict]:
        return [asdict(r) for r in self.results]
