from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports"


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.paths.append(values["src"])
        if tag == "link" and values.get("href"):
            self.paths.append(values["href"])


def test_report_references_only_existing_local_assets() -> None:
    index = REPORT_ROOT / "index.html"
    parser = _AssetParser()
    parser.feed(index.read_text())

    assert parser.paths
    for relative in parser.paths:
        assert not relative.startswith(("http://", "https://", "//"))
        assert (REPORT_ROOT / relative).resolve().is_file(), relative


def test_report_registers_both_direction_specific_data_modules() -> None:
    concept_data = ROOT / "Concept_intervention" / "reports" / "data.js"
    jspace_data = ROOT / "J_space" / "reports" / "data.js"

    assert 'register("concept"' in concept_data.read_text()
    assert 'register("jspace"' in jspace_data.read_text()
    assert concept_data.stat().st_size < 100_000
    assert jspace_data.stat().st_size < 100_000


def test_report_assets_do_not_embed_generated_tensor_paths() -> None:
    frontend_files = [
        REPORT_ROOT / "index.html",
        REPORT_ROOT / "assets" / "registry.js",
        REPORT_ROOT / "assets" / "report.js",
        REPORT_ROOT / "assets" / "report.css",
    ]
    forbidden = (".npy", ".npz", ".pt", ".safetensors", ".parquet")

    for path in frontend_files:
        text = path.read_text()
        assert not any(extension in text for extension in forbidden), path
