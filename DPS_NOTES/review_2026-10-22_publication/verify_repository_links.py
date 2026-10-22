"""Check rendered GitHub citations against published Git objects and local code."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
DPS = OUT.parent
WORKSPACE = DPS.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()

    def git(*command):
        return subprocess.check_output(["git", *command], cwd=args.repository)

    preamble = (DPS / "draft_support/slides_preamble.tex").read_text()
    base = re.search(r"\\newcommand\{\\AnalysisRepo\}\{([^}]+)\}", preamble).group(1)
    revision = re.search(r"\\newcommand\{\\AnalysisRevision\}\{([0-9a-f]{40})\}", preamble).group(1)
    git("merge-base", "--is-ancestor", revision, "origin/master")
    urls = set()
    for name in ("note1_mcp_reference", "note2_dual_threshold"):
        for filename in ("slides.pdf", "publication_candidate_slides.pdf"):
            for page in PdfReader(DPS / name / filename).pages:
                for annotation in page.get("/Annots", []):
                    uri = str(annotation.get_object().get("/A", {}).get("/URI", ""))
                    if uri.startswith(base + "/"):
                        urls.add(uri)
    assert urls, "No GitHub citations found"
    records = []
    scripts = set()
    for uri in sorted(urls):
        parsed = urlparse(uri)
        relative = unquote(parsed.path).removeprefix(urlparse(base).path + "/")
        kind, commit, path = relative.split("/", 2)
        assert kind == "blob" and commit == revision and path.endswith(".py"), uri
        obj = f"{revision}:{path}"
        assert git("cat-file", "-t", obj).decode().strip() == "blob", uri
        record = dict(url=uri, path=path, kind=kind, committed_target_exists=True)
        content = git("show", obj)
        assert (WORKSPACE / path).read_bytes() == content, f"Local code differs: {path}"
        assert (args.repository / path).read_bytes() == content, f"Checkout differs: {path}"
        record.update(sha256=hashlib.sha256(content).hexdigest(), local_and_checkout_match=True)
        if parsed.fragment:
            lines = re.fullmatch(r"L(\d+)(?:-L(\d+))?", parsed.fragment)
            assert lines, uri
            first = int(lines[1])
            last = int(lines[2] or lines[1])
            assert 1 <= first <= last <= len(content.splitlines()), uri
        scripts.add(path)
        records.append(record)
    report = dict(status="passed", repository=base, revision=revision,
                  published_ancestor_of_origin_master=True,
                  cited_scripts=sorted(scripts), links=records)
    (OUT / "repository_links.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(records)} GitHub targets and {len(scripts)} matching local scripts at {revision}.")


if __name__ == "__main__":
    main()
