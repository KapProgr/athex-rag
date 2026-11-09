"""Deploy the live demo to a Hugging Face Space with one command.

Note: Hugging Face now requires a PRO subscription for Docker Spaces. For a free
deployment use Render instead (render.yaml, see README); this script's
``--export-index deploy/demo-index`` builds the index that deployment uses.

    pip install -e ".[deploy]"
    setx HF_TOKEN "<hf write token>"      (once; https://huggingface.co/settings/tokens)
    python scripts/deploy_space.py

What it does:
1. builds a keyword-only (BM25) index from data/raw (parsed pages come from data/cache)
2. stages a Space folder: code, index, company metadata, Dockerfile and Space README
3. creates the Space <your-username>/athex-rag if needed, stores LLM_API_KEY as a Space
   *secret* (never uploaded as a file) and the settings as Space variables
4. uploads the folder; Hugging Face then builds and starts the container

The PDFs are not uploaded: "Open PDF" in the demo links to the company's official copy.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from athex_rag.embeddings import NullEmbedder  # noqa: E402
from athex_rag.index import Index  # noqa: E402
from athex_rag.ingest import chunks_for_document, iter_pdfs  # noqa: E402


def build_index(out: Path, chunk: int, pdfs: Path, cache: Path) -> int:
    chunks = []
    for pdf in iter_pdfs(pdfs):
        chunks += chunks_for_document(pdf, chunk, chunk // 6, True, cache_dir=cache)
        print(f"  {pdf.name}")
    if not chunks:
        sys.exit(f"No PDFs in {pdfs} (see data/README.md)")
    Index.build(chunks, NullEmbedder()).save(out)
    plain = out / "chunks.jsonl"
    with open(plain, "rb") as src, gzip.open(out / "chunks.jsonl.gz", "wb", 6) as dst:
        shutil.copyfileobj(src, dst)
    plain.unlink()
    return len(chunks)


def stage(dest: Path, chunk: int) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(
        ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    shutil.copy(ROOT / "pyproject.toml", dest)
    (dest / "data").mkdir()
    for name in ("companies.json", "sources.json"):
        shutil.copy(ROOT / "data" / name, dest / "data" / name)
    shutil.copy(ROOT / "deploy" / "space" / "Dockerfile", dest)
    shutil.copy(ROOT / "deploy" / "space" / "README.md", dest)  # also the package readme
    print("Building keyword index ...")
    n = build_index(dest / "index", chunk, ROOT / "data" / "raw", ROOT / "data" / "cache")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"Staged {dest} ({n} chunks, {size:.1f} MB)")


def deploy(dest: Path, space: str | None, llm: str, private: bool) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    key = os.environ.get("LLM_API_KEY")
    if not token:
        sys.exit(
            "HF_TOKEN is not set: create a *write* token at "
            'https://huggingface.co/settings/tokens, then  setx HF_TOKEN "..."'
        )
    if not key and not llm.startswith("ollama:"):
        sys.exit("LLM_API_KEY is not set (the demo's LLM needs it)")
    hf = HfApi(token=token)
    user = hf.whoami()["name"]
    repo_id = space or f"{user}/athex-rag"
    print(f"Space: {repo_id}")
    hf.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    if key:
        hf.add_space_secret(repo_id, "LLM_API_KEY", key)
    settings = {
        "ATHEX_LLM": llm,
        "ATHEX_RATE_PER_MINUTE": "6",  # per visitor
        "ATHEX_RATE_PER_DAY": "300",  # whole demo: protects the free LLM quota
        "ATHEX_LLM_MAX_RETRIES": "2",  # fail fast instead of making a visitor wait
    }
    for name, value in settings.items():
        hf.add_space_variable(repo_id, name, value)
    hf.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(dest),
        commit_message="Deploy athex-rag demo",
        delete_patterns=["src/**", "index/**", "data/**"],
    )
    owner, name = repo_id.split("/")
    print("\nUploaded. Hugging Face is building the container (a few minutes):")
    print(f"  page: https://huggingface.co/spaces/{repo_id}")
    print(f"  app:  https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", help="user/name (default: <your username>/athex-rag)")
    parser.add_argument("--llm", default="groq:qwen/qwen3.8-27b")
    parser.add_argument("--chunk", type=int, default=600, help="best setting in the evaluation")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="stage only, do not upload")
    parser.add_argument(
        "--export-index",
        metavar="DIR",
        help="only build the keyword index into DIR (used for the Render deploy)",
    )
    args = parser.parse_args()
    if args.export_index:
        out = Path(args.export_index)
        if out.exists():
            shutil.rmtree(out)
        n = build_index(out, args.chunk, ROOT / "data" / "raw", ROOT / "data" / "cache")
        print(f"Wrote {n} chunks to {out}")
        return
    dest = ROOT / "build" / "space"
    stage(dest, args.chunk)
    if args.dry_run:
        print(f"Dry run: test it with  cd {dest}  and  uvicorn athex_rag.api:app")
        return
    deploy(dest, args.space, args.llm, args.private)


if __name__ == "__main__":
    main()
