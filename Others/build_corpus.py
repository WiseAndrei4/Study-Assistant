# build_wiki_corpus.py
import json
import re
from datasets import load_dataset

OUTPUT_FILE = "wiki_tech_corpus.jsonl"
TARGET_PASSAGES = 5000
CHUNK_WORDS = 180
OVERLAP_WORDS = 30

# Optional keyword filter to keep the corpus focused on STEM / CS / Math
TOPIC_KEYWORDS = re.compile(
    r"\b(algorithm|matrix|eigenvalue|probability|theorem|neural|gradient|"
    r"calculus|differential|manifold|entropy|fourier|optimization|graph theory|"
    r"linear algebra|bayesian|markov|transformer|embedding|complexity)\b",
    re.IGNORECASE,
)


def chunk_text(text: str, chunk_size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if len(words) >= 40 else []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        segment = " ".join(words[start : start + chunk_size]).strip()
        if len(segment.split()) >= 40:
            chunks.append(segment)
    return chunks


def main() -> None:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for article in ds:
            title = article["title"].strip()
            text = article["text"].strip()

            # Drop disambiguation/list stubs and off-topic articles
            if title.startswith("List of ") or "(disambiguation)" in title:
                continue
            if not TOPIC_KEYWORDS.search(text[:1500]):
                continue

            # Strip trailing boilerplate sections
            for cutoff in ("\nSee also\n", "\nReferences\n", "\nExternal links\n"):
                if cutoff in text:
                    text = text.split(cutoff, 1)[0]

            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks, start=1):
                record = {
                    "title": f"{title} [§{idx}]" if len(chunks) > 1 else title,
                    "text": chunk,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= TARGET_PASSAGES:
                    print(f"Saved {written} clean Wikipedia passages to {OUTPUT_FILE}")
                    return


if __name__ == "__main__":
    main()