import glob
import json
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from transformers import AutoTokenizer

import hnsw_core
from ml.bi_encoder import BiEncoder
from ml.bm25 import BM25Index
from ml.cross_encoder import CrossEncoderReranker
from exporter import Exporter
from ml.hybrid import reciprocal_rank_fusion
from ml.metrics import mrr_at_k, ndcg_at_k
from ocr_engine import OCREngine

STORAGE_DIR = "storage"
HNSW_PATH = os.path.join(STORAGE_DIR, "index.hnsw")
BM25_PATH = os.path.join(STORAGE_DIR, "bm25.json")
DOCSTORE_PATH = os.path.join(STORAGE_DIR, "docstore.json")
VECTORS_PATH = os.path.join(STORAGE_DIR, "vectors.npy")

BI_ENCODER_MODEL = "allenai/scibert_scivocab_uncased"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIM = 768


class RetrievalEngineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        os.makedirs(STORAGE_DIR, exist_ok=True)

        self.title("Hybrid Vector Search & Learned Reranking Engine")
        self.geometry("1200x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Engine state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.bi_encoder = None
        self.cross_encoder = None
        self.ocr_engine = None
        self.models_ready = False

        self.hnsw = hnsw_core.HNSWIndex(dim=EMBED_DIM, M=16, ef_construction=200, seed=42)
        self.bm25 = BM25Index()
        self.docstore: list[dict] = []
        self.vectors = np.empty((0, EMBED_DIM), dtype=np.float32)
        self.latency_history: list[dict] = []
        self.last_results: list[dict] = []

        self._load_persisted_state()
        self._build_ui()
        self._start_model_loader()

    # -------------------- Persistence --------------------

    def _load_persisted_state(self) -> None:
        if os.path.exists(DOCSTORE_PATH):
            with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
                self.docstore = json.load(f)
        if os.path.exists(HNSW_PATH):
            self.hnsw.load(HNSW_PATH)
        if os.path.exists(BM25_PATH):
            self.bm25.load(BM25_PATH)
        if os.path.exists(VECTORS_PATH):
            self.vectors = np.load(VECTORS_PATH).astype(np.float32)

    def _save_persisted_state(self) -> None:
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.docstore, f, indent=2, ensure_ascii=False)
        self.hnsw.save(HNSW_PATH)
        self.bm25.save(BM25_PATH)
        np.save(VECTORS_PATH, self.vectors)

    # -------------------- UI Layout --------------------

    def _build_ui(self) -> None:
        self.sidebar_frame = ctk.CTkFrame(self, width=230, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Retrieval Engine",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        self.index_stats_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=f"Indexed Docs: {self.hnsw.size}",
            font=ctk.CTkFont(size=12),
        )
        self.index_stats_label.grid(row=1, column=0, padx=20, pady=(0, 15))

        self.btn_ingest_corpus = ctk.CTkButton(
            self.sidebar_frame,
            text="Ingest Corpus (JSONL/MD)",
            command=self.ingest_corpus_event,
        )
        self.btn_ingest_corpus.grid(row=2, column=0, padx=20, pady=8, sticky="ew")

        self.btn_ingest_images = ctk.CTkButton(
            self.sidebar_frame,
            text="Ingest Math Images (OCR)",
            command=self.ingest_image_folder_event,
        )
        self.btn_ingest_images.grid(row=3, column=0, padx=20, pady=8, sticky="ew")

        self.btn_index_editor = ctk.CTkButton(
            self.sidebar_frame,
            text="Index Text from Editor",
            command=self.index_editor_text_event,
        )
        self.btn_index_editor.grid(row=4, column=0, padx=20, pady=8, sticky="ew")

        self.btn_archive = ctk.CTkButton(
            self.sidebar_frame,
            text="Browse Indexed Docs",
            command=self.open_archive,
        )
        self.btn_archive.grid(row=5, column=0, padx=20, pady=8, sticky="ew")

        self.btn_dashboard = ctk.CTkButton(
            self.sidebar_frame,
            text="Recall / QPS Dashboard",
            command=self.open_telemetry_dashboard,
        )
        self.btn_dashboard.grid(row=6, column=0, padx=20, pady=8, sticky="ew")

        self.btn_export = ctk.CTkButton(
            self.sidebar_frame,
            text="Export Results (.md)",
            command=self.export_event,
        )
        self.btn_export.grid(row=7, column=0, padx=20, pady=8, sticky="ew")

        # Main workspace
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        self.label_status = ctk.CTkLabel(
            self.main_frame,
            text="Loading Bi-Encoder & Cross-Encoder weights...",
            font=ctk.CTkFont(size=13),
        )
        self.label_status.grid(row=0, column=0, padx=20, pady=(12, 4), sticky="w")

        # Search bar row
        search_bar = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        search_bar.grid(row=1, column=0, padx=20, pady=6, sticky="ew")
        search_bar.grid_columnconfigure(0, weight=1)

        self.query_entry = ctk.CTkEntry(
            search_bar,
            placeholder_text="Enter technical query (e.g., 'InfoNCE temperature scaling hard negatives')...",
        )
        self.query_entry.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.query_entry.bind("<Return>", lambda _: self.run_search_event())

        self.btn_search = ctk.CTkButton(search_bar, text="Search & Rerank", width=140, command=self.run_search_event)
        self.btn_search.grid(row=0, column=1)

        # Retrieval parameter controls
        controls = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        controls.grid(row=2, column=0, padx=20, pady=4, sticky="ew")

        self.var_dense = tk.BooleanVar(value=True)
        self.var_sparse = tk.BooleanVar(value=True)
        self.var_rerank = tk.BooleanVar(value=True)

        ctk.CTkCheckBox(controls, text="HNSW Dense", variable=self.var_dense).pack(side="left", padx=(0, 12))
        ctk.CTkCheckBox(controls, text="BM25 Sparse", variable=self.var_sparse).pack(side="left", padx=(0, 12))
        ctk.CTkCheckBox(controls, text="Cross-Encoder Rerank", variable=self.var_rerank).pack(side="left", padx=(0, 18))

        self.ef_label = ctk.CTkLabel(controls, text="efSearch: 64")
        self.ef_label.pack(side="left", padx=(0, 6))

        self.ef_slider = ctk.CTkSlider(
            controls,
            from_=16,
            to=256,
            number_of_steps=15,
            width=160,
            command=lambda val: self.ef_label.configure(text=f"efSearch: {int(val)}"),
        )
        self.ef_slider.set(64)
        self.ef_slider.pack(side="left")

        # Output inspector
        self.textbox = ctk.CTkTextbox(self.main_frame, wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
        self.textbox.grid(row=3, column=0, padx=20, pady=(8, 16), sticky="nsew")

    # -------------------- Model & Encoding --------------------

    def _start_model_loader(self) -> None:
        def _load():
            try:
                tok = AutoTokenizer.from_pretrained(BI_ENCODER_MODEL)
                bi = BiEncoder(model_name=BI_ENCODER_MODEL).to(self.device)
                bi.eval()
                ce = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL, device=self.device)
                self.tokenizer = tok
                self.bi_encoder = bi
                self.cross_encoder = ce
                self.models_ready = True
                self.after(
                    0,
                    lambda: self.label_status.configure(
                        text=f"Ready on {self.device.upper()} | HNSW Vectors: {self.hnsw.size}"
                    ),
                )
            except Exception as exc:
                self.after(0, lambda: self.label_status.configure(text=f"Model load error: {exc}"))

        threading.Thread(target=_load, daemon=True).start()

    @torch.inference_mode()
    def _encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            emb = self.bi_encoder.encode(enc["input_ids"], enc["attention_mask"])
            all_vecs.append(emb.cpu().numpy())
        return np.ascontiguousarray(np.vstack(all_vecs), dtype=np.float32)

    def _add_documents(self, new_entries: list[dict]) -> None:
        if not new_entries:
            return
        start_id = len(self.docstore)
        ids = np.arange(start_id, start_id + len(new_entries), dtype=np.uint32)
        texts = [f"{item['title']}\n{item['text']}" for item in new_entries]

        vecs = self._encode_texts(texts)
        self.hnsw.add_batch(vecs, ids)
        self.bm25.add_documents(list(zip(ids.tolist(), texts)))

        for idx, item in zip(ids.tolist(), new_entries):
            self.docstore.append(
                {
                    "id": idx,
                    "title": item["title"],
                    "text": item["text"],
                    "source": item.get("source", "manual"),
                }
            )

        self.vectors = np.vstack([self.vectors, vecs]) if self.vectors.size else vecs
        self._save_persisted_state()

    # -------------------- Ingestion Handlers --------------------

    def ingest_corpus_event(self) -> None:
        if not self.models_ready:
            messagebox.showwarning("Wait", "Encoders are still initializing.")
            return
        paths = filedialog.askopenfilenames(
            filetypes=[("JSONL / Text / Markdown", "*.jsonl *.txt *.md")]
        )
        if not paths:
            return

        self.label_status.configure(text="Ingesting and indexing corpus files...")

        def _worker():
            entries = []
            for path in paths:
                fname = os.path.basename(path)
                if path.endswith(".jsonl"):
                    with open(path, "r", encoding="utf-8") as f:
                        for line_no, line in enumerate(f, start=1):
                            if not line.strip():
                                continue
                            obj = json.loads(line)
                            title = obj.get("title") or obj.get("id") or f"{fname}:{line_no}"
                            body = obj.get("abstract") or obj.get("text") or obj.get("body") or ""
                            if body.strip():
                                entries.append({"title": str(title), "text": body.strip(), "source": path})
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 30]
                    for idx, chunk in enumerate(chunks, start=1):
                        entries.append({"title": f"{fname} [chunk {idx}]", "text": chunk, "source": path})

            self._add_documents(entries)
            self.after(0, self._on_ingest_complete, len(entries))

        threading.Thread(target=_worker, daemon=True).start()

    def ingest_image_folder_event(self) -> None:
        if not self.models_ready:
            messagebox.showwarning("Wait", "Encoders are still initializing.")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return

        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        if not image_files:
            messagebox.showinfo("Empty", "No images found in selected folder.")
            return

        self.label_status.configure(text=f"Running local LaTeX OCR on {len(image_files)} images...")

        def _worker():
            if self.ocr_engine is None:
                self.ocr_engine = OCREngine()
            entries = []
            for idx, img_path in enumerate(image_files, start=1):
                fname = os.path.basename(img_path)
                self.after(
                    0,
                    lambda i=idx, n=fname: self.label_status.configure(
                        text=f"OCR [{i}/{len(image_files)}]: {n}"
                    ),
                )
                latex_text = self.ocr_engine.process(img_path)
                entries.append({"title": fname, "text": latex_text, "source": img_path})

            self._add_documents(entries)
            self.after(0, self._on_ingest_complete, len(entries))

        threading.Thread(target=_worker, daemon=True).start()

    def index_editor_text_event(self) -> None:
        if not self.models_ready:
            messagebox.showwarning("Wait", "Encoders are still initializing.")
            return
        raw = self.textbox.get("0.0", "end").strip()
        if len(raw) < 15:
            messagebox.showwarning("Too Short", "Paste technical text or LaTeX notes into the editor first.")
            return

        self.label_status.configure(text="Embedding and inserting into HNSW + BM25...")

        def _worker():
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            title = lines[0][:80]
            self._add_documents([{"title": title, "text": raw, "source": "editor"}])
            self.after(0, self._on_ingest_complete, 1)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ingest_complete(self, count: int) -> None:
        self.index_stats_label.configure(text=f"Indexed Docs: {self.hnsw.size}")
        self.label_status.configure(text=f"Indexed {count} document(s). Total vectors: {self.hnsw.size}")

    # -------------------- Search & Reranking --------------------

    def run_search_event(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            return
        if not self.models_ready:
            messagebox.showwarning("Wait", "Models are still loading.")
            return
        if self.hnsw.size == 0:
            messagebox.showinfo("Empty Index", "Ingest a corpus or index notes before searching.")
            return

        use_dense = self.var_dense.get()
        use_sparse = self.var_sparse.get()
        use_rerank = self.var_rerank.get()
        ef_search = int(self.ef_slider.get())

        if not use_dense and not use_sparse:
            messagebox.showwarning("Invalid Mode", "Enable at least HNSW Dense or BM25 Sparse.")
            return

        self.label_status.configure(text="Running hybrid retrieval & reranking...")

        def _worker():
            t0 = time.perf_counter()
            dense_hits, sparse_hits = [], []
            t_embed = t_hnsw = t_bm25 = t_rerank = 0.0

            if use_dense:
                te0 = time.perf_counter()
                q_vec = self._encode_texts([query])[0]
                t_embed = (time.perf_counter() - te0) * 1000.0

                th0 = time.perf_counter()
                dense_hits = self.hnsw.search(q_vec, k=30, ef_search=ef_search)
                t_hnsw = (time.perf_counter() - th0) * 1000.0

            if use_sparse:
                ts0 = time.perf_counter()
                sparse_hits = self.bm25.search(query, k=30)
                t_bm25 = (time.perf_counter() - ts0) * 1000.0

            fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=25)
            dense_scores = dict(dense_hits)
            sparse_scores = dict(sparse_hits)

            candidates = []
            for doc_id, rrf_score, prov in fused:
                doc = self.docstore[doc_id]
                candidates.append(
                    {
                        "id": doc_id,
                        "title": doc["title"],
                        "text": doc["text"],
                        "rrf_score": rrf_score,
                        "dense_sim": dense_scores.get(doc_id),
                        "sparse_score": sparse_scores.get(doc_id),
                        "dense_rank": prov["dense_rank"],
                        "sparse_rank": prov["sparse_rank"],
                        "ce_score": None,
                    }
                )

            if use_rerank and candidates:
                tr0 = time.perf_counter()
                passages = [f"{c['title']}\n{c['text']}" for c in candidates]
                ce_scores = self.cross_encoder.rerank(query, passages)
                for cand, score in zip(candidates, ce_scores):
                    cand["ce_score"] = score
                candidates.sort(key=lambda x: x["ce_score"], reverse=True)
                t_rerank = (time.perf_counter() - tr0) * 1000.0

            total_ms = (time.perf_counter() - t0) * 1000.0
            telemetry = {
                "query": query,
                "ef_search": ef_search,
                "embed_ms": t_embed,
                "hnsw_ms": t_hnsw,
                "bm25_ms": t_bm25,
                "rerank_ms": t_rerank,
                "total_ms": total_ms,
            }
            self.latency_history.append(telemetry)
            self.last_results = candidates[:10]
            self.after(0, self._render_search_results, self.last_results, telemetry)

        threading.Thread(target=_worker, daemon=True).start()

    def _render_search_results(self, results: list[dict], telemetry: dict) -> None:
        self.textbox.delete("0.0", "end")
        header = (
            f"QUERY: {telemetry['query']}\n"
            f"LATENCY: Total {telemetry['total_ms']:.2f} ms | "
            f"Embed {telemetry['embed_ms']:.2f} ms | "
            f"HNSW (ef={telemetry['ef_search']}) {telemetry['hnsw_ms']:.2f} ms | "
            f"BM25 {telemetry['bm25_ms']:.2f} ms | "
            f"Cross-Encoder {telemetry['rerank_ms']:.2f} ms\n"
            f"{'=' * 88}\n\n"
        )
        self.textbox.insert("end", header)

        for rank, item in enumerate(results, start=1):
            ce_str = f"{item['ce_score']:+.3f}" if item["ce_score"] is not None else "N/A"
            d_rank = f"#{item['dense_rank']}" if item["dense_rank"] else "-"
            s_rank = f"#{item['sparse_rank']}" if item["sparse_rank"] else "-"
            d_sim = f"{item['dense_sim']:.4f}" if item["dense_sim"] is not None else "-"

            block = (
                f"[{rank}] {item['title']} (doc_id={item['id']})\n"
                f"    CE Logit: {ce_str} | RRF: {item['rrf_score']:.5f} | "
                f"HNSW Rank: {d_rank} (cos={d_sim}) | BM25 Rank: {s_rank}\n"
                f"{'-' * 88}\n"
                f"{item['text'][:600]}\n\n"
            )
            self.textbox.insert("end", block)

        self.label_status.configure(
            text=f"Top-{len(results)} retrieved in {telemetry['total_ms']:.2f} ms (efSearch={telemetry['ef_search']})"
        )

    # -------------------- Archive & Telemetry Dashboard --------------------

    def open_archive(self) -> None:
        if not self.docstore:
            messagebox.showinfo("Archive", "No documents indexed yet.")
            return

        win = ctk.CTkToplevel(self)
        win.title(f"Indexed Corpus ({len(self.docstore)} docs)")
        win.geometry("950x700")
        win.attributes("-topmost", True)

        scroll = ctk.CTkScrollableFrame(win)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        for item in reversed(self.docstore[-100:]):
            card = ctk.CTkFrame(scroll)
            card.pack(pady=8, fill="x")

            ctk.CTkLabel(
                card,
                text=f"[ID {item['id']}] {item['title']}",
                font=("Arial", 13, "bold"),
            ).pack(anchor="w", padx=10, pady=(6, 2))

            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=6)

            src = item.get("source", "")
            if os.path.exists(src) and src.lower().endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(src)
                img.thumbnail((200, 200))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
                ctk.CTkLabel(row, image=ctk_img, text="").pack(side="left", padx=(0, 10))

            txt = ctk.CTkTextbox(row, height=150)
            txt.insert("0.0", item["text"])
            txt.pack(side="left", fill="x", expand=True)

    def open_telemetry_dashboard(self) -> None:
        if self.vectors.shape[0] < 5:
            messagebox.showinfo("Insufficient Data", "Index at least 5 documents to run Recall/QPS sweeps.")
            return

        if hasattr(self, "telemetry_win") and self.telemetry_win.winfo_exists():
            self.telemetry_win.lift()
            return

        win = ctk.CTkToplevel(self)
        self.telemetry_win = win
        win.title("Retrieval Engine Telemetry — Recall, QPS & Stage Latency")
        win.geometry("1120x740")
        win.attributes("-topmost", True)
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)

        loading_lbl = ctk.CTkLabel(
            win,
            text="Benchmarking C++ HNSW across efSearch = [16, 32, 64, 128, 256]...",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        loading_lbl.grid(row=0, column=0, pady=40)

        def _worker():
            sweep = self._compute_sweep_metrics()
            self.after(0, lambda: self._populate_telemetry_ui(win, loading_lbl, sweep))

        threading.Thread(target=_worker, daemon=True).start()

    def _compute_sweep_metrics(self) -> list[dict]:
        # Perturb sampled document vectors in 768D (sigma=0.025 -> ~0.82 cosine sim to target)
        # so queries behave like realistic unseen paraphrases rather than trivial self-hits.
        rng = np.random.default_rng(42)
        sample_n = min(64, self.vectors.shape[0])
        q_indices = np.linspace(0, self.vectors.shape[0] - 1, sample_n, dtype=int)
        base = self.vectors[q_indices]
        noisy = base + rng.normal(0.0, 0.025, size=base.shape).astype(np.float32)
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        queries = np.ascontiguousarray(noisy / np.maximum(norms, 1e-9), dtype=np.float32)

        # Exact brute-force cosine ground truth
        exact_sims = queries @ self.vectors.T
        k_eval = min(10, self.vectors.shape[0])
        gt_top_k = [set(np.argsort(-row)[:k_eval].tolist()) for row in exact_sims]

        ef_values = [16, 32, 64, 128, 256]
        rows = []

        for ef in ef_values:
            latencies_ms = []
            recalls, ndcgs, mrrs = [], [], []

            for q, gt_set in zip(queries, gt_top_k):
                t0 = time.perf_counter()
                hits = self.hnsw.search(q, k=k_eval, ef_search=ef)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                pred_ids = [doc_id for doc_id, _ in hits]
                recalls.append(len(set(pred_ids) & gt_set) / len(gt_set))
                qrels = {doc_id: 1 for doc_id in gt_set}
                ndcgs.append(ndcg_at_k(pred_ids, qrels, k=k_eval))
                mrrs.append(mrr_at_k(pred_ids, gt_set, k=k_eval))

            mean_ms = float(np.mean(latencies_ms))
            rows.append(
                {
                    "ef": ef,
                    "recall": float(np.mean(recalls)),
                    "ndcg": float(np.mean(ndcgs)),
                    "mrr": float(np.mean(mrrs)),
                    "p50_ms": float(np.percentile(latencies_ms, 50)),
                    "p95_ms": float(np.percentile(latencies_ms, 95)),
                    "qps": 1000.0 / max(mean_ms, 1e-6),
                }
            )
        return rows

    def _apply_ef_setting(self, ef_value: int) -> None:
        self.ef_slider.set(ef_value)
        self.ef_label.configure(text=f"efSearch: {ef_value}")
        self.label_status.configure(text=f"Updated HNSW search beam width to efSearch={ef_value}")

    def _populate_telemetry_ui(self, win: ctk.CTkToplevel, loading_lbl: ctk.CTkLabel, sweep: list[dict]) -> None:
        if not win.winfo_exists():
            return
        loading_lbl.destroy()

        # Pick fastest efSearch that hits >= 95% Recall@10 (or highest recall available)
        passing = [r for r in sweep if r["recall"] >= 0.95]
        best = passing[0] if passing else max(sweep, key=lambda r: r["recall"])
        vec_mb = (self.vectors.nbytes) / (1024 * 1024)

        # ---- Row 0: KPI Summary Cards ----
        kpi_frame = ctk.CTkFrame(win, fg_color="transparent")
        kpi_frame.grid(row=0, column=0, padx=16, pady=(14, 6), sticky="ew")
        for col in range(4):
            kpi_frame.grid_columnconfigure(col, weight=1)

        cards = [
            ("RECOMMENDED efSearch", f"ef = {best['ef']}",
             f"Recall@10: {best['recall'] * 100:.1f}% | {best['p50_ms']:.2f} ms"),
            ("THROUGHPUT AT OPTIMAL", f"{best['qps']:,.0f} QPS", f"p95 Latency: {best['p95_ms']:.2f} ms"),
            ("RANKING QUALITY", f"nDCG@10: {best['ndcg']:.3f}", f"MRR@10: {best['mrr']:.3f}"),
            ("INDEX FOOTPRINT", f"{self.hnsw.size:,} Vectors", f"768-dim FP32 ({vec_mb:.1f} MB)"),
        ]

        for idx, (title, big_val, sub_val) in enumerate(cards):
            card = ctk.CTkFrame(kpi_frame, corner_radius=10)
            card.grid(row=0, column=idx, padx=6, sticky="nsew")
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color="#9aa5b1").pack(
                anchor="w", padx=12, pady=(8, 0)
            )
            ctk.CTkLabel(card, text=big_val, font=ctk.CTkFont(size=20, weight="bold"), text_color="#4cc9f0").pack(
                anchor="w", padx=12, pady=2
            )
            ctk.CTkLabel(card, text=sub_val, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=12, pady=(0, 8))

        # ---- Row 1: Tabbed Charts & Breakdown ----
        tabs = ctk.CTkTabview(win)
        tabs.grid(row=1, column=0, padx=16, pady=(4, 14), sticky="nsew")
        tab_sweep = tabs.add("1. HNSW Recall vs. QPS Sweep")
        tab_live = tabs.add("2. Live Pipeline Latency Breakdown")

        # === TAB 1: Dual-Axis efSearch Curve + Clean Monospace Table ===
        tab_sweep.grid_columnconfigure(0, weight=1)
        tab_sweep.grid_rowconfigure(0, weight=1)

        fig, ax1 = plt.subplots(figsize=(10, 3.6), dpi=100)
        fig.patch.set_facecolor("#242424")
        ax1.set_facecolor("#2b2b2b")

        x = np.arange(len(sweep))
        ef_labels = [f"ef={r['ef']}" for r in sweep]
        recalls = [r["recall"] * 100.0 for r in sweep]
        qps_vals = [r["qps"] for r in sweep]

        color_rec = "#4cc9f0"
        color_qps = "#f72585"

        l1 = ax1.plot(x, recalls, marker="o", linewidth=2.5, markersize=8, color=color_rec, label="Recall@10 (%)")
        ax1.axhline(95.0, linestyle="--", linewidth=1, color="#80ed99", alpha=0.7, label="95% Recall Target")
        ax1.set_ylabel("Recall@10 (%)", color=color_rec, fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_rec)
        ax1.tick_params(axis="x", colors="white")
        ax1.set_xticks(x)
        ax1.set_xticklabels(ef_labels, fontsize=10, fontweight="bold")
        ax1.set_ylim(max(0, min(recalls) - 10), 103)
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        l2 = ax2.plot(x, qps_vals, marker="s", linewidth=2.5, markersize=7, color=color_qps, label="Throughput (QPS)")
        ax2.set_ylabel("Queries Per Second (QPS)", color=color_qps, fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=color_qps)
        ax2.set_ylim(0, max(qps_vals) * 1.18)

        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        leg = ax1.legend(lines, labels, loc="lower left", facecolor="#1e1e1e", edgecolor="#444444")
        for text in leg.get_texts():
            text.set_color("white")

        ax1.set_title("Effect of Beam Width (efSearch) on Accuracy vs. Throughput", color="white", fontsize=12, pad=10)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab_sweep)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=8, pady=(6, 4), sticky="nsew")

        # Action bar + readable table underneath chart
        action_bar = ctk.CTkFrame(tab_sweep, fg_color="transparent")
        action_bar.grid(row=1, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            action_bar,
            text=f"Takeaway: efSearch={best['ef']} hits {best['recall'] * 100:.1f}% recall at {best['p50_ms']:.2f} ms/query. Higher ef wastes CPU for minimal gain.",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        ctk.CTkButton(
            action_bar,
            text=f"Apply Recommended (ef={best['ef']})",
            width=190,
            command=lambda: self._apply_ef_setting(best["ef"]),
        ).pack(side="right")

        table_box = ctk.CTkTextbox(tab_sweep, height=145, font=ctk.CTkFont(family="Consolas", size=12))
        table_box.grid(row=2, column=0, padx=8, pady=(2, 8), sticky="ew")

        header = f"{'efSearch':<10} | {'Recall@10':<11} | {'nDCG@10':<9} | {'MRR@10':<9} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'QPS':<10} | Status\n"
        table_box.insert("end", header + "-" * 92 + "\n")
        for r in sweep:
            tag = "<-- OPTIMAL" if r["ef"] == best["ef"] else (
                "Fast / Lossy" if r["recall"] < 0.95 else "High Accuracy")
            line = (
                f"{r['ef']:<10} | {r['recall'] * 100:>8.2f}%  | {r['ndcg']:>8.4f} | {r['mrr']:>8.4f} | "
                f"{r['p50_ms']:>8.2f}  | {r['p95_ms']:>8.2f}  | {r['qps']:>9,.0f} | {tag}\n"
            )
            table_box.insert("end", line)
        table_box.configure(state="disabled")

        # === TAB 2: Real Search Latency Breakdown ===
        self._build_live_latency_tab(tab_live)

    def _build_live_latency_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        if not self.latency_history:
            ctk.CTkLabel(
                parent,
                text="No live queries recorded yet.\nRun 1 or more searches in the main window, then reopen this dashboard to inspect stage latency.",
                font=ctk.CTkFont(size=14),
            ).pack(expand=True)
            return

        recent = self.latency_history[-8:]
        q_labels = [
            f"Q{i + 1}: {item['query'][:24]}..." if len(item["query"]) > 24 else f"Q{i + 1}: {item['query']}"
            for i, item in enumerate(recent)
        ]
        embed_ms = np.array([r["embed_ms"] for r in recent])
        hnsw_ms = np.array([r["hnsw_ms"] for r in recent])
        bm25_ms = np.array([r["bm25_ms"] for r in recent])
        rerank_ms = np.array([r["rerank_ms"] for r in recent])

        fig, ax = plt.subplots(figsize=(10, 4.6), dpi=100)
        fig.patch.set_facecolor("#242424")
        ax.set_facecolor("#2b2b2b")

        y = np.arange(len(recent))
        ax.barh(y, embed_ms, color="#4361ee", label="1. Bi-Encoder Query Embed")
        ax.barh(y, hnsw_ms, left=embed_ms, color="#4cc9f0", label="2. C++ HNSW Search")
        ax.barh(y, bm25_ms, left=embed_ms + hnsw_ms, color="#80ed99", label="3. BM25 Sparse")
        ax.barh(y, rerank_ms, left=embed_ms + hnsw_ms + bm25_ms, color="#f72585", label="4. Cross-Encoder Rerank")

        ax.set_yticks(y)
        ax.set_yticklabels(q_labels, color="white", fontsize=10)
        ax.set_xlabel("Latency (milliseconds)", color="white", fontsize=11)
        ax.set_title("End-to-End Retrieval Latency Breakdown (Last 8 Live Queries)", color="white", fontsize=12)
        ax.tick_params(axis="x", colors="white")
        ax.grid(True, axis="x", alpha=0.2)

        leg = ax.legend(loc="lower right", facecolor="#1e1e1e", edgecolor="#444444")
        for text in leg.get_texts():
            text.set_color("white")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        last = recent[-1]
        bottleneck = max(
            [
                ("Bi-Encoder", last["embed_ms"]),
                ("C++ HNSW", last["hnsw_ms"]),
                ("BM25", last["bm25_ms"]),
                ("Cross-Encoder", last["rerank_ms"]),
            ],
            key=lambda x: x[1],
        )
        pct = (bottleneck[1] / max(last["total_ms"], 1e-6)) * 100.0
        ctk.CTkLabel(
            parent,
            text=f"Last Query Bottleneck: {bottleneck[0]} took {bottleneck[1]:.2f} ms ({pct:.1f}% of {last['total_ms']:.2f} ms total).",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=1, column=0, pady=(0, 8))

    def export_event(self) -> None:
        if not self.last_results:
            messagebox.showerror("No Results", "Run a search query first before exporting.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md")],
        )
        if not file_path:
            return
        entries = [
            {
                "title": f"{r['title']} (RRF={r['rrf_score']:.4f})",
                "content": r["text"],
            }
            for r in self.last_results
        ]
        Exporter.to_markdown(entries, filename=file_path)
        messagebox.showinfo("Exported", f"Saved top-{len(entries)} results to {file_path}")


if __name__ == "__main__":
    app = RetrievalEngineApp()
    app.mainloop()
    app.hnsw = None
    del app
