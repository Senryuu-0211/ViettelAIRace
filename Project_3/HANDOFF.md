# HANDOFF — Trạng thái toàn bộ (đọc là hiểu ngay)

> File này tóm tắt mọi thứ đang diễn ra để nhanh nắm bắt lại sau khi nén context / sang phiên mới.
> Repo: `d:\ViettelAIRace\MedicalKnowledgeRetrieval` · Remote: `github.com/Senryuu-0211/ViettelAIRace`

---

## 0. Quy tắc & môi trường (QUAN TRỌNG — đọc trước)

**Giao tiếp:**
- Gọi người dùng là **"Mr. Senryuu"** ít nhất 1 lần mỗi câu trả lời. Ưu tiên tiếng Việt.
- Trước khi xin chạy lệnh terminal, ghi 1 dòng phân loại: `Lv: SAFE` (chỉ đọc) / `MODERATE` (đổi cục bộ, hoàn tác được) / `CRITICAL` (khó hoàn tác/outward: rm -rf, push, xóa data).

**Môi trường (Windows):**
- **Python cho Project 2:** dùng conda env `firstconda` → `C:\Users\Nguyen\miniconda3\envs\firstconda\python.exe` (KHÔNG dùng system Python312 — thiếu package). `conda` không có trong PATH của shell → gọi python.exe trực tiếp.
- Console cp1252 làm hỏng tiếng Việt → set `PYTHONIOENCODING=utf-8`.
- **Máy này KHÔNG có GPU** (CPU-only) + RAM trống ít (~4-8GB). Ollama chạy được model nhỏ.
- **Docker CLI có (v29) nhưng daemon (Docker Desktop) đang TẮT.**
- Ollama models có sẵn: `qwen3.5:2b-q4_K_M`, `qwen3.5:4b-q4_K_M`, `gemma4:e2b-it`, `phi3:mini`, `bge-m3`, `paraphrase-multilingual`.

**3 project trong repo:**
- `Project 1/` — 3D Gaussian Splatting (NVS). Không liên quan, không đụng.
- `Project 2/` — Viettel AI Race: Medical Knowledge Retrieval (NLP y khoa tiếng Việt).
- `Project_3/` — LLM Inference Optimization Challenge (tối ưu vLLM serving trên H200).

---

## 1. Git — trạng thái nhánh

| Nhánh | Nội dung |
|---|---|
| `develop` | **Bản chính hiện tại** = việc Neo4j + Project_3 của **bạn bạn**, đã merge xong (conflict lấy theirs) |
| `develop-hybrid` | Việc **hybrid ICD của tôi** (Project 2) — an toàn trên origin, để reconcile sau |
| `main` | Cũ |

- Đã có 1 lần **merge conflict** giữa 2 hướng ICD (hybrid vs Neo4j) → giải quyết trên `develop` bằng cách **lấy bản bạn bạn (theirs)**; hybrid giữ riêng ở `develop-hybrid`. Không mất gì.
- **`develop` đang ahead 2 so với origin/develop** — CHƯA push (chờ Mr. Senryuu quyết vì là nhánh chung).

---

## 2. Project 2 — Medical Knowledge Retrieval (NLP)

**Bài toán:** trích khái niệm y khoa từ text lâm sàng tiếng Việt tự do → mỗi khái niệm: `text` (substring nguyên văn), `position` [start,end] char 0-index, `type` (5 nhãn), `assertions` (isNegated/isFamily/isHistorical, chỉ THUỐC/CHẨN_ĐOÁN/TRIỆU_CHỨNG), `candidates` (ICD-10 cho CHẨN_ĐOÁN, RxNorm cho THUỐC).
**Chấm:** `0.3·text(WER) + 0.3·assertions(Jaccard) + 0.4·candidates(Jaccard weighted)`. **Sai `type` = tính 2 lần, cả 2 = 0đ cả 3 metric** → type là nền móng.
**Ràng buộc:** self-host ≤9B, không API LLM ngoài (nhưng LLM ngoài để TẠO DATA thì được); reproducible cho BTC dựng lại.

**2 hướng đã rẽ nhánh:**
- **Hybrid (của tôi, `develop-hybrid`):** qwen3.5 prompt extraction + ICD linker lexical/embedding + RxNorm qua RxNav API.
- **Neo4j (của bạn bạn, `develop`):** knowledge-graph cho ICD linking (`LLM-solution/`, `icd10_knowledge_graph.json`, `neo4j_linker.py`).

**Đã làm được (hướng hybrid):**
- **KB ICD-10 TT06 sạch:** crawl từ `icd.kcb.vn` (API thật: `https://ccs.whiteneuron.com/api/ICD10_TT06/`, duyệt `childs/<model>?id=<id>`: chapter→section→type→disease). Kết quả **18.237 mã + 2.090 category** có tên + SET expansion (K21→{K21.0,K21.9}). File: `data/icd10_tt06.json`. Crawler: `tools/crawl_icd10_tt06.py`.
- **ICD linker hybrid** (`pipeline/linker/icd10.py`): lexical (rapidfuzz) + embedding (cosine in-memory) + fuse min-max + SET expansion. **Mặc định lexical-only** (`USE_EMBEDDING=0`) vì embedding trên CPU quá chậm (7-11s/query) và không cải thiện (baseline lexical 66.7% ~ hybrid). Bench: `tools/test_icd_linker.py`.
- **Embedder** (`pipeline/linker/embedder.py`): gọi thẳng Ollama `/api/embed` (batch), KHÔNG dùng langchain_ollama (bản 1.1.0 lỗi `/tokenize`).
- **Scorer** (`scorer.py`): đúng công thức, align entity theo `(text, type)`, WER + Jaccard weighted + luật empty-set.
- **Model extraction:** `qwen3.5:4b-q4_K_M`. **QUAN TRỌNG: phải `reasoning=False`** trong ChatOllama — nếu không qwen3.5 chèn token thinking → 0 entity + chậm 18 phút/file. Với reasoning off: ~2.6 phút/file, baseline **FINAL ~0.50** trên gold drug-list (`tools/eval_druglist.py`).

**Còn nợ / cần làm:**
- Fix RxNorm linker (bug ép `SCD+SBD` → mất mã ingredient như 7597; verify bằng 13 anchor).
- **Chưa có gold dev set thật** (mới chấm được trên 1 ví dụ drug-list từ đề) → cần label vài file `input/` qua UI hoặc hand-label.
- Prompt tuning: lỗi chính "thuốc + điều trị X" bị gán nhầm TRIỆU_CHỨNG (đã thử sửa 1 lần → regress, đã revert).
- Quyết định: **train hay không** — luật chỉ *giả định/khuyến khích* train ("cần tạo thêm data nhằm huấn luyện mô hình"), **KHÔNG bắt buộc**; prompt-only vẫn hợp lệ. Train (fine-tune NER+assertion encoder) để ăn điểm, cần GPU.
- Reconcile hybrid vs Neo4j (quyết định của team).

---

## 3. Project_3 — LLM Inference Optimization (đang tập trung)

**Bài toán:** tối ưu vLLM serving cho **LiquidAI/LFM2.5-1.2B-Instruct** (kiến trúc **hybrid**: 10 conv + 6 attention layer) trên **1 MiG H200 (18GB VRAM, 3 CPU core, 8GB RAM)**. Nộp Docker image (endpoint OpenAI-compatible). **Chỉ được dùng vLLM.** Baseline image `vllm/vllm-openai:v0.22.1`.
**Chấm:** `Score = 100 × ERS × f(Δ)`.
- ERS từ TTFT+TPOT (Round 2: `F_ttft=10ms, C_ttft=400ms, F_tpot=1ms, C_tpot=10ms, γ=2, w=0.5`). Fail/timeout/0-token → 0.
- f(Δ) accuracy gate (GPQA Diamond vs BF16, baseline acc=0.4): Δ≤0.10→1.0, →0 tại Δ≥0.16. Chỉ chấm SAU vòng online, trên ≤5 submission đội chọn.
**Anti-cheat:** cấm pre-bake/hardcode, dual-path, gaming metrics, gọi mạng ngoài, sửa tokenizer/weights, tráo image sau nộp.

**Kết quả hiện tại (submission AWQ): `59.33/100`**
- ERS 59.33 · **f_delta=1 (accuracy_drop=0)** · TTFT p50=55ms/p95=79ms (tốt, xa ceiling 400) · **TBT/TPOT median=4ms (BOTTLENECK)** · 6/420 fail.
- Config: `docker-compose.yml` — AWQ INT4 (compressed-tensors, conv layers giữ BF16), CUDA graph FULL (chạy thật), async-scheduling, prefix-caching, `max-model-len=16384`, `max-num-seqs=16`, `gpu-mem=0.90`.

**Phân tích:** bottleneck = **TPOT**, bản chất **CPU-dispatch-bound** (3 core yếu, CUDA graph đã tối đa). Với γ=2, **mỗi 1ms TPOT ≈ +8 điểm** (4ms→2ms ≈ ERS ~76). **100đ bất khả thi vật lý** (cần MỌI request TTFT≤10ms & TPOT≤1ms & 0 fail); **trần thực tế ~85-90**.

**Đòn bẩy #1 = speculative decoding** (prompt-lookup/ngram, KHÔNG cần model phụ; multi-turn lặp history → acceptance khá). **NHƯNG crash/corrupt trên hybrid ở vLLM 0.22.1.**

**Chẩn đoán crash (từ GitHub vLLM):**
1. 🔴 Gốc: ngram spec làm **hỏng conv/SSM-state** (không rollback khi reject) — issue **#39273**. **Fix ở PR #40738** → **phải nâng vLLM** (0.22.1 chưa có).
2. `--async-scheduling` + spec: không tương thích.
3. `prompt_lookup_min=2` → corruption (#40875) → dùng **=8**.
4. KV-quant × spec → degenerate loop (#40831) → **không FP8 KV** khi spec.

**Deliverables đã tạo (cho bạn bạn test H200):**
- `Project_3/Dockerfile.spec` → `FROM vllm/vllm-openai:v0.25.1` (tag đã verify tồn tại; chứa fix; download ~**8.2GB nén**, on-disk mới ~15GB) + COPY awq_model.
- `Project_3/docker-compose-spec-v2.yml` → ngram_gpu spec đã sửa 4 lỗi, giữ AWQ+CUDA graph FULL+prefix-cache, bỏ async, không FP8 KV.
- `Project_3/build_and_push_spec.ps1` → build+push turnkey (sửa `DOCKERHUB_USER` + `docker login`).
- `Project_3/SPEC_DECODING_FINDINGS.md` → chẩn đoán đầy đủ + **GATE bắt buộc: verify output spec == greedy (lossless) trước khi nộp** kẻo trượt Accuracy Gate + nguồn.

**Trạng thái build:** CHƯA build (Docker daemon tắt). **Nên để bạn bạn build+test+push trên máy H200** (nơi verify được). Fallback an toàn = submission `59.33` (không đụng).

**Còn nợ Project_3:** vá 6 fail (nghi OOM/timeout, không có log); tune operating point qua `trace_tools/simulate_scheduler.py` (chạy CPU được).

---

## 4. Quyết định đang treo (chờ Mr. Senryuu)
1. **Project_3:** bạn bạn build image spec trên H200 & test? Hay Mr. Senryuu bật Docker Desktop + cho Docker Hub username để build ở đây (nhưng máy này không test được).
2. **Git:** push `develop` lên `origin/develop` (nhánh chung) không?
3. **Project 2:** chọn hướng hybrid hay Neo4j; dựng gold dev set; train hay không.

---
*Cập nhật lần cuối theo phiên làm việc hiện tại. Chi tiết đề bài: `Project 2/.claude/CLAUDE.md`, `Project_3/NEW_PLAN.md` + `SUMMARY.md`.*
