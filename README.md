# SemEval-2026 Task 13 — Subtask A
## Detecting Machine-Generated Code (Binary Classification)

---

## Thông tin học viên

| Mục | Thông tin |
|---|---|
| Họ và tên | **Nguyễn Trường Thịnh** |
| Mã số sinh viên | **25C11066** |
| Trường | **Trường Đại học Khoa học Tự nhiên (HCMUS)** |
| Cuộc thi | SemEval-2026 Task 13 — Subtask A |
| Trang nộp bài (Kaggle) | https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-a/submissions |

> Trang submissions ở trên là minh chứng performance cho hệ thống được mô tả trong báo cáo này (cần đăng nhập Kaggle để xem điểm leaderboard cá nhân).

---

## Tổng quan kết quả

Hệ thống nộp cuối cùng là một **soft-voting ensemble 3 thành phần**, đạt **macro F1 = 0.527** trên hidden test set của SemEval-2026 Task 13 — Subtask A.

Ba thành phần:
1. **Language-robust gradient-boosted ensemble** trên 75 đặc trưng thủ công (đã loại bỏ các đặc trưng bị shift mạnh giữa train/test).
2. **CodeBERT fine-tuned** với ngưỡng quyết định calibrated `τ = 0.93`.
3. **IsolationForest + ComplementNB hybrid** trên 20 đặc trưng phong cách (style features).

Xác suất thô từ mỗi mô hình được rank-normalise về `[0, 1]`, lấy trung bình, sau đó đặt ngưỡng sao cho tỉ lệ AI dự đoán khớp với phân phối train (52% AI).

---

## Experiments and Results

### Bảng 1 — Macro F1 trên validation (100K) và hidden test

| System | #Feat | Val F1 | Test F1 |
|---|---:|---:|---:|
| XGBoost (handcrafted) | 76 | 0.982 | — |
| LightGBM (handcrafted) | 76 | 0.984 | — |
| CatBoost (handcrafted) | 76 | 0.981 | — |
| Gradient-Boosted Ensemble (original, 83-feat) | 83 | 0.989 | 0.254 |
| Gradient-Boosted Ensemble (language-robust) | 75 | 0.973 | 0.472 |
| CodeBERT (fine-tuned, τ=0.93) | — | 0.954 | 0.451 |
| IsolationForest + ComplementNB | 20 | 0.823 | 0.535 |
| LR Stacking (Ensemble + IF+CNB) | — | 0.990 | 0.359 |
| **Soft-Voting (Ensemble + CodeBERT + IF+CNB)** | — | **0.986** | **0.527** |

> Cuộc thi chỉ công bố một điểm test tổng, nên các cột per-regime (Seen/Unseen × language/generator) không có dữ liệu công khai. **In đậm**: hệ thống nộp cuối cùng.

### Validation–test gap

Validation F1 **anti-correlated mạnh** với leaderboard: mô hình đơn lẻ tốt nhất (gradient-boosted ensemble 83 feature) đạt 0.989 trên validation nhưng chỉ 0.254 trên hidden test — chênh **0.735**. Ngược lại, baseline yếu hơn nhiều là `IsolationForest + ComplementNB` (val 0.823) lại đạt **0.535** trên hidden test, vượt mọi mô hình đơn lẻ phức tạp hơn mà chúng tôi đã thử.

### Feature-shift diagnosis

Tính standardized mean difference `|d|` giữa phân phối feature train vs. test. Các feature bị shift mạnh nhất đều liên quan tới **identifier và syntax**:

| Feature | \|d\| |
|---|---:|
| `avg_identifier_len` | 1.76 |
| `id_avg_len` | 1.69 |
| `id_short_ratio` | 1.54 |
| `line_entropy_std` | 1.40 |
| `camel_case_ratio` | 1.26 |
| `burstiness` | 1.15 |
| `punct_density` | 1.05 |

Ngược lại, các feature liên quan tới whitespace gần như không shift: `tab_count` (`|d|=0.000`), `leading_tab_lines` (0.029), `comment_ratio` (0.010).

Nguyên nhân: **test set có các ngôn ngữ brace-heavy chưa thấy trong train** — JavaScript, TypeScript, Go, Ruby, C# — với naming convention và syntax density khác biệt so với corpus train (91.5% Python, 4.7% C++, 3.9% Java).

### Mitigation

Loại bỏ 7 feature có `|d| > 1.0`, cộng với `overall_ppl` (perplexity quá đắt để tính trên test 500K mẫu), thu được language-robust ensemble. Test F1 tăng **+0.218** (`0.254 → 0.472`) đổi lại **−0.016** trên validation.

### Ensemble effects

Rank-average soft voting của 3 thành phần cải thiện thêm **+0.055** so với mô hình đơn lẻ tốt nhất (`0.472 → 0.527`).

LR stacking meta-learner trên validation probabilities chỉ đạt **0.359** — thấp hơn đáng kể vì gradient-boosted ensemble đã được early-stopped trên validation, khiến validation probabilities gần như hoàn hảo (val F1 = 0.973). Meta-learner gán hệ số trội (`≈ 10.9`) cho thành phần này, làm sụp đổ tính đa dạng (diversity) của ensemble.

### Cross-generator robustness (LOGO CV)

Leave-One-Generator-Out cross-validation trên một XGBoost nhẹ qua **34 generator family** (xem `logo_cv_results.csv`):

| Metric | Value |
|---|---:|
| Generators evaluated | 34 |
| Mean macro F1 | 0.949 |
| Std (σ) | 0.014 |
| Min F1 (`deepseek-coder-1.3b-instruct`) | 0.909 |
| Max F1 (`Phi-3-medium-4k-instruct`) | 0.977 |

Vì degradation do generator-shift bị chặn ở mức **~0.04**, khoảng cách 0.2–0.5 quan sát trên hidden test được quy cho **language shift**, không phải generator shift.

---

## Cấu trúc repo (chính)

| File / dir | Mô tả |
|---|---|
| `feature_extractor.py`, `extract_*.py` | Trích xuất đặc trưng thủ công cho train/val/test |
| `train_v6_ensemble.py`, `train_v7_full_ensemble.py` | Train gradient-boosted ensemble (83-feat & 75-feat language-robust) |
| `finetune_codebert.py`, `postprocess_codebert.py` | Fine-tune CodeBERT và calibrate ngưỡng τ=0.93 |
| `train_v8_if_cnb_tuned.py`, `if-cnb-style-only.ipynb` | IsolationForest + ComplementNB hybrid |
| `blend_submissions.py`, `quick_ensemble_fix.py` | Rank-average soft-voting ensemble cuối cùng |
| `stacking_meta.py` | LR stacking meta-learner (baseline so sánh) |
| `logo_cv.py`, `logo_cv_results.csv` | Leave-One-Generator-Out cross-validation |
| `_diag_shift.py`, `diagnose.py` | Tính `|d|` shift train/test, vẽ Cohen's d |
| `selected_features.txt` | Danh sách feature dùng cho language-robust ensemble |
| `REPORT.md` | Báo cáo phân tích đặc trưng (Cohen's d) chi tiết |

---

## Citation

Bộ dữ liệu của task này dựa trên các công trình trước đó của ban tổ chức:

```bibtex
@inproceedings{orel2025droid,
  title={Droid: A resource suite for ai-generated code detection},
  author={Orel, Daniil and Paul, Indraneil and Gurevych, Iryna and Nakov, Preslav},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={31251--31277},
  year={2025}
}

@inproceedings{orel-etal-2025-codet,
  title={{C}o{D}et-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings},
  author={Orel, Daniil and Azizov, Dilshod and Nakov, Preslav},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025},
  pages={10570--10593},
  url={https://aclanthology.org/2025.findings-acl.550/}
}
```
