# Báo Cáo Phân Tích Đặc Trưng: Code Người Viết vs Code AI Sinh

## 1. Tổng Quan

Báo cáo này tổng hợp kết quả phân tích **45 đặc trưng thủ công (handcrafted features)** được trích xuất từ tập dữ liệu huấn luyện Task A — phân loại nhị phân: **Human (người viết)** vs **Machine (AI sinh)**.

Các đặc trưng được xếp hạng theo **|Cohen's d| effect size** — đo khoảng cách trung bình chuẩn hóa giữa phân bố của Human và AI code. Giá trị |d| càng lớn, khả năng phân biệt càng mạnh.

---

## 2. Kết Quả Chính

### 2.1. Top 15 Đặc Trưng Phân Biệt Mạnh Nhất

| Hạng | Đặc trưng | |d| | TB Human | TB AI | Giải thích |
|------|-----------|-----|----------|-------|------------|
| 1 | `indent_consistency` | **1.095** | 1.28 | 3.53 | AI có phương sai indent cao gấp ~2.8× |
| 2 | `max_indent` | **0.981** | 4.17 | 11.42 | AI lồng sâu gấp ~2.7× (cấu trúc class/function dài) |
| 3 | `space_ratio` | **0.925** | 0.163 | 0.256 | AI dùng nhiều dấu cách hơn theo tỷ lệ |
| 4 | `empty_line_ratio` | **0.849** | 0.046 | 0.138 | AI chèn dòng trống gấp 3× (format "sạch") |
| 5 | `avg_identifier_len` | **0.799** | 3.15 | 4.01 | AI đặt tên biến/hàm dài hơn, mô tả rõ hơn |
| 6 | `leading_tab_lines` | **0.634** | 13.54 | 1.36 | Human dùng tab nhiều gấp 10×, AI gần như chỉ dùng space |
| 7 | `unique_char_ratio` | **0.617** | 0.129 | 0.083 | Human có độ đa dạng ký tự cao hơn theo tỷ lệ |
| 8 | `unique_chars` | **0.601** | 39.63 | 49.05 | AI dùng nhiều ký tự riêng biệt hơn (do code dài hơn) |
| 9 | `compression_ratio` | **0.534** | 0.631 | 0.509 | AI code nén tốt hơn → pattern lặp, dễ đoán hơn |
| 10 | `leading_space_lines` | **0.528** | 5.36 | 20.10 | AI dùng dòng thụt đầu bằng space gấp ~4× |
| 11 | `tab_count` | **0.517** | 27.24 | 2.99 | Xác nhận: Human = tab, AI = space |
| 12 | `snake_case_ratio` | **0.495** | 0.019 | 0.052 | AI dùng nhiều snake_case hơn (phong cách Python) |
| 13 | `comment_ratio` | **0.453** | 0.006 | 0.070 | AI thêm comment gấp ~12× |
| 14 | `shannon_entropy` | **0.434** | 4.524 | 4.370 | Human code có entropy cao hơn (khó đoán hơn) |
| 15 | `double_quotes` | **0.428** | 0.672 | 4.863 | AI dùng ngoặc kép gấp ~7× (output nhiều string) |

### 2.2. Phân Loại Theo Mức Độ Phân Biệt

- **Rất mạnh (|d| > 0.8):** `indent_consistency`, `max_indent`, `space_ratio`, `empty_line_ratio`, `avg_identifier_len`
- **Mạnh (|d| 0.5–0.8):** `leading_tab_lines`, `unique_char_ratio`, `compression_ratio`, `leading_space_lines`, `tab_count`
- **Trung bình (|d| 0.4–0.5):** `snake_case_ratio`, `comment_ratio`, `shannon_entropy`, `double_quotes`

---

## 3. Phân Tích Theo Nhóm Pattern

### 3.1. Phong Cách Thụt Lề (Indentation Style) — Tín Hiệu Mạnh Nhất

Sự khác biệt **tabs vs spaces** là pattern nhất quán nhất:

- **Human code**: thụt lề bằng tab (`leading_tab_lines` = 13.5, `tab_count` = 27.2)
- **AI code**: thụt lề bằng space (`leading_space_lines` = 20.1, `space_ratio` = 0.256)

**Nguyên nhân**: Hầu hết LLM được huấn luyện trên dữ liệu GitHub nơi spaces chiếm ưu thế, và model mặc định xuất spaces.

> ⚠️ **Lưu ý**: Tín hiệu này có thể **phụ thuộc ngôn ngữ** (ví dụ Go quy ước dùng tabs). Model cần normalize theo ngôn ngữ lập trình.

### 3.2. Định Dạng Code — AI "Sạch" Hơn Nhưng Sâu Hơn

- AI tạo ra **nhiều dòng trống hơn** (phân cách giữa các block), **nhiều comment hơn**, và **tên biến dài hơn**
- Tuy nhiên, AI code có **độ sâu indent cao hơn** (`max_indent` = 11.4 vs 4.2) — có thể do sinh cấu trúc class/method dài dòng

### 3.3. Tính Dự Đoán Được (Predictability) — AI Code Lặp Pattern Hơn

| Đặc trưng | Human | AI | Ý nghĩa |
|-----------|-------|----|---------|
| `compression_ratio` | 0.631 | 0.509 | AI nén tốt hơn → lặp nhiều hơn |
| `shannon_entropy` | 4.524 | 4.370 | Human code "ngẫu nhiên" hơn |
| `unique_char_ratio` | 0.129 | 0.083 | Human dùng ký tự đa dạng hơn |

Ba đặc trưng này kết hợp tạo thành **"dấu vân tay dự đoán"** (predictability fingerprint) của AI code.

### 3.4. Quy Ước Đặt Tên — AI Theo Phong Cách "Sách Giáo Khoa"

- **`avg_identifier_len`**: AI = 4.01 vs Human = 3.15 → AI đặt tên mô tả hơn
- **`snake_case_ratio`**: AI = 0.052 vs Human = 0.019 → AI ưu tiên snake_case (Pythonic)
- **`comment_ratio`**: AI = 0.070 vs Human = 0.006 → AI thêm comment thường xuyên hơn ~12×

---

## 4. Đề Xuất Cho Mô Hình Hybrid

### 4.1. Chọn Đặc Trưng Cho Nhánh Thủ Công (Feature Branch)

**Tầng 1 — Bắt buộc (|d| > 0.8):**
- `indent_consistency`, `max_indent`, `space_ratio`, `empty_line_ratio`, `avg_identifier_len`

**Tầng 2 — Tín hiệu mạnh (|d| 0.5–0.8):**
- `leading_tab_lines`, `unique_char_ratio`, `compression_ratio`, `leading_space_lines`, `tab_count`

**Tầng 3 — Tín hiệu trung bình (|d| 0.4–0.5):**
- `snake_case_ratio`, `comment_ratio`, `shannon_entropy`, `double_quotes`

### 4.2. Gợi Ý Kỹ Thuật Đặc Trưng Bổ Sung

1. **Normalize theo ngôn ngữ**: Chia đặc trưng cho trung bình theo ngôn ngữ để loại bỏ bias
2. **Đặc trưng tỷ lệ kết hợp**: `tab_count / (tab_count + space_count)` — gộp tab vs space thành 1 tín hiệu
3. **Đặc trưng tương tác**: `compression_ratio × shannon_entropy` — điểm dự đoán kết hợp

### 4.3. Kiến Trúc Hybrid Đề Xuất

```
Input: chuỗi code thô
    ├── Nhánh Neural: CodeBERT/UniXcoder → [768-dim embedding] → hiểu ngữ nghĩa
    ├── Nhánh Đặc Trưng: 45 features → BatchNorm → [64-dim] → nắm bắt cấu trúc
    └── Hợp Nhất: Concatenate → Linear(832, 256) → ReLU → Dropout → Linear(256, 2) → Softmax
```

### 4.4. Điểm Yếu Tiềm Ẩn

- **Tabs/spaces**: Có thể overfit vào công cụ format, không phản ánh thực sự AI hay không
- **Comment ratio**: Có thể bị qua mặt bằng post-processing (xóa/thêm comment)
- Cần kiểm tra đặc trưng có **nhất quán qua các generator và ngôn ngữ khác nhau** hay không

---

## 5. Tổng Kết

| Nhóm Pattern | Đặc Trưng Chính | Mức Độ |
|--------------|-----------------|--------|
| Phong cách thụt lề | `indent_consistency`, `max_indent`, `space_ratio`, tabs vs spaces | Rất mạnh |
| Định dạng code | `empty_line_ratio`, `comment_ratio`, `double_quotes` | Mạnh |
| Tính dự đoán | `compression_ratio`, `shannon_entropy`, `unique_char_ratio` | Mạnh vừa |
| Quy ước đặt tên | `avg_identifier_len`, `snake_case_ratio` | Trung bình |

**Kết luận**: 45 đặc trưng đã trích xuất cung cấp **nền tảng vững chắc** cho nhánh thủ công trong mô hình hybrid. Riêng top 5 đặc trưng (|d| > 0.8) đã nắm bắt được sự khác biệt cấu trúc đáng kể. Kết hợp với embedding ngữ nghĩa từ neural network, hướng tiếp cận hybrid này có tiềm năng đạt hiệu suất cao cho Task A.
