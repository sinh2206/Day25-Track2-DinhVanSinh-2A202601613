# Bài Viết Phân Tích — Lab 25: GPU FinOps Optimization
**Sinh viên:** Đinh Văn Sinh — 2A202601613  
**Track 2 Infrastructure · Day 25 · AICB Phase 2**

---

## 1. Baseline vs. Optimized — Chi phí trước và sau

| Chỉ số | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Chi phí tháng | $27,133 | $14,626 | **$12,507 (46%)** |
| $/1M-token (inference) | $6.488 | $1.126 | **-82.6%** |
| Chi phí ngày (inference) | $48.87 | $8.48 | -82.6% |

**Kết luận:** Từ 4 đòn bẩy tối ưu hóa, chúng tôi đã cắt giảm 46% tổng chi phí GPU hàng tháng (từ $27,133 xuống $14,626), đồng thời giảm chi phí inference đo theo $/1M-token xuống còn 83% so với ban đầu.

---

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm | % trong tổng | Ghi chú |
|---|---|---|---|
| **Inference (cascade/cache/batch)** | $1,212/tháng | 9.7% | 82.6% savings so với baseline inference |
| **Purchasing (spot/reserved)** | $10,040/tháng | **80.3%** | Đòn bẩy lớn nhất |
| **Right-size util-lies** | $655/tháng | 5.2% | Hạ GPU có GPU-Util lie xuống tier thấp hơn |
| **Kill idle GPUs** | $600/tháng | 4.8% | Tắt GPU chạy không, util < 10% |

**Purchasing là đòn bẩy lớn nhất (80.3%)** vì NimbusAI đang trả on-demand $2.5/giờ cho H100 trong khi nhiều workload có thể chuyển sang spot ($1.5/giờ, -40%) hoặc reserved ($1.4/giờ, -44%). Với 8 jobs tổng cộng, tổng GPU-hours rất lớn nên discount nhỏ mang lại savings tuyệt đối lớn.

**Inference là đòn bẩy nhỏ nhất tuyệt đối ($1,212)** nhưng lại có tỷ lệ savings cao nhất (82.6%). Stacking cascade + caching + batch:
- Cascade: định tuyến request sang model nhỏ ($0.20/$0.40 vs $3/$15 per 1M tokens) → ~15× rẻ hơn
- Prompt caching: 31.9% input tokens đã được cache → 90% discount trên phần đó
- Batch API: các request không cần real-time → 50% discount
- Combined: `discount_stack(batch=True, cache_hit_frac=1.0) = 0.05` = chỉ tốn 5% giá gốc!

---

## 3. GPU-Util Lie — Tại sao đây là vấn đề tài chính nghiêm trọng

**GPU bị phát hiện:** `gpu-h100-4` và `gpu-a10g-1`
- `gpu-h100-4`: GPU-Util = **98.2%** nhưng MFU = **0.194 (19.4%)**
- `gpu-a10g-1`: GPU-Util = **96.9%** nhưng MFU = **0.268 (26.8%)**

**Cơ chế của "lie":** `nvidia-smi` báo GPU-Util = % thời gian GPU đang có ít nhất 1 kernel chạy trên SM (Streaming Multiprocessor). Điều này **không đo hiệu quả tính toán**. GPU có thể "bận" 98% thời gian vì:
- **Memory stall:** GPU đang chờ data từ HBM (băng thông cạn), clock cycle trôi qua nhưng không compute
- **Kernel launch overhead:** CPU không kịp dispatch kernel mới, GPU idle micro-giây giữa các kernel nhưng vẫn báo "busy"
- **Load imbalance trong grid:** một số SM chạy xong sớm, chờ các SM khác → GPU báo busy nhưng compute thực thấp

**Tác động tài chính:** H100 on-demand = $2.5/giờ. MFU 19.4% có nghĩa là bạn chỉ nhận được 19.4% FLOPs mà bạn đã trả. Tương đương trả $2.5/giờ nhưng chỉ nhận được giá trị của GPU $0.485/giờ → **lãng phí $2.015/giờ**.

**Đo đúng:** Dùng `achieved_tflops / peak_tflops_fp16` để tính MFU thực. Đây là chỉ số không thể nói dối.

---

## 4. Phần mở rộng đã thực hiện

### Extension 1 — Enhanced `recommend_tier()` với GPU interruption rates + 1yr vs 3yr

**File sửa:** `finops/pricing.py`  
**Hàm mới:** `recommend_tier()` cải tiến + `recommend_tier_detailed()`

**Cải tiến:**
1. **GPU-specific spot interrupt rates** (data-driven): H100=2%, A100=5%, A10G=8%, L4=6% thay vì boolean đơn giản
2. **1yr vs 3yr reserved decision tree** dựa trên `job_days`:
   - `job_days < 365` → recommend 1yr commitment (tránh over-commit)
   - `job_days >= 365` → recommend 3yr để maximize discount

**Kết quả đo lường:**

| Chính sách | Chi phí (30 ngày) | Ghi chú |
|---|---|---|
| Naive (luôn 3yr nếu stable) | $14,562 | Over-commits 30-day jobs |
| Smart (1yr cho short jobs) | $17,679 | $3,117 đắt hơn nhưng... |

**Insight quan trọng:** Smart policy đắt hơn $3,117 nhưng **tránh rủi ro over-commitment**. Nếu 3yr reserved cho 30-day jobs, sau khi job xong còn 335 ngày idle phải trả tiền. Ví dụ: `job-infer-chat` (A10G, $3,456/30 ngày ở 3yr) nếu commit 3yr = phải trả 3yr × $0.6/hr × 24h × 365 × 6 GPUs = **$94,608 total** dù chỉ cần 30 ngày. 1yr ($80,352) tiết kiệm $14,256 so với 3yr nếu không gia hạn.

**GPU interruption insight:** A10G có interrupt rate 8% (4× so với H100 2%), nhưng vẫn dưới ngưỡng 10% → vẫn OK cho spot. Nếu một GPU mới có rate > 10%, policy sẽ tự động switch sang on_demand — điều mà baseline policy không làm được.

---

### Extension 3 — `cache_is_worth_it()` với break-even analysis

**File sửa:** `finops/pricing.py`  
**Hàm mới:** `cache_is_worth_it()`, `cache_break_even_reads()`

**Logic:** Cache chỉ có lợi khi `total_read_savings > write_cost`:
- Mỗi read saves `(1 - read_discount) × price = 90% × price`
- Break-even: `min_reads = write_cost / (1 - read_discount) = 1 / 0.9 = 1.11×`

**Kết quả đo lường trên dataset:**

| Chỉ số | Giá trị |
|---|---|
| Requests có cache hit | 2,400 / 2,400 (100%) |
| Avg cache hit fraction | 31.9% của input tokens |
| Estimated avg reads/prefix | 2,400× |
| Break-even threshold | 1.11× |
| Cache justified? | **YES ✅** |

**Insight:** Dataset NimbusAI có 100% requests đọc cache (2,400 reads với cùng prefixes). Đây là trường hợp lý tưởng — system prompt hoặc RAG context được reuse cao. Với avg reads = 2,400×, cache cực kỳ có lợi.

**Câu trả lời cho câu hỏi "Cần bao nhiêu lần đọc?":**
- `read_discount = 10%` → break-even = **1.11 lần đọc**
- Thực tế: ai cũng hiểu nếu cache chỉ được đọc 1 lần thì không có lợi (phải bù chi phí write)
- Dataset đạt 2,400× → caching hoàn toàn justified

---

### Extension 4 — Reasoning Budget: Chi phí & Năng lượng cho is_reasoning

**File sửa:** `missions/m2_inference_levers.py`  
**Hàm mới:** `reasoning_budget_analysis()`

**Kết quả đo lường:**

| Chỉ số | Reasoning (8.4% traffic) | Normal (91.6% traffic) |
|---|---|---|
| Số requests/ngày | 201 | 2,199 |
| Chi phí/ngày | $2.82 (24.5% tổng) | $8.66 (75.5%) |
| Năng lượng/ngày | **29,787 Wh (94% tổng)** | 1,887 Wh (6%) |
| Avg tokens/request | 6,175 | 2,861 |

**Tại sao reasoning dùng ~80× năng lượng hơn?** Reasoning (chain-of-thought) tạo ra intermediate "thinking" tokens trước khi trả lời. Mỗi thinking token:
- Cần decode step riêng (memory-bound operation)
- Tích lũy KV cache lớn → memory bandwidth tăng bậc 2
- Thực tế là nhiều forward passes thay vì 1 → tổng compute ×80 cho cùng số output tokens

**Routing Cap Proposal:** Giới hạn reasoning ≤ 10% traffic:
- Hiện tại: 8.4% → đã dưới ngưỡng, không cần hành động ngay
- Rule đề xuất: `if small_model_confidence < 0.7 → escalate to reasoning`
- Monitoring critical: nếu traffic tăng 20%, reasoning có thể vượt 10%

---

## 5. Khuyến nghị 3 hành động đầu tiên cho NimbusAI

### Hành động 1 (ROI cao nhất, làm ngay): Chuyển tất cả interruptible jobs sang Spot
- **Impact:** $7,596 + $1,393 + $570 + $203 + $142 = **~$9,904/tháng** (training jobs)
- **Effort:** Thêm checkpoint logic, dùng `spot_checkpoint_cost()` để tính effective hours
- **Risk:** Low — H100 spot interrupt rate chỉ 2%, overhead checkpoint 3%

### Hành động 2 (Implement ngay, cost ~0): Bật Cascade + Batch API cho inference
- **Impact:** $48.87 → $8.48/ngày = **$1,212/tháng saved**
- **Effort:** Thêm `route_tier` logic để phân loại request, batch non-realtime requests
- **Risk:** Cần evaluation xem model nhỏ có đủ chất lượng không (A/B test 1 tuần)

### Hành động 3 (Foundation cho tương lai): Tag 100% resources → Enable Chargeback
- **Impact:** Tạo accountability — teams biết họ tiêu bao nhiêu → tự tối ưu
- **Hiện tại:** Tag coverage 92% → chargeback ready, nhưng cần đẩy lên 98%+
- **Action:** Enforce tagging policy: không có tag → block deployment

---

## Kết luận

Lab này dạy một bài học đơn giản nhưng có giá trị lớn: **đo đúng trước khi tối ưu**. GPU-Util 98% trông như ổn — nhưng MFU 19% nói lên sự thật. Tương tự, nhìn vào hóa đơn `$/GPU-hr` sẽ không thấy đội nào lãng phí — chỉ khi chuyển sang `$/1M-token` mới thấy hiệu quả thực sự.

**Tóm tắt số liệu:**
- Baseline → Optimized: **$27,133 → $14,626/tháng (-46%)**
- Inference $/1M-token: **$6.488 → $1.126 (-82.6%)**
- GPU-Util lie: `gpu-h100-4` báo 98.2% nhưng MFU chỉ 19.4%
- Reasoning: 8.4% traffic tiêu 94% tổng năng lượng

