# Lab 25 — Bonus Tracks Overview

Thư mục này chứa 3 bài thực hành nâng cao (Bonus Tracks) minh họa các khía cạnh thực tế của **GPU FinOps & LLM Cost Observability**:

---

## 📁 1. LiteLLM-style Token-Cost Tracker (`bonus/litellm_tracker/`)
- **Mục tiêu:** Xây dựng Proxy theo dõi chi phí $/request theo API Key của từng đội và thực thi **Hard-Stop Budget Cap** khi vượt ngân sách.
- **Thực thi:**
  ```bash
  cd bonus/litellm_tracker
  python demo.py
  ```
- **Kết quả chính:**
  - `team-chat` (dùng large model không tối ưu) bị **BLOCKED** sau 10 requests khi chạm trần `$0.05`.
  - `team-eval` (dùng small model + batch API + prompt caching) phục vụ 40 requests với chi phí `< $0.0001` (rẻ hơn **83×**).

---

## 📁 2. Real CPU Throughput & $/Token Benchmark (`bonus/local_model/`)
- **Mục tiêu:** Chạy mô hình ngôn ngữ thực tế (`sshleifer/tiny-gpt2`) trên CPU, đo tốc độ sinh token thực tế (tok/s) và so sánh `$/1M-token` thực tế với mô phỏng.
- **Thực thi:**
  ```bash
  source .venv/bin/activate
  cd bonus/local_model
  python run_local.py
  ```
- **Kết quả chính:**
  - Throughput thực tế đạt **~160 tok/s**.
  - Chi phí thực tế **~$0.17/1M-token** (so với simulation small-model là `$0.30/1M-token`).
  - **Bài học:** Thông lượng (tok/s) và mức độ sử dụng (utilization) quyết định chi phí `$/token` thực tế nhiều hơn so với giá niêm yết phần cứng theo giờ (`$/hr`).

---

## 📁 3. Prometheus & Grafana Cost Observability Dashboard (`bonus/docker/`)
- **Mục tiêu:** Xuất dữ liệu telemetry GPU thành các metrics chi phí FinOps theo chuẩn Prometheus (bao gồm `gpu_mfu`, `gpu_mbu`, `gpu_util_lie`, `gpu_wasted_cost_usd_per_hr`).
- **Thực thi (Pure Python Exporter):**
  ```bash
  cd bonus/docker
  python exporter.py
  # Kiểm tra metrics tại:
  curl http://localhost:9101/metrics
  ```
- **Thực thi với Docker Stack (Prometheus + Grafana):**
  ```bash
  cd bonus/docker
  docker compose up -d
  # Grafana dashboard tại: http://localhost:3000 (admin/admin)
  ```
- **Kết quả chính:**
  - Phát hiện chính xác `gpu_util_lie = 1` cho `gpu-h100-4` và `gpu-a10g-1`.
  - Định lượng chính xác chi phí lãng phí `gpu_wasted_cost_usd_per_hr = (1 - MFU) * $/hr`.

