# 📘 Đề Cương Chi Tiết & Lộ Trình Triển Khai Khóa Luận

**Đề tài (đề xuất):**
*"Hệ thống trợ lý thị giác đa tầng dựa trên mô hình ngôn ngữ thị giác (VLM) lượng tử hóa, vận hành CPU-only cục bộ trên laptop phổ thông (8GB RAM), hỗ trợ người khiếm thị nhận thức không gian trong nhà."*

**English title:** *A Multi-Layer CPU-Only Quantized Vision-Language Assistant for Indoor Scene Understanding for the Visually Impaired on Commodity Laptops.*

---

## Mục Lục
1. [Tổng quan & Động lực](#1-tổng-quan--động-lực)
2. [Bài toán & Câu hỏi nghiên cứu](#2-bài-toán--câu-hỏi-nghiên-cứu)
3. [Mục tiêu & Đóng góp](#3-mục-tiêu--đóng-góp)
4. [Khảo sát công trình liên quan](#4-khảo-sát-công-trình-liên-quan)
5. [Kiến trúc hệ thống đề xuất](#5-kiến-trúc-hệ-thống-đề-xuất)
6. [Lựa chọn mô hình & Công cụ](#6-lựa-chọn-mô-hình--công-cụ)
7. [Phương pháp đánh giá](#7-phương-pháp-đánh-giá)
8. [Lộ trình triển khai](#8-lộ-trình-triển-khai)
9. [Rủi ro & Phương án dự phòng](#9-rủi-ro--phương-án-dự-phòng)
10. [Đạo đức & Giới hạn nghiên cứu](#10-đạo-đức--giới-hạn-nghiên-cứu)
11. [Cấu trúc báo cáo khóa luận](#11-cấu-trúc-báo-cáo-khóa-luận)
12. [Tài liệu tham khảo trọng yếu](#12-tài-liệu-tham-khảo-trọng-yếu)

---

## 1. Tổng quan & Động lực

### 1.1. Bối cảnh

> ⚠️ **Lưu ý cho tác giả:** Các con số thống kê dưới đây là **ước tính tham khảo**. Bắt buộc verify lại trước khi đưa vào báo cáo chính thức bằng các nguồn dẫn ở cuối mỗi gạch đầu dòng.

- **Toàn cầu:** Theo WHO, ước tính **~2.2 tỷ người** trên thế giới có suy giảm thị lực gần hoặc xa, trong đó **ít nhất ~1 tỷ trường hợp** lẽ ra có thể phòng ngừa hoặc chưa được điều trị. `[CẦN VERIFY — WHO World Report on Vision]`
  - Nguồn gợi ý: [WHO Fact Sheet — Blindness and vision impairment](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment) · [WHO World Report on Vision (2019, có thể có bản update)](https://www.who.int/publications/i/item/9789241516570)
- **Việt Nam:** Theo Hội Người mù Việt Nam, ước tính **~2 triệu** người khiếm thị, trong đó **~600.000 người mù** hoàn toàn. `[CẦN VERIFY — số liệu thay đổi theo năm]`
  - Nguồn gợi ý: [Hội Người mù Việt Nam (hoinguoimuvietnam.vn)](http://hoinguoimuvietnam.vn/) · báo cáo của Bộ Y tế · Cục Quản lý Khám chữa bệnh.

Các giải pháp hỗ trợ hiện nay chia làm hai nhóm:

- **Phần cứng chuyên dụng:**
  - [**OrCam MyEye 3 Pro**](https://www.getorcam.com/buy-now/my-eye-3-pro): ~3.990–4.250 USD (giá niêm yết 2024). `[Verified]`
  - [**Envision Glasses**](https://www.letsenvision.com/glasses): ~1.700–3.500 USD tùy gói. `[CẦN VERIFY giá hiện hành]`
  - Hạn chế chung: giá cao, khó phổ cập tại Việt Nam.
- **Ứng dụng dựa trên đám mây:**
  - [**Be My AI** (Be My Eyes × OpenAI GPT-4V)](https://www.bemyeyes.com/blog/announcing-be-my-ai) — miễn phí nhưng phụ thuộc Internet.
  - [**Microsoft Seeing AI**](https://www.seeingai.com/) — miễn phí, đa nền tảng, gọi API cloud.
  - [**Google Lookout**](https://support.google.com/accessibility/android/answer/9031274) — Android, một số tính năng cần kết nối.
  - Hạn chế chung: phụ thuộc mạng, rủi ro về **quyền riêng tư**, **độ trễ**, và **giới hạn ngôn ngữ tiếng Việt**.

### 1.2. Khoảng trống nghiên cứu (Research Gap)
- Các nghiên cứu VLM cho edge (MobileVLM, TinyLLaVA, MiniCPM-V) **chủ yếu tập trung vào benchmark học thuật**, ít quan tâm đến *pipeline thực tế* khi camera bị mờ, người dùng cầm máy run, hoặc khi mô hình ảo giác (hallucinate).
- Các hệ thống thương mại **không công bố kiến trúc**, không thể nghiên cứu lại.
- **Tiếng Việt** gần như chưa có công trình mở về VLM hỗ trợ accessibility.

### 1.3. Bài toán cốt lõi
Hệ thống cần giải quyết đồng thời **bộ ba ràng buộc**:

| Ràng buộc | Mâu thuẫn |
|---|---|
| **Quyền riêng tư** | ⟷ Mô hình lớn cần đám mây |
| **Độ trễ thấp** | ⟷ VLM suy luận chậm trên CPU |
| **Độ chính xác ngữ nghĩa** | ⟷ Mô hình nhỏ thường ảo giác/yếu không gian |

---

## 2. Bài toán & Câu hỏi nghiên cứu

### 2.1. Câu hỏi nghiên cứu (Research Questions)
- **RQ1:** Có thể vận hành VLM ≤3B tham số trên laptop phổ thông (VRAM ≤6GB) với độ trễ end-to-end < 3 giây mà vẫn giữ chất lượng mô tả cảnh trong nhà ở mức chấp nhận được không?
- **RQ2:** Có thể giảm tỷ lệ ảo giác (hallucination) và sai sót về quan hệ không gian của VLM nhỏ bằng cách kết hợp tín hiệu phụ trợ (object detector, depth) thay vì fine-tune mô hình hay không?
- **RQ3:** Một cơ chế phản hồi dựa trên độ bất định (uncertainty-aware feedback) có cải thiện đáng kể tỷ lệ thành công tác vụ của người dùng so với việc gọi VLM trực tiếp một lần hay không?

### 2.2. Phạm vi (Scope)
- **Trong phạm vi:** Cảnh **trong nhà** (phòng khách, bếp, phòng ngủ, văn phòng); các đối tượng gia dụng phổ biến; tương tác bằng giọng nói tiếng Việt.
- **Ngoài phạm vi:** Điều hướng ngoài trời, nhận diện khuôn mặt, đọc văn bản dài (OCR), thay thế gậy/chó dẫn đường.

---

## 3. Mục tiêu & Đóng góp

### 3.1. Mục tiêu cụ thể (SMART)

> 📌 **Đã hiệu chỉnh** theo phần cứng thực tế: laptop **8GB RAM single channel DDR4-3200**, CPU Intel Gen 10+, iGPU Intel Iris Xe (không CUDA).

| # | Mục tiêu | Chỉ số thành công |
|---|---|---|
| O1 | Triển khai VLM lượng tử hóa chạy local trên **8GB RAM, CPU-only** | Mô hình nạp & suy luận thành công, peak RAM ≤6GB |
| O2 | Đạt **first-audio-token latency** (user nghe từ đầu tiên) **<5s** ở 90% truy vấn nhờ streaming TTS | Đo trên 100 truy vấn mẫu |
| O2b | Đạt **full response latency** <12s cho câu trả lời ngắn (≤30 tokens) | Đo trên 100 truy vấn mẫu |
| O3 | Giảm ≥30% truy vấn vô ích nhờ lớp Orchestrator | So sánh với baseline gọi VLM trực tiếp |
| O4 | Đạt VQA accuracy ≥ baseline VLM thô (gọi 1 lần) trên benchmark công khai | VizWiz-VQA + VQAv2 subset indoor |
| O5 | Giảm tỉ lệ ảo giác đối tượng (POPE F1) ≥ 5 điểm so với baseline khi bật Confidence Estimator | Đo trên POPE benchmark |
| O6 | Bộ demo định tính 30-50 ảnh cảnh phòng cho thấy hệ thống xử lý tốt cả thành công & thất bại | Case study trong Chương 6 báo cáo |

### 3.2. Đóng góp chính (Contributions)
1. **[Kiến trúc]** Đề xuất **kiến trúc 3 lớp (Filter–Verify–Compose)** tách rời tiền xử lý, kiểm tra bất định, và hậu xử lý ưu tiên cho VLM trên edge — *re-usable design* có thể áp cho các bài toán accessibility khác.
2. **[Phương pháp ★ điểm nhấn]** Cơ chế **đánh giá độ tin cậy không cần fine-tune** dựa trên kết hợp log-probability, self-consistency và phát hiện cụm từ mơ hồ — chứng minh tương quan với độ đúng trên POPE và VQAv2.
3. **[Lai ghép]** Module **Spatial Grounding lai** kết hợp output VLM với object detector nhằm giảm lỗi quan hệ không gian — một điểm yếu nổi tiếng của VLM nhỏ.
4. **[Kỹ thuật]** Chứng minh tính khả thi của **VLM pipeline accessibility chạy CPU-only trên 8GB RAM** — báo cáo chi tiết latency, RAM peak, ablation, đi kèm bộ **demo định tính** trên ảnh phòng thật.

---

## 4. Khảo sát công trình liên quan

> 📝 **Quy ước:** mỗi mục có (a) link arXiv hoặc paper gốc, (b) link Hugging Face nếu là mô hình mở. Tất cả arXiv ID nên được verify lại tại [arxiv.org](https://arxiv.org/) trước khi đưa vào bibliography chính thức.

### 4.1. VLM nén và chạy trên thiết bị biên
| Công trình | Năm | Link | Đóng góp chính | Khoảng trống |
|---|---|---|---|---|
| **LLaVA-1.5 / 1.6** | 2023-24 | [arXiv:2310.03744](https://arxiv.org/abs/2310.03744) · [HF](https://huggingface.co/liuhaotian) | Kiến trúc VLM mở phổ biến, instruction tuning | Quá lớn cho edge (7B+), tiếng Việt yếu |
| **MobileVLM V2** | 2024 | [arXiv:2402.03766](https://arxiv.org/abs/2402.03766) | VLM tối ưu cho mobile (1.4–3B) | Thiếu pipeline ứng dụng thực tế |
| **MiniCPM-V 2.6** | 2024 | [arXiv:2408.01800](https://arxiv.org/abs/2408.01800) · [HF](https://huggingface.co/openbmb/MiniCPM-V-2_6) | 8B-class hiệu năng GPT-4V level | Vẫn nặng với VRAM 4GB |
| **Moondream 2** | 2024 | [HF — vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) · [GitHub](https://github.com/vikhyat/moondream) | ~1.9B chạy được CPU, latency thấp | Yếu suy luận không gian, không hỗ trợ tiếng Việt |
| **Qwen2-VL-2B** | 2024 | [arXiv:2409.12191](https://arxiv.org/abs/2409.12191) · [HF](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Đa ngữ, dynamic resolution, hỗ trợ tiếng Việt cơ bản | Chưa có pipeline accessibility; `[CẦN VERIFY license: Tongyi Qianwen hay Apache-2.0]` |
| **Phi-3.5-Vision** | 2024 | [HF — microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | 4.2B, mạnh document/chart understanding | Yếu mô tả tự nhiên cảnh sống, tiếng Việt hạn chế |
| **TinyLLaVA** | 2024 | [arXiv:2402.14289](https://arxiv.org/abs/2402.14289) | Recipe huấn luyện VLM nhỏ (≤3B) | Chỉ tập trung benchmark học thuật |

### 4.2. Hệ thống hỗ trợ người khiếm thị (Sản phẩm & Nghiên cứu)
| Hệ thống | Loại | Link | Hạn chế |
|---|---|---|---|
| **Microsoft Seeing AI** | App, Cloud | [seeingai.com](https://www.seeingai.com/) | Phụ thuộc mạng, tiếng Việt không đầy đủ |
| **Be My AI** (Be My Eyes + GPT-4V) | App, Cloud | [bemyeyes.com](https://www.bemyeyes.com/blog/announcing-be-my-ai) | Phụ thuộc GPT-4V cloud, quyền riêng tư |
| **Google Lookout** | App Android | [Google Support](https://support.google.com/accessibility/android/answer/9031274) | Một số tính năng cần kết nối |
| **Envision AI / Glasses** | App + thiết bị | [letsenvision.com](https://www.letsenvision.com/) | Giá ~1.700–3.500 USD |
| **OrCam MyEye 3 Pro** | Phần cứng đóng | [getorcam.com](https://www.getorcam.com/buy-now/my-eye-3-pro) | Giá ~3.990–4.250 USD |
| **VizWiz Grand Challenge** (research) | Dataset + benchmark | [vizwiz.org](https://vizwiz.org/) | Tập trung VQA, không có pipeline ứng dụng |

### 4.3. Active Perception, Spatial Reasoning & Uncertainty trong VLM
- **Bajcsy (1988)** — *Active Perception*. [IEEE Xplore](https://ieeexplore.ieee.org/document/5968) — định nghĩa gốc của Active Vision.
- **SpatialVLM** (Chen et al., CVPR 2024) — [arXiv:2401.12168](https://arxiv.org/abs/2401.12168) — chỉ rõ VLM yếu spatial reasoning, đề xuất tăng cường bằng depth.
- **SpatialBot** (Cai et al., 2024) — [arXiv:2406.13642](https://arxiv.org/abs/2406.13642) — VLM kết hợp depth map cho spatial understanding.
- **POPE** (Li et al., EMNLP 2023) — [arXiv:2305.10355](https://arxiv.org/abs/2305.10355) — benchmark đo object hallucination trong VLM.
- **Self-Consistency** (Wang et al., ICLR 2023) — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — giảm sai số bằng sample đa lần.
- **HallusionBench** (Guan et al., CVPR 2024) — [arXiv:2310.14566](https://arxiv.org/abs/2310.14566) — phân tích sâu hallucination & visual illusion ở VLM.

### 4.4. Mô hình phụ trợ
- **YOLOv8** — [Ultralytics docs](https://docs.ultralytics.com/) · [GitHub](https://github.com/ultralytics/ultralytics).
- **Depth Anything V2** (Yang et al., 2024) — [arXiv:2406.09414](https://arxiv.org/abs/2406.09414) · [HF — Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small).
- **Vosk Speech Recognition** — [alphacephei.com/vosk](https://alphacephei.com/vosk/models) (có model `vosk-model-small-vn-0.4`).
- **Whisper** (Radford et al., OpenAI 2022) — [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
- **Piper TTS** — [github.com/rhasspy/piper](https://github.com/rhasspy/piper) (giọng `vi_VN` chính thức).

> **Định vị đề tài:** Đề tài lấp khoảng trống giữa (a) nghiên cứu VLM-on-edge thuần academic và (b) hệ thống accessibility thương mại đóng — bằng cách đưa ra một **pipeline mở, chạy local, có cơ chế tự kiểm tra độ tin cậy**, đánh giá trên **benchmark công khai** và **bộ demo định tính ảnh phòng thật tiếng Việt**. Đề tài định vị là **research prototype kỹ thuật**; phần đánh giá với người dùng khiếm thị thật được đề xuất là **Future Work**.

---

## 5. Kiến trúc hệ thống đề xuất

Hệ thống gồm **3 lớp xử lý** + các module phụ trợ:

```
┌────────────────────────────────────────────────────────────────────┐
│           [User Voice] → STT → Query                               │
│                              │                                     │
│           [Camera Stream] ───┼──► Layer 1: INPUT FILTER            │
│                              │      • Blur check (Laplacian var)   │
│                              │      • Scene-change check (SSIM)    │
│                              │      • Frame quality scoring        │
│                              ▼                                     │
│                       Layer 2: VLM CORE + VERIFIER                 │
│                       ┌────────────────────────────┐               │
│                       │  Quantized VLM (Qwen2-VL)  │               │
│                       │  + Aux: YOLOv8n + DepthV2  │               │
│                       │  ───────────────────────── │               │
│                       │  Confidence Estimator:     │               │
│                       │   • token logprob avg      │               │
│                       │   • self-consistency       │               │
│                       │   • vague-phrase detector  │               │
│                       └────────────┬───────────────┘               │
│                                    │                               │
│              Low confidence?  ─────┼──► Re-acquisition feedback    │
│                                    ▼     ("nghiêng máy sang phải") │
│                       Layer 3: OUTPUT COMPOSER                     │
│                       • Priority filter (Safety > Goal > Context)  │
│                       • Spatial grounding (merge VLM + YOLO+Depth) │
│                       • Vietnamese natural-language formatter      │
│                              │                                     │
│                              ▼                                     │
│                            TTS → [User Audio]                      │
└────────────────────────────────────────────────────────────────────┘
```

### 5.1. Lớp 1 — Adaptive Input Filter
**Mục tiêu:** Loại bỏ khung hình kém chất lượng & tránh suy luận trùng lặp.

| Thuật toán | Công thức / Phương pháp | Ngưỡng dự kiến |
|---|---|---|
| Phát hiện mờ | $\text{Var}(L) = E[L^2] - (E[L])^2$, với $L$ = Laplacian ảnh xám | $\tau_{\text{blur}} \in [80, 150]$, hiệu chỉnh thực nghiệm |
| Phát hiện cảnh tĩnh | SSIM giữa khung $t$ và $t-1$ | $\text{SSIM} > 0.95$ → coi là cảnh không đổi |
| Chất lượng phơi sáng | Histogram analysis (tránh ảnh quá tối/quá sáng) | Mean luminance ∈ [40, 220] |

**Hành vi:**
- Ảnh mờ → TTS gợi ý: *"Ảnh bị mờ, vui lòng giữ máy ổn định."*
- Ảnh tối → *"Cảnh quá tối, hãy quay về phía sáng hơn."*
- Cảnh không đổi & không có query mới → bỏ qua suy luận, tiết kiệm tài nguyên.

### 5.2. Lớp 2 — VLM Core + Confidence Verifier
**Mục tiêu:** Suy luận chính & tự đánh giá độ tin cậy.

#### 5.2.1. Suy luận chính
- VLM nhận: `(image, system_prompt, user_query)`.
- System prompt được thiết kế để output **JSON có cấu trúc**:
  ```json
  {
    "scene": "phòng bếp",
    "objects": [{"name":"dao","position":"trước mặt","distance":"gần"}],
    "hazards": ["dao sắc"],
    "answer": "..."
  }
  ```

#### 5.2.2. Đo độ tin cậy (Confidence Estimator) — *Đóng góp chính*
Hợp ba tín hiệu:

| Tín hiệu | Cách tính | Trọng số |
|---|---|---|
| **Log-probability trung bình** của token output | $\bar{p} = \frac{1}{N}\sum \log P(t_i)$ | $w_1 = 0.4$ |
| **Self-consistency** | Sample $k=3$ lần với temperature=0.5; đo tương đồng bằng BERTScore | $w_2 = 0.4$ |
| **Vague-phrase detector** | Đếm tần suất cụm mơ hồ ("có thể", "dường như", "không rõ") | $w_3 = 0.2$ |

$$C = w_1 \cdot \tilde{p} + w_2 \cdot \text{sim}_{\text{avg}} - w_3 \cdot \text{vague\_rate}$$

Nếu $C < \tau_C$ → kích hoạt **Re-acquisition Feedback**.

#### 5.2.3. Re-acquisition Feedback (đổi tên từ "Active Vision")
Khi tin cậy thấp **và** VLM phát hiện hint về vị trí ("góc khuất", "ngoài khung"), hệ thống đưa hướng dẫn cụ thể: *"Vật thể có vẻ bị khuất ở bên phải khung hình, vui lòng nghiêng máy nhẹ sang phải"*.

> ⚠️ **Lưu ý thuật ngữ:** Đây **không phải** Active Vision theo nghĩa Bajcsy (1988). Trong báo cáo dùng thuật ngữ chính xác là *"Confidence-Guided Re-acquisition"* hoặc *"Uncertainty-Aware Human-in-the-Loop"*.

### 5.3. Lớp 3 — Output Composer + Spatial Grounding
**Mục tiêu:** Biến raw output thành câu nói tiếng Việt có ưu tiên & chính xác về không gian.

#### 5.3.1. Spatial Grounding lai
Vì VLM nhỏ yếu spatial reasoning, dùng **YOLOv8n** làm tín hiệu phụ:
- YOLO cung cấp bounding box & class.
- Vị trí ngang trái/phải/giữa được tính từ tọa độ x-center của box.
- Vị trí trên/dưới được tính từ y-center.
- Hệ thống ánh xạ tên đối tượng từ VLM ↔ YOLO bằng so khớp ngữ nghĩa, **ghi đè** vị trí ngang/dọc nếu VLM mâu thuẫn với YOLO.

> ⚠️ **Đã loại bỏ Depth Anything V2 khỏi Phase 1** vì ràng buộc RAM 8GB. Việc ước lượng "gần/xa" sẽ dựa trên **kích thước bounding box tương đối** (heuristic) thay vì depth thật. **Depth thật được đẩy sang Future Work** (Chương 7) — nếu phần cứng nâng cấp hoặc thử trên cloud.

#### 5.3.2. Bộ lọc ưu tiên (Priority Filter)
Thứ tự đọc:
1. **An toàn (Safety):** vật sắc nhọn, lửa, nước, cầu thang, vật cản đường đi.
2. **Mục tiêu (Goal):** trả lời trực tiếp câu hỏi người dùng.
3. **Ngữ cảnh (Context):** mô tả bối cảnh chung.

#### 5.3.3. Bộ tạo câu tiếng Việt
- Template-based + post-edit (không gọi thêm LLM để tránh tăng latency).
- Sử dụng thì hiện tại, câu ngắn ≤ 15 từ, ưu tiên động từ đầu câu.

---

## 6. Lựa chọn mô hình & Công cụ

> 🎯 **Chiến lược 2 Track** (do ràng buộc phần cứng 8GB RAM, không có GPU CUDA):
> - **Track A — Local (deliverable cuối):** Mô hình siêu nhẹ chạy 100% offline trên máy 8GB → đây là *sản phẩm khóa luận thực sự*.
> - **Track B — Cloud (reference baseline):** Mô hình lớn hơn (Qwen2-VL-7B, MiniCPM-V 2.6) chạy Colab/Kaggle T4 để **đối chứng học thuật** trong Chương 5 (so sánh hiệu năng với "thiên đường").
> 
> Cả hai track đều phục vụ báo cáo nhưng *demo nộp = Track A*. Cloud KHÔNG phải sản phẩm vì sẽ phá vỡ argument privacy & offline.

### 6.1. VLM Track A (Local — chạy thực trên 8GB RAM)
| Vai trò | Mô hình | Kích thước Q4 | Lý do |
|---|---|---|---|
| **Chính** | **[Moondream 2 (1.9B)](https://huggingface.co/vikhyatk/moondream2)** | ~1.6 GB | Thiết kế cho edge, latency thấp nhất phân khúc, GGUF sẵn, hỗ trợ tốt llama.cpp |
| **Dự phòng A** | **[SmolVLM-Instruct (2.2B)](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)** | ~1.4 GB | Cực nhẹ, mới (HF 2024), pipeline đơn giản |
| **Dự phòng B** | **[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)** Q4_K_S | ~1.5 GB | Đa ngữ tốt hơn, nhưng dynamic resolution → chậm hơn Moondream trên CPU |
| **Loại bỏ** | LLaVA-7B, MiniCPM-V 2.6 8B, Phi-3.5-Vision 4.2B | — | Vượt budget RAM 8GB |

> **Quyết định cuối** sau khi benchmark thực ở Tuần 3-4 (theo timeline §8).

### 6.2. VLM Track B (Cloud — chỉ để so sánh trong báo cáo)
| Mô hình | Mục đích |
|---|---|
| **Qwen2-VL-7B-Instruct** | Reference upper bound cho captioning + VQA tiếng Việt |
| **MiniCPM-V 2.6 (8B)** | Reference upper bound cho mô tả ảnh chi tiết |
| **GPT-4o / Gemini 1.5 Flash** (free tier) | So sánh với hệ thống cloud thương mại |

Chạy trên: **Google Colab Free (T4 15GB) / Kaggle (P100 16GB)**.

### 6.3. Mô hình phụ trợ Phase 1
| Vai trò | Mô hình | Kích thước | Lý do |
|---|---|---|---|
| Object detection | **[YOLOv8n](https://docs.ultralytics.com/)** (3.2M params) | ~12 MB | Cực nhẹ, có pretrained COCO 80 class, CPU-friendly |
| STT tiếng Việt | **[Vosk vosk-model-small-vn-0.4](https://alphacephei.com/vosk/models)** | ~42 MB | Offline thực sự, nhanh trên CPU |
| TTS tiếng Việt | **[Piper TTS (vi_VN)](https://github.com/rhasspy/piper)** | ~60 MB | Offline, latency ~0.3s, hỗ trợ streaming |

### 6.4. Mô hình bị defer sang Future Work
| Mô hình | Lý do defer | Khi nào dùng được |
|---|---|---|
| **Depth Anything V2 Small** (24M) | Tổng RAM với 3 model concurrent vượt budget; latency tăng | Khi nâng RAM lên 16GB hoặc test trên cloud |
| **Whisper-small/medium** | Lớn hơn Vosk đáng kể; chỉ cần khi Vosk fail | Dùng Whisper-tiny (~75MB) nếu phải fallback |

### 6.5. Ngăn xếp phần mềm
- **Inference VLM:** `llama-cpp-python` (CPU build với AVX2/AVX-512, hoặc thử **Vulkan backend** để tận dụng Iris Xe).
- **Alternative inference:** `Intel OpenVINO` cho ViT image encoder của Moondream (có thể tăng 1.5-2× tốc độ encoding).
- **CV:** OpenCV, Ultralytics (YOLO).
- **Audio:** `sounddevice` cho I/O; `piper-tts` cho TTS (hỗ trợ streaming token-by-sentence).
- **Orchestration:** Python `asyncio` để chồng lấp capture–inference–TTS.
- **Logging & Eval:** TensorBoard local + Jupyter notebook + log CSV.

### 6.6. Cấu hình phần cứng thực tế (máy hiện tại)
- **CPU:** Intel Core i5/i7 Gen 10+ (AVX2/AVX-512). `[CẦN VERIFY model cụ thể]`
- **RAM:** **8 GB DDR4-3200 single channel** ⚠️ — nút thắt chính của hệ thống.
- **GPU:** Intel Iris Xe Graphics (iGPU, không CUDA, dùng chung RAM).
- **OS:** Windows 10/11.

**Khuyến nghị mạnh nếu có ngân sách:**
1. **Nâng dual channel** (mua thêm 1 thanh 8GB DDR4-3200, ~25-30 USD) → bandwidth tăng ~2× → inference nhanh ~1.7-1.9×. Đây là upgrade rẻ nhất, hiệu quả nhất.
2. **Nâng lên 16GB tổng** → cho phép load Qwen2-VL-2B + YOLO + Depth Anything cùng lúc.

---

## 7. Phương pháp đánh giá

### 7.1. Bộ dữ liệu
| Bộ dữ liệu | Vai trò | Số lượng |
|---|---|---|
| **VQAv2** (subset indoor) | Benchmark VQA chuẩn | ~1.000 cặp Q-A |
| **VizWiz-VQA** | Ảnh do người khiếm thị chụp | ~500 cặp |
| **COCO Captions** (indoor subset) | Đánh giá captioning | ~500 ảnh |
| **POPE** | Đo tỷ lệ ảo giác đối tượng | Toàn bộ |

> Đánh giá dùng hoàn toàn **benchmark công khai cho định lượng** + **demo định tính trên ảnh phòng thật** cho định tính (§7.3). Phần đánh giá người dùng thật được khoanh là **Future Work** để giữ phạm vi khóa luận khả thi cho một sinh viên.

### 7.2. Metric tự động
| Loại tác vụ | Metric | Ghi chú |
|---|---|---|
| Captioning | **BLEU-4, METEOR, CIDEr, BERTScore** | Có cả phiên bản tiếng Anh & Việt (BERTScore với PhoBERT) |
| VQA | **VQA Accuracy (soft)**, **Exact Match** | Trên VQAv2/VizWiz |
| Ảo giác | **POPE F1 / Accuracy** | Đo trực tiếp object hallucination |
| Spatial reasoning | **Relation accuracy** trên subset POPE-spatial + 50 ảnh user-study | left/right/front/behind/on/under |
| Hệ thống | **First-audio latency** (p50, p90); **Full-response latency** (p50, p90, p99); **RAM peak**; **CPU usage** | Đo trên 100 truy vấn |

### 7.3. Phân tích định tính qua bộ Demo (Qualitative Demo Analysis)

> Thay cho user study (định vị là Future Work), Chương 6 báo cáo sẽ phân tích sâu **bộ demo tự chụp** để minh họa khả năng & giới hạn thực tế của hệ thống.

**Quy trình:**
1. Tự chụp **30-50 ảnh** ở 5 loại phòng (khách / bếp / ngủ / làm việc / vệ sinh) tại nhà bằng webcam laptop.
2. Với mỗi ảnh, chạy cả 4 cấu hình ablation và lưu output JSON + lời mô tả tiếng Việt.
3. Phân loại kết quả thành 3 nhóm:
   - **Thành công (Hit):** Mô tả đúng vật + vị trí, không ảo giác.
   - **Thất bại nhẹ (Near-miss):** Đúng vật nhưng sai vị trí, hoặc bỏ sót vật phụ.
   - **Thất bại nghiêm trọng (Critical):** Ảo giác vật không có, bỏ sót hazard, hoặc cảnh báo sai.
4. Báo cáo bảng tần suất + ≥6 ví dụ minh họa (cả thành công lẫn thất bại) kèm ảnh, prompt, output, latency, confidence score.

**Tiêu chí đo của bộ demo:**
- Hit-rate trên 30-50 ảnh tự chụp.
- Critical-failure rate (kỳ vọng <10%).
- Trung bình confidence khi đúng vs khi sai (chứng minh tương quan).
- Latency thực tế khi chạy end-to-end (so với benchmark ở §7.2).

**Lưu ý phạm vi:** đây *không phải* user study chính thức và **không có** chỉ số SUS/TSR/NASA-TLX. Đề tài đề xuất rõ user study là **Future Work** cần phối hợp với Hội Người mù trong giai đoạn sau khóa luận.

### 7.4. Ablation Study
So sánh 4 cấu hình:
1. **Baseline:** Chỉ VLM, gọi 1 lần.
2. **+L1:** Thêm Input Filter.
3. **+L1+L2:** Thêm Confidence + Re-acquisition.
4. **Full (L1+L2+L3):** Bổ sung Output Composer + Spatial Grounding.

Mỗi cấu hình đo đầy đủ metric tự động ở §7.2 và được phân tích định tính trên cùng bộ ảnh demo ở §7.3.

---

## 8. Lộ trình triển khai

Tổng thời gian dự kiến: **16 tuần**.

| Tuần | Giai đoạn | Đầu ra | Track |
|---|---|---|---|
| **1-2** | Thiết lập môi trường (Win + Python + llama.cpp CPU build); dựng repo, viết `VLM_Engine` wrapper | `VLM_Engine` chạy được Moondream 2 GGUF local | A |
| **3-4** | **Benchmark 3 VLM ứng viên trên 8GB RAM** (Moondream 2 / SmolVLM / Qwen2-VL-2B Q4_K_S); chọn chính thức | Báo cáo: latency p50/p90, RAM peak, BLEU sơ bộ | A |
| **3-4** | Song song: setup Colab/Kaggle notebooks cho Track B (Qwen2-VL-7B, MiniCPM-V 2.6) | Notebook chạy được baseline lớn | B |
| **5-6** | Cài Lớp 1 (blur, SSIM, exposure) + tích hợp camera realtime | Demo realtime phát hiện khung kém | A |
| **7-8** | Cài Lớp 2 (Confidence Estimator + Re-acquisition); thử nghiệm Vulkan backend cho Iris Xe | Báo cáo correlation confidence ↔ độ đúng | A |
| **9** | Tích hợp **YOLOv8n** → Spatial Grounding (chỉ ngang/dọc, không depth) | Demo trả lời câu hỏi không gian tốt hơn baseline | A |
| **10** | Tích hợp **lớp dịch máy** (Argos/NLLB nhỏ) nếu chọn Moondream làm primary | Output tiếng Việt ổn định | A |
| **11** | Cài Lớp 3 (Priority Filter + Vietnamese formatter + streaming TTS Piper) | Demo end-to-end voice-in/voice-out, first-audio <5s | A |
| **12** | Hiệu chỉnh threshold cuối (blur, confidence, SSIM) + tinh chỉnh prompt | Báo cáo calibration | A |
| **13-14** | Chạy đánh giá tự động + **ablation 4 cấu hình** + so sánh Track A vs Track B | Bảng số liệu đầy đủ | A+B |
| **15** | Chụp 30-50 ảnh phòng + chạy cả 4 ablation + phân tích case study (hit / near-miss / critical) | Bộ ảnh + bảng phân loại + ≥6 ví dụ minh họa | A |
| **16** | Viết báo cáo + chuẩn bị bảo vệ | Bản thảo khóa luận | — |

---

## 9. Rủi ro & Phương án dự phòng

| # | Rủi ro | Khả năng | Tác động | Phương án |
|---|---|---|---|---|
| R1 | Moondream 2 chạy quá chậm trên CPU single-channel (< 2 tok/s) | **Cao** | **Cao** | Streaming TTS để mask latency; giảm n_predict; thử SmolVLM; cuối cùng dùng llama.cpp Vulkan backend trên Iris Xe |
| R2 | Tiếng Việt của VLM kém, output toàn tiếng Anh | **Rất cao** (Moondream chỉ tiếng Anh) | Cao | **Thêm lớp dịch máy nhẹ** (NLLB-200-distilled 600M Q4 hoặc Argos Translate offline) sau VLM; hoặc thay bằng Qwen2-VL-2B |
| R3 | Confidence Estimator không tương quan với độ đúng | Thấp | Cao | Thay bằng học có giám sát nhẹ (logistic regression trên features) |
| R4 | VLM ảo giác nguy hiểm (báo có lối đi trong khi có cầu thang) | Trung bình | **Rất cao** | Safety-keyword sanity check; luôn thêm disclaimer TTS; fail-safe: nếu confidence quá thấp → chỉ đọc danh sách object thô từ YOLO |
| R5 | Bộ demo 30-50 ảnh tự chụp không đủ đại diện (chỉ 1 căn nhà) | Cao | Trung bình | Chụp đa dạng góc, ánh sáng, thời điểm; mượn cảnh ở 2-3 nhà khác nhau; thừa nhận limitation trong báo cáo |
| R6 | Latency vượt mục tiêu ngay cả sau tối ưu | Trung bình | Trung bình | Streaming output; cache scene representation; giảm độ phân giải ảnh đầu vào |
| R7 | Pháp lý ảnh demo (ảnh có người, không gian cá nhân) | Thấp | Trung bình | Chỉ chụp môi trường tự sở hữu; làm mờ mặt nếu có |
| **R8** | **Hết RAM do leak khi chạy nhiều model concurrent (VLM + YOLO + STT + TTS)** | **Cao** | **Cao** | Lazy-load model; unload khi idle; giới hạn n_ctx; monitor `psutil` realtime |
| **R9** | **CPU thermal throttling sau vài phút inference liên tục → latency tăng vọt** | Trung bình | Trung bình | Thêm cooling pad khi demo; design pipeline có khoảng nghỉ; giảm `n_threads` xuống dưới max |
| **R10** | **Cloud track (Track B) bị giới hạn quota Colab/Kaggle** | Trung bình | Thấp | Phân lô đánh giá; chạy off-peak; backup Vast.ai trả phí (~$0.3/giờ) nếu cần |

---

## 10. Đạo đức & Giới hạn nghiên cứu

### 10.1. Vấn đề đạo đức
- **Quyền riêng tư:** toàn bộ inference chạy local, không gửi ảnh lên cloud (đây cũng là một động lực kỹ thuật của đề tài).
- **Ảnh demo:** chỉ chụp trong môi trường tự sở hữu; nếu xuất hiện người thân, sẽ làm mờ mặt trước khi đưa vào báo cáo.
- **Disclaimer bắt buộc** khi trình bày demo: *"Hệ thống chỉ là prototype nghiên cứu, không thay thế các phương tiện an toàn khác như gậy/chó dẫn đường."*

### 10.2. Giới hạn rõ ràng (cần ghi vào Chương 7)
- **Không có user study với người khiếm thị thật** — định vị là Future Work.
- Đánh giá định tính chỉ trên 30-50 ảnh tự chụp ở số ít môi trường → không đảm bảo tổng quát.
- Chỉ hoạt động **trong nhà**, ánh sáng đủ.
- Không hỗ trợ điều hướng động (di chuyển nhanh).
- Tiếng Việt phổ thông, chưa hỗ trợ phương ngữ vùng.
- Yêu cầu phần cứng tối thiểu (loại trừ điện thoại cấp thấp).

---

## 11. Cấu trúc báo cáo khóa luận

1. **Chương 1 — Mở đầu:** Bối cảnh, động lực, câu hỏi nghiên cứu, đóng góp, phạm vi.
2. **Chương 2 — Cơ sở lý thuyết:** VLM (kiến trúc, training), Lượng tử hóa (GGUF/AWQ/GPTQ), CV cơ bản (Laplacian, SSIM, YOLO, monocular depth), STT/TTS, các độ đo đánh giá.
3. **Chương 3 — Công trình liên quan:** Phân tích sâu các nhóm ở §4, định vị đề tài.
4. **Chương 4 — Thiết kế hệ thống:** Triển khai chi tiết 3 lớp, sơ đồ khối, mã giả thuật toán, công thức Confidence Estimator.
5. **Chương 5 — Cài đặt & Thực nghiệm:** Môi trường, dataset (VQAv2 / VizWiz / COCO / POPE), giao thức đánh giá, ablation.
6. **Chương 6 — Kết quả & Thảo luận:** Bảng metric, biểu đồ latency, ablation, phân tích định tính trên bộ demo (case study hit / near-miss / critical), so sánh baseline.
7. **Chương 7 — Kết luận & Hướng phát triển:** Trả lời RQ1-RQ3, giới hạn (đặc biệt: chưa có user study), đề xuất mở rộng — **ưu tiên hàng đầu là user study với người khiếm thị thật**, ngoài trời, kính AR, fine-tune Vietnamese VLM.

**Phụ lục:** Mã nguồn (link GitHub), bộ ảnh demo (lọc riêng tư), prompt template, hyperparameter.

---

## 12. Tài liệu tham khảo trọng yếu

> *Danh sách rút gọn; bản đầy đủ sẽ được biên tập theo IEEE/APA trong báo cáo chính thức. Mọi arXiv ID nên được verify lại tại [arxiv.org](https://arxiv.org) trước khi nộp.*

### Nền tảng & VLM
1. Bajcsy, R. (1988). *Active Perception*. **Proceedings of the IEEE**, 76(8), 966–1005. [IEEE Xplore](https://ieeexplore.ieee.org/document/5968).
2. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). *Visual Instruction Tuning (LLaVA)*. **NeurIPS 2023**. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485).
3. Liu, H. et al. (2024). *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)*. **CVPR 2024**. [arXiv:2310.03744](https://arxiv.org/abs/2310.03744).
4. Wang, P. et al. (2024). *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*. [arXiv:2409.12191](https://arxiv.org/abs/2409.12191).
5. Yao, Y. et al. (2024). *MiniCPM-V: A GPT-4V Level MLLM on Your Phone*. [arXiv:2408.01800](https://arxiv.org/abs/2408.01800).
6. Chu, X. et al. (2024). *MobileVLM V2: Faster and Stronger Baseline for Vision Language Model*. [arXiv:2402.03766](https://arxiv.org/abs/2402.03766).
7. Zhou, B. et al. (2024). *TinyLLaVA: A Framework of Small-scale Large Multimodal Models*. [arXiv:2402.14289](https://arxiv.org/abs/2402.14289).

### Spatial Reasoning, Uncertainty & Hallucination
8. Chen, B. et al. (2024). *SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities*. **CVPR 2024**. [arXiv:2401.12168](https://arxiv.org/abs/2401.12168).
9. Cai, W. et al. (2024). *SpatialBot: Precise Spatial Understanding with Vision Language Models*. [arXiv:2406.13642](https://arxiv.org/abs/2406.13642).
10. Li, Y. et al. (2023). *Evaluating Object Hallucination in Large Vision-Language Models (POPE)*. **EMNLP 2023**. [arXiv:2305.10355](https://arxiv.org/abs/2305.10355).
11. Guan, T. et al. (2024). *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion*. **CVPR 2024**. [arXiv:2310.14566](https://arxiv.org/abs/2310.14566).
12. Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. **ICLR 2023**. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).

### Mô hình phụ trợ (Detection, Depth, STT, TTS)
13. Jocher, G. et al. (2023). *Ultralytics YOLOv8*. [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).
14. Yang, L. et al. (2024). *Depth Anything V2*. [arXiv:2406.09414](https://arxiv.org/abs/2406.09414).
15. Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
16. Alpha Cephei. *Vosk Speech Recognition Toolkit*. [alphacephei.com/vosk](https://alphacephei.com/vosk/).
17. Hansen, M. *Piper: A fast, local neural text-to-speech system*. [github.com/rhasspy/piper](https://github.com/rhasspy/piper).

### Accessibility & Benchmark
18. Gurari, D. et al. (2018). *VizWiz Grand Challenge: Answering Visual Questions from Blind People*. **CVPR 2018**. [arXiv:1802.08218](https://arxiv.org/abs/1802.08218).

### Thống kê y tế (cần verify năm cập nhật)
19. World Health Organization. *World Report on Vision*. [WHO publications](https://www.who.int/publications/i/item/9789241516570). `[Verify version]`
20. WHO Fact Sheet. *Blindness and vision impairment*. [WHO Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment). `[Verify số liệu mới nhất]`
21. GBD 2019 Blindness and Vision Impairment Collaborators. *Causes of blindness and vision impairment in 2020*. **The Lancet Global Health**, 9(2), e144–e160, 2021. [DOI:10.1016/S2214-109X(20)30489-7](https://doi.org/10.1016/S2214-109X(20)30489-7).

### Quantization & Inference
22. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. **NeurIPS 2023**. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
23. ggml-org. *llama.cpp & GGUF format documentation*. [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).

---

> 📌 *File này là đề cương sống. Mỗi mục sẽ được cập nhật khi có kết quả benchmark thực tế hoặc thay đổi phạm vi sau buổi gặp giảng viên hướng dẫn.*
