# ĐỀ CƯƠNG CHI TIẾT LUẬN VĂN THẠC SĨ

> Trình bày theo *Hướng dẫn viết đề cương luận văn thạc sĩ* (định hướng nghiên cứu).
> Định dạng nộp: Times New Roman cỡ 13–14, giãn dòng 1.5; lề trên 3.5cm, dưới 3cm,
> trái 3.5cm, phải 2cm; số trang ở giữa phía dưới; bìa mềm màu xanh; ~30 trang.
> Các ô `<...>` cần học viên điền thông tin cá nhân.

---

**ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH — TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN**

**TÊN ĐỀ TÀI:**
**Nghiên cứu kỹ thuật đầu độc mô hình tàng hình và thích nghi đa ràng buộc trong Học Liên kết trên dữ liệu phi-IID**
*(A Multi-constraint Stealthy and Adaptive Model-Poisoning Technique for Federated Learning under Non-IID Data)*

- **Chuyên ngành:** Khoa học Máy tính / An toàn Thông tin
- **Học viên thực hiện:** `<Họ tên học viên – MSSV>`
- **Cán bộ hướng dẫn:** `<Học hàm, học vị, Họ tên CBHD>`
- **Thời gian thực hiện:** từ `<…>` đến `<…>`

---

## MỤC LỤC

1. Lời nói đầu
   - 1.1. Tính cấp thiết của đề tài
   - 1.2. Tình hình nghiên cứu
   - 1.3. Mục đích và nhiệm vụ nghiên cứu
   - 1.4. Đối tượng và phạm vi nghiên cứu
   - 1.5. Phương pháp nghiên cứu
   - 1.6. Kết cấu của luận văn
2. Kết cấu nội dung luận văn (chi tiết các chương)
3. Kế hoạch thực hiện
4. Tài liệu tham khảo

## DANH MỤC CÁC KÝ HIỆU VÀ CHỮ VIẾT TẮT

| Viết tắt | Thuật ngữ |
|---|---|
| FL | Federated Learning (Học Liên kết) |
| IID / phi-IID | Independent and Identically Distributed / Non-IID |
| FedAvg | Federated Averaging |
| AGR | Aggregation Rule (luật tổng hợp) |
| ASR | Attack Success Rate (tỉ lệ tấn công thành công) |
| MA | Main-task Accuracy (độ chính xác tác vụ chính) |
| DP | Differential Privacy (riêng tư vi phân) |
| SOTA | State-of-the-art |

---

# 1. LỜI NÓI ĐẦU

## 1.1. Tính cấp thiết của đề tài

Học Liên kết (Federated Learning – FL) đã trở thành mô hình tiêu chuẩn để huấn
luyện mô hình học máy phân tán trên hàng loạt thiết bị (điện thoại, IoT, bệnh
viện, ngân hàng) mà **không cần tập trung dữ liệu thô**, qua đó bảo vệ quyền
riêng tư và tuân thủ pháp lý (GDPR, HIPAA). Tuy nhiên, chính việc máy chủ **mất
khả năng kiểm duyệt dữ liệu và quá trình huấn luyện cục bộ** đã mở ra một bề mặt
tấn công mới: **đầu độc mô hình (model poisoning)**, trong đó một số máy khách
độc hại gửi lên những bản cập nhật được chế tạo có chủ đích nhằm cấy **cửa sập
(backdoor)** vào mô hình toàn cục — khiến mô hình vẫn hoạt động tốt trên tác vụ
chính nhưng phân loại sai theo ý kẻ tấn công đối với các đầu vào mang **tín hiệu
kích hoạt (trigger)**.

Tính cấp thiết của đề tài xuất phát từ **ba thực tế chưa được giải quyết thấu đáo**:

1. **Cuộc chạy đua tấn công–phòng thủ đang bế tắc theo chiều "đơn đặc trưng".**
   Các hệ thống phòng thủ Byzantine-robust hiện đại thường chỉ kiểm tra **một**
   đặc trưng của bản cập nhật: độ lớn (Krum, Trimmed-Mean), hướng gradient
   (FLTrust), hay phân cụm mật độ + nhiễu (FLAME). Nhiều kỹ thuật tấn công vượt
   được lớp này lại gãy trước lớp khác. Câu hỏi *liệu một kẻ tấn công có thể thỏa
   mãn đồng thời nhiều ràng buộc để vượt qua tổ hợp nhiều lớp phòng thủ hay không*
   vẫn còn bỏ ngỏ.

2. **Tính bất đồng nhất dữ liệu (phi-IID) — đặc trưng cố hữu của FL thực tế — có
   thể trở thành "đồng minh" của kẻ tấn công.** Khi dữ liệu giữa các máy khách
   lệch mạnh, phương sai tự nhiên của các bản cập nhật trung thực tăng cao, tạo
   "bức màn nhiễu" để cập nhật độc hại ẩn náu. Mức độ rủi ro an ninh do phi-IID
   gây ra **chưa được định lượng có hệ thống**.

3. **Độ bền (durability) của backdoor sau khi kẻ tấn công rời mạng** là yếu tố
   quyết định mức nguy hại thực tế, nhưng ít được khảo sát trong tương quan với
   tính tàng hình.

**Sự phù hợp với chuyên ngành:** đề tài thuộc lĩnh vực **an toàn thông tin cho
hệ thống học máy phân tán**, kết hợp tối ưu hóa có ràng buộc, học sâu và phân
tích an ninh — đúng định hướng nghiên cứu của chuyên ngành Khoa học Máy tính /
An toàn Thông tin.

**Câu hỏi nghiên cứu (Research Questions):**
- **RQ1.** Tồn tại quan hệ đánh đổi giữa **độ tàng hình** (khả năng lọt qua bộ
  lọc, *evasion*) và **hiệu quả tấn công** (*ASR*) của một kỹ thuật đầu độc đa
  ràng buộc như thế nào, dưới các họ phòng thủ khác nhau?
- **RQ2.** Mức độ bất đồng nhất phi-IID (tham số Dirichlet α) dịch chuyển ranh
  giới đánh đổi đó về phía kẻ tấn công ra sao?
- **RQ3.** Backdoor do kỹ thuật đề xuất cấy vào tồn tại được bao lâu sau khi kẻ
  tấn công ngừng tham gia?

## 1.2. Tình hình nghiên cứu

### a) Các kỹ thuật tấn công đầu độc

Tấn công trong FL chia thành **không nhắm mục tiêu** (làm suy giảm độ chính xác)
và **backdoor nhắm mục tiêu** (tinh vi, tàng hình). Baruch và cộng sự (LIE,
NeurIPS 2019) chỉ ra rằng *"một chút là đủ"*: nhiễu nhỏ μ−z·σ quanh phân phối
benign đủ để qua mặt các bộ lọc thống kê. Fang và cộng sự (USENIX 2020) và
Shejwalkar & Houmansadr (Min-Max/DnC, NDSS 2021) đề xuất **tấn công thích nghi
theo từng luật tổng hợp (AGR-tailored)** — chuẩn vàng để *đánh giá* phòng thủ.
Bagdasaryan và cộng sự (AISTATS 2020) đề xuất **Model Replacement** (nhân tỉ lệ
cập nhật để thay thế mô hình); Xie và cộng sự (DBA, ICLR 2020) phân tán trigger.
**Neurotoxin** (Zhang và cộng sự, ICML 2022) tăng độ bền backdoor bằng cách cấy
vào các tham số ít được cập nhật. **3DFed** (Li và cộng sự, IEEE S&P 2023) là
khung tấn công thích nghi black-box dùng chỉ báo phản hồi và mô hình mồi nhử.

### b) Các kỹ thuật phòng thủ

- **Thế hệ khoảng cách/thống kê:** Krum, Multi-Krum (Blanchard 2017), Median &
  Trimmed-Mean (Yin 2018), Bulyan (Mhamdi 2018).
- **Thế hệ định hướng:** FLTrust (Cao 2021) dùng tập "gốc" sạch để chấm điểm tin
  cậy theo cosine.
- **Thế hệ phân cụm + nhiễu/lịch sử:** FLAME (Nguyen, USENIX 2022) phân cụm
  HDBSCAN theo cosine + cắt chuẩn + nhiễu; FLDetector (Zhang, KDD 2022) phát hiện
  qua tính nhất quán lịch sử.

### c) Những vấn đề CHƯA được giải quyết thấu đáo (khoảng trống nghiên cứu)

1. Phần lớn tấn công được thiết kế để vượt **một** họ phòng thủ; **thiếu** một
   khung tối ưu hóa **đa ràng buộc** (đồng thời chuẩn + hướng + phân cụm) và
   đánh giá trên **tổ hợp** nhiều phòng thủ.
2. Vai trò của **phi-IID như một yếu tố an ninh** (không chỉ là thách thức hội
   tụ) **chưa được khảo sát định lượng** qua quét α một cách hệ thống.
3. Quan hệ **đánh đổi tàng hình↔hiệu quả** hiếm khi được *đặc trưng hóa tường
   minh*; các công trình thường chỉ báo cáo điểm "thắng" mà không vẽ ra **đường
   biên (frontier)**.

> *Lưu ý:* phần này là **tổng quan phân tích** (nêu cái đã/chưa giải quyết),
> không phải liệt kê tài liệu; các nguồn đều là công trình hội nghị/ tạp chí
> hàng đầu, không phải giáo trình.

## 1.3. Mục đích và nhiệm vụ nghiên cứu

**Mục đích:** đề xuất và đánh giá một **kỹ thuật đầu độc mô hình mới** trong FL
có khả năng **tàng hình đa ràng buộc, thích nghi với phòng thủ và bền vững**,
đồng thời **đặc trưng hóa** quan hệ đánh đổi và vai trò của phi-IID — qua đó
phơi bày lỗ hổng của các phòng thủ SOTA và làm cơ sở phát triển phòng thủ thế
hệ sau.

**Nhiệm vụ:**
1. Xây dựng nền tảng benchmark FL chuẩn (không gian cập nhật Δ, các phòng thủ
   Krum/Median/Bulyan/FLTrust/FLAME, các tấn công nền LIE/Min-Max/Model
   Replacement).
2. Thiết kế kỹ thuật **GeoTox**: kết hợp **(i)** tàng hình hướng (ép cosine với
   hướng benign ≥ τ), **(ii)** tàng hình độ lớn (chuẩn ≈ trung vị benign),
   **(iii)** cơ chế bền vững tùy chọn (mặt nạ Neurotoxin), và biến thể
   **GeoTox-Adaptive** (white-box, dò biên chấp nhận của phòng thủ).
3. Định nghĩa và đo các **chỉ số mới**: Evasion Rate, đường cong đánh đổi
   Evasion↔ASR, độ nhạy theo α, và độ bền (retention).
4. Thực nghiệm có hệ thống và **phân tích trung thực** (kể cả giới hạn).

## 1.4. Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu:** kỹ thuật **đầu độc mô hình (model poisoning)** dạng
backdoor trong hệ thống Học Liên kết, và tương tác của nó với các luật tổng hợp
Byzantine-robust dưới dữ liệu phi-IID.

**Phạm vi (phù hợp quy mô luận văn thạc sĩ):**
- *Mô hình đe dọa:* kẻ tấn công kiểm soát 10–30% máy khách, tham gia ngắt quãng;
  hai cấp tri thức: **thực tế** (chỉ ước lượng thống kê benign) và **white-box
  thích nghi** (biết luật phòng thủ — cận trên toàn tri).
- *Tập dữ liệu/mô hình:* MNIST và Fashion-MNIST với LeNet (chạy được trên CPU);
  CIFAR-10/ResNet là phần mở rộng nếu có GPU.
- *Phòng thủ khảo sát:* FedAvg (không phòng thủ), Krum, Median/Trimmed-Mean,
  Bulyan, **FLTrust**, **FLAME** (bản phụ thuộc-tối-thiểu, thay HDBSCAN bằng lựa
  chọn lõi-đa-số theo cosine — *nêu rõ là xấp xỉ*).
- *Phân bố dữ liệu:* IID và phi-IID Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0}.
- *Ngoài phạm vi:* secure aggregation mã hóa, FL trên mô hình nền/LLM (đề xuất là
  hướng phát triển).

## 1.5. Phương pháp nghiên cứu

- **Phương pháp định lượng – thực nghiệm có kiểm soát:** xây dựng khung mô phỏng
  FL, quét tham số (τ, α, tỉ lệ độc, seed), đo các chỉ số khách quan; so sánh có
  đối chứng (baseline không tấn công, baseline không phòng thủ). *Lý do chọn:* an
  ninh học máy đòi hỏi bằng chứng số liệu tái lập được, không thể kết luận bằng
  lập luận định tính.
- **Phương pháp mô hình hóa toán học:** phát biểu tấn công trong **không gian cập
  nhật** (u = w_local − w_global); ràng buộc tàng hình giải bằng phép chiếu/hòa
  trộn vector và tìm kiếm nhị phân (đạt cos ≥ τ; dò biên chấp nhận của AGR).
- **Phương pháp phân tích đối chứng (adversarial evaluation):** đánh giá tấn công
  trước **tổ hợp** phòng thủ, dùng tấn công thích nghi làm chuẩn nghiêm ngặt.
- **Nguồn số liệu:** bộ dữ liệu công khai MNIST, Fashion-MNIST (torchvision) —
  mức khả thi cao, hoàn toàn tái lập; mọi kết quả kèm seed và cấu hình.
- **Công cụ:** Python, PyTorch; mã nguồn tự phát triển (kho `FL-ModelPoisoning-
  benchmark`), kèm 25+ kiểm thử đơn vị và script phân tích/đồ thị tự động.

## 1.6. Kết cấu của luận văn

Luận văn dự kiến gồm **5 chương** (xem chi tiết Mục 2), bố cục bảo đảm có **01
chương cơ sở lý thuyết** (Chương 1) và **01 chương khảo sát thực trạng** (Chương
2), tiếp theo là đề xuất (Chương 3), thực nghiệm (Chương 4) và kết luận (Chương
5), kèm Danh mục tài liệu tham khảo.

---

# 2. KẾT CẤU NỘI DUNG LUẬN VĂN

## 2.1. Số chương và tên các chương

- **Chương 1. Tổng quan và cơ sở lý thuyết về Học Liên kết và đầu độc mô hình**
- **Chương 2. Khảo sát thực trạng: các kỹ thuật tấn công–phòng thủ và lỗ hổng còn tồn tại**
- **Chương 3. Kỹ thuật đề xuất: GeoTox và GeoTox-Adaptive**
- **Chương 4. Thực nghiệm và đánh giá**
- **Chương 5. Kết luận và hướng phát triển**

## 2.2. Các tiểu mục chi tiết (đến 3 chữ số)

### Chương 1. Tổng quan và cơ sở lý thuyết *(chương cơ sở lý luận)*
- 1.1. Học Liên kết và thuật toán FedAvg
  - 1.1.1. Kiến trúc client–server và vòng giao tiếp
  - 1.1.2. Tổng hợp có trọng số; không gian trọng số vs không gian cập nhật
  - 1.1.3. Dữ liệu IID và phi-IID; phân phối Dirichlet
- 1.2. Tấn công đầu độc trong FL
  - 1.2.1. Phân loại: data poisoning vs model poisoning; targeted vs untargeted
  - 1.2.2. Backdoor: trigger, nhãn mục tiêu, độ chính xác tác vụ chính
  - 1.2.3. Mô hình đe dọa và giả định tri thức của kẻ tấn công
- 1.3. Phòng thủ Byzantine-robust
  - 1.3.1. Theo khoảng cách/thống kê (Krum, Median, Trimmed-Mean, Bulyan)
  - 1.3.2. Theo định hướng (FLTrust)
  - 1.3.3. Theo phân cụm + nhiễu (FLAME)
- 1.4. Các độ đo đánh giá
  - 1.4.1. ASR và độ chính xác tác vụ chính
  - 1.4.2. Evasion Rate và quan hệ đánh đổi
  - 1.4.3. Độ bền (durability/retention)

### Chương 2. Khảo sát thực trạng *(chương thực trạng vấn đề nghiên cứu)*
- 2.1. Thực trạng tấn công SOTA và giới hạn
  - 2.1.1. LIE, Min-Max/DnC, Fang (AGR-tailored)
  - 2.1.2. Model Replacement, DBA, Neurotoxin (độ bền)
  - 2.1.3. 3DFed và Chameleon (trùng tên — phân biệt tại 2.4)
- 2.2. Thực trạng phòng thủ và điểm yếu theo "đơn đặc trưng"
- 2.3. Phi-IID: thách thức hội tụ hay rủi ro an ninh?
- 2.4. Định vị khác biệt và đóng góp của luận văn (bảng đối chiếu)
- 2.5. Phát biểu bài toán và các giả thuyết RQ1–RQ3

### Chương 3. Kỹ thuật đề xuất: GeoTox và GeoTox-Adaptive
- 3.1. Ý tưởng và tổng quan kiến trúc đa ràng buộc
- 3.2. Tấn công trong không gian cập nhật
- 3.3. Ràng buộc tàng hình hướng (hòa trộn để cos(Δ, μ_benign) ≥ τ)
  - 3.3.1. Tham số τ như "núm" đánh đổi tàng hình↔hiệu quả
  - 3.3.2. Giải bằng tìm kiếm nhị phân hệ số hòa trộn
- 3.4. Ràng buộc tàng hình độ lớn (chuẩn ≈ trung vị benign)
- 3.5. Cơ chế bền vững tùy chọn (mặt nạ kiểu Neurotoxin) và **đánh đổi với ASR**
- 3.6. GeoTox-Adaptive (white-box): dò biên chấp nhận của phòng thủ
- 3.7. Giả thuyết "phối hợp" (các client độc gửi cập nhật đồng nhất) và hệ quả
  với phòng thủ phân cụm
- 3.8. Phân tích độ phức tạp và tính khả thi

### Chương 4. Thực nghiệm và đánh giá
- 4.1. Thiết lập thực nghiệm (dữ liệu, mô hình, siêu tham số, hạ tầng, seed)
- 4.2. Giao thức đánh giá và các baseline (none / model_replacement)
- 4.3. RQ1 — Đường cong đánh đổi Evasion↔ASR theo từng phòng thủ
- 4.4. RQ2 — Độ nhạy theo mức phi-IID (α)
- 4.5. RQ3 — Độ bền backdoor sau khi kẻ tấn công rời mạng
- 4.6. Thảo luận, giới hạn và mối đe dọa tới tính hợp lệ (validity)

### Chương 5. Kết luận và hướng phát triển
- 5.1. Tóm tắt đóng góp
- 5.2. Hàm ý cho thiết kế phòng thủ thế hệ sau
- 5.3. Hạn chế
- 5.4. Hướng phát triển (đa seed/đa dataset, FLDetector, secure aggregation, FL-LoRA/LLM)

## 2.3. Bảng đối chiếu định vị (đặt tại mục 2.4 của luận văn)

| Năng lực | Model Repl. | LIE/Min-Max | Neurotoxin | 3DFed | **GeoTox (đề xuất)** |
|---|---|---|---|---|---|
| Vượt phòng thủ khoảng cách | Kém | TB | Tốt | Tốt | Có (tàng hình độ lớn) |
| Vượt phòng thủ định hướng | Kém | Kém | Kém | Tốt | Có (ép cosine τ) |
| Thích nghi theo AGR | Không | Một phần | Không | Có | **Có (Adaptive dò biên)** |
| Độ bền backdoor | Thấp | Thấp | **Rất cao** | TB | Có (mặt nạ tùy chọn) |
| Khảo sát hệ thống theo α | Không | Không | Không | Kém | **Có (đóng góp chính)** |
| Đặc trưng hóa trade-off | Không | Không | Không | Không | **Có** |

> *Phân biệt trùng tên:* "Chameleon" (Dai và cộng sự, ICML 2023) dùng học đối
> lập trong không gian đặc trưng — **khác hoàn toàn** kỹ thuật ở đây; luận văn
> dùng tên **GeoTox** để tránh nhầm lẫn.

---

# 3. KẾT QUẢ THỰC NGHIỆM SƠ BỘ (3 seed, mean±std)

Khung thực nghiệm đã được hiện thực hóa và kiểm thử (25+ unit test). Kết quả dưới
đây trên **MNIST, phi-IID α=0.5, 20% máy khách độc, 30 vòng, trung bình ±độ lệch
chuẩn trên 3 seed**.

**RQ1 — Đánh đổi Evasion↔ASR theo phòng thủ (ASR %, mean±std):**

| Phòng thủ | τ=0.0 | τ=0.3 | τ=0.6 | τ=0.9 | Evas@τ0.9 | Acc |
|---|---|---|---|---|---|---|
| FedAvg (không pt) | 37.5±37.1 | 21.3±24.8 | 2.5±2.3 | 0.3±0.0 | 100% | 98.7 |
| **Krum** | 2.4±1.9 | 2.3±2.0 | 44.2±35.7 | **97.1±1.3** | 47% | 97.6 |
| **FLTrust** | 0.3±0.1 | 0.4±0.1 | 0.4±0.1 | 0.3±0.1 | 64% | 97.5 |
| FLAME | 38.2±44.4 | 42.3±47.9 | 23.7±19.8 | 0.4±0.1 | 100% | 98.6 |

*Đọc kết quả:*
- **Krum — kết quả chủ lực, ỔN ĐỊNH:** GeoTox phải nâng τ để lọt qua bộ lọc
  (Evasion 4%→47%); tại τ=0.9 đạt **ASR 97.1±1.3%** (độ lệch chuẩn rất nhỏ),
  trong khi độ chính xác tác vụ chính vẫn 97.6%. Đây là minh chứng rõ ràng và lặp
  lại được cho đường cong đánh đổi (RQ1).
- **FLTrust — phòng thủ mạnh nhất (negative result sạch):** ASR giữ ở
  **0.3–0.4±0.1%** tại MỌI τ, ngay cả khi evasion tăng tới 64%. Lý do: để khớp
  hướng tham chiếu *sạch* của FLTrust thì cập nhật phải thật sự lành tính, qua đó
  triệt tiêu tín hiệu backdoor. **GeoTox không phá được FLTrust.**
- **FedAvg (không phòng thủ):** không cần tàng hình nên τ thấp hiệu quả hơn; ASR
  giảm khi τ tăng. Phương sai cao ở τ thấp do lấy mẫu client ngẫu nhiên + số vòng
  ngắn.
- **FLAME — chưa kết luận:** phương sai **rất lớn (±40–48)**: tùy seed, GeoTox
  *đôi khi* xuyên (ASR ~90%), *đôi khi* bị chặn (~1%); trung bình ~25–40% nhưng
  **không ổn định**. Nguyên nhân: chiến thuật "client độc gửi cập nhật đồng nhất"
  tạo cụm cosine dày có thể đánh lừa phân cụm của FLAME, nhưng hiệu quả phụ thuộc
  cách các cập nhật benign phân tán theo seed. Do đó luận văn **không** tuyên bố
  "đánh bại FLAME"; đây là một quan sát về **tính bất ổn định của phòng thủ phân
  cụm dưới tấn công phối hợp**, cần thêm seed/vòng để khẳng định.

**RQ2 — Độ nhạy theo mức phi-IID (FLAME, τ=0.5, ASR mean±std):**

| α (Dirichlet) | 0.1 | 0.3 | 0.5 | 1.0 | IID |
|---|---|---|---|---|---|
| ASR (%) | 54.7±40.6 | 31.6±30.4 | 28.7±22.5 | 6.9±9.6 | 16.8±21.6 |

ASR **giảm dần khi α tăng** (0.1→1.0), *ủng hộ định hướng* "dữ liệu càng bất đồng
nhất, kẻ tấn công càng dễ ẩn náu". Tuy nhiên phương sai lớn và điểm IID chưa khớp
hoàn toàn tính đơn điệu → đây là **bằng chứng định hướng (suggestive)**, cần tăng
số seed/vòng để kết luận chắc chắn.

**RQ3 — Độ bền sau khi kẻ tấn công rời mạng (vòng 15):** độ bền chỉ có ý nghĩa ở
những lần backdoor thực sự được cấy. Ở seed mà GeoTox-Adaptive xuyên được FLAME
(ASR@stop 92.7%), backdoor giữ **retention ~68%** (ASR 92.7%→63.3% sau khi rời);
các lần backdoor không cấy được thì retention không đại diện.

> *Tính trung thực (quan trọng):* một số kết quả (FLAME, FedAvg/τ thấp, RQ2) có
> **phương sai cao** do (i) mới 3 seed, (ii) 30 vòng, (iii) lấy mẫu client ngẫu
> nhiên khiến số client độc/vòng dao động. Hai kết quả **vững và lặp lại được** là
> *đường cong đánh đổi trên Krum* và *khả năng phòng thủ của FLTrust*. Luận văn
> đầy đủ sẽ tăng số vòng và số seed, bổ sung Fashion-MNIST, và trình bày những
> phòng thủ mà GeoTox **không** vượt qua như kết quả hợp lệ — đúng tinh thần đánh
> giá đối chứng nghiêm ngặt.

---

# 4. KẾT QUẢ MONG ĐỢI VÀ SẢN PHẨM DỰ KIẾN

- **Sản phẩm khoa học:** (i) kỹ thuật tấn công mới **GeoTox/GeoTox-Adaptive**;
  (ii) **đặc trưng hóa đường cong đánh đổi** Evasion↔ASR; (iii) **khảo sát hệ
  thống vai trò an ninh của phi-IID**; (iv) phân tích **độ bền**.
- **Sản phẩm kỹ thuật:** mã nguồn benchmark tái lập được (PyTorch) + bộ kiểm thử
  + script phân tích/đồ thị.
- **Kết quả mong đợi:** chứng minh được rằng tổ hợp phòng thủ SOTA vẫn có thể bị
  xuyên bởi tấn công đa ràng buộc thích nghi, và phi-IID làm trầm trọng rủi ro;
  từ đó đề xuất hàm ý cho phòng thủ thế hệ sau.

---

# 5. KẾ HOẠCH THỰC HIỆN

| Giai đoạn | Nội dung | Thời gian dự kiến |
|---|---|---|
| 1 | Tổng quan tài liệu, hoàn thiện Chương 1–2 | Tháng 1–2 |
| 2 | Hiện thực nền benchmark + phòng thủ (đã xong phần lớn) | Tháng 2–3 |
| 3 | Thiết kế & cài đặt GeoTox/Adaptive (Chương 3) | Tháng 3–4 |
| 4 | Thực nghiệm đa seed/đa dataset + phân tích (Chương 4) | Tháng 4–6 |
| 5 | Viết luận văn, chỉnh sửa theo CBHD, bảo vệ | Tháng 6–8 |

*(Học viên điều chỉnh mốc thời gian theo lịch đào tạo thực tế.)*

---

# 6. TÀI LIỆU THAM KHẢO (trích dẫn theo số, sắp xếp khi hoàn thiện)

[1] H. B. McMahan và cộng sự, "Communication-Efficient Learning of Deep Networks
from Decentralized Data," *AISTATS*, 2017.
[2] P. Blanchard và cộng sự, "Machine Learning with Adversaries: Byzantine
Tolerant Gradient Descent (Krum)," *NeurIPS*, 2017.
[3] D. Yin và cộng sự, "Byzantine-Robust Distributed Learning: Trimmed Mean and
Median," *ICML*, 2018.
[4] E. M. El Mhamdi và cộng sự, "The Hidden Vulnerability of Distributed Learning
(Bulyan)," *ICML*, 2018.
[5] G. Baruch, M. Baruch, Y. Goldberg, "A Little Is Enough (LIE)," *NeurIPS*, 2019.
[6] E. Bagdasaryan và cộng sự, "How to Backdoor Federated Learning," *AISTATS*, 2020.
[7] C. Xie và cộng sự, "DBA: Distributed Backdoor Attacks on FL," *ICLR*, 2020.
[8] M. Fang và cộng sự, "Local Model Poisoning Attacks to Byzantine-Robust FL,"
*USENIX Security*, 2020.
[9] V. Shejwalkar, A. Houmansadr, "Manipulating the Byzantine (Min-Max/DnC),"
*NDSS*, 2021.
[10] X. Cao và cộng sự, "FLTrust: Byzantine-robust FL via Trust Bootstrapping,"
*NDSS*, 2021.
[11] T. D. Nguyen và cộng sự, "FLAME: Taming Backdoors in FL," *USENIX Security*, 2022.
[12] Z. Zhang và cộng sự, "FLDetector: Defending FL Against Model Poisoning via
Detecting Malicious Clients," *KDD*, 2022.
[13] Z. Zhang và cộng sự, "Neurotoxin: Durable Backdoors in FL," *ICML*, 2022.
[14] H. Li và cộng sự, "3DFed: Adaptive and Extensible Framework for Covert
Backdoor Attack in FL," *IEEE S&P*, 2023.
[15] Y. Dai, S. Li, "Chameleon: Adapting to Peer Images for Planting Durable
Backdoors in FL," *ICML*, 2023.
[16] H. Wang và cộng sự, "Attack of the Tails: Edge-case Backdoors in FL,"
*NeurIPS*, 2020.

---

*Xác nhận của CBHD* — *Học viên* (ký, ghi rõ họ tên) — TP.HCM, ngày … tháng … năm …
