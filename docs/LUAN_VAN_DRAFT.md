# LUẬN VĂN THẠC SĨ (BẢN THẢO)
## Nghiên cứu kỹ thuật đầu độc mô hình tàng hình và thích nghi đa ràng buộc trong Học Liên kết trên dữ liệu phi-IID

> Bản thảo nội dung 5 chương. Số liệu thực nghiệm là **kết quả thật (3 seed,
> MNIST)** từ kho mã đính kèm; các ô đánh dấu `[CHỜ run_baselines]` cần điền sau
> khi chạy `run_baselines.py`. Định dạng nộp: Times New Roman 13, giãn dòng 1.5,
> lề 3.5/3/3.5/2 cm.

---

# CHƯƠNG 1. CƠ SỞ LÝ THUYẾT

## 1.1. Học Liên kết và thuật toán FedAvg

Học Liên kết (Federated Learning – FL) là mô hình huấn luyện phân tán trong đó
một máy chủ điều phối $N$ máy khách cùng xây dựng một mô hình toàn cục $w$ mà
không tập trung dữ liệu thô. Tại mỗi vòng giao tiếp $t$, máy chủ chọn một tập con
máy khách, gửi mô hình hiện tại $w^t$; mỗi máy khách $i$ huấn luyện cục bộ trên
dữ liệu riêng $D_i$ rồi gửi về mô hình cục bộ $w_i^{t+1}$. Thuật toán **FedAvg**
(McMahan và cộng sự, 2017) tổng hợp theo trung bình có trọng số kích thước dữ
liệu:
$$ w^{t+1} = \sum_i \frac{n_i}{\sum_j n_j}\, w_i^{t+1}. $$

**Không gian trọng số và không gian cập nhật.** Đặt cập nhật của máy khách là
$u_i = w_i^{t+1} - w^t$. Vì $\sum_i p_i = 1$ với $p_i = n_i/\sum_j n_j$, ta có
$w^{t+1} = w^t + \sum_i p_i u_i$. Luận văn làm việc nhất quán trong **không gian
cập nhật** $u_i$ vì mọi luật tổng hợp Byzantine-robust đều đo chuẩn, khoảng cách
và độ tương tự cosine trên cập nhật (gradient), không phải trên giá trị trọng số
tuyệt đối.

## 1.2. Dữ liệu IID và phi-IID

Trong FL thực tế, dữ liệu giữa các máy khách thường **không độc lập và đồng nhất
(phi-IID)**. Luận văn mô phỏng mức bất đồng nhất bằng phân phối **Dirichlet** với
tham số tập trung $\alpha$: với mỗi lớp nhãn, tỉ lệ phân bổ cho các máy khách lấy
từ $\text{Dir}(\alpha)$. $\alpha$ nhỏ (vd 0.1) → mỗi máy khách chỉ có vài nhãn
(bất đồng nhất mạnh); $\alpha$ lớn → tiệm cận IID. Tham số $\alpha$ là **biến độc
lập chính** của luận văn.

## 1.3. Tấn công đầu độc trong FL

**Phân loại.** *Data poisoning* sửa dữ liệu huấn luyện; *model poisoning* thao
túng trực tiếp cập nhật gửi lên. Theo mục tiêu: *untargeted* (làm giảm độ chính
xác) và *targeted/backdoor* (tinh vi hơn).

**Tấn công backdoor.** Kẻ tấn công cấy một hành vi ẩn: mô hình vẫn chính xác trên
tác vụ chính nhưng phân loại các đầu vào mang **tín hiệu kích hoạt (trigger)**
sang **nhãn mục tiêu** định trước. Luận văn dùng trigger là ô vuông trắng ở góc
phải-dưới ảnh và nhãn mục tiêu cố định.

**Mô hình đe dọa.** Kẻ tấn công kiểm soát $m$ máy khách độc (10–30%), được lấy
mẫu ngẫu nhiên như máy khách thường (tham gia ngắt quãng), có thể gửi cập nhật
tuỳ ý. Hai cấp tri thức: **thực tế** (chỉ ước lượng thống kê benign từ chính dữ
liệu độc) và **white-box thích nghi** (biết luật phòng thủ — cận trên toàn tri).

## 1.4. Các luật tổng hợp Byzantine-robust (phòng thủ)

- **Krum / Multi-Krum** (Blanchard 2017): chọn cập nhật có tổng bình phương
  khoảng cách tới $n-f-2$ láng giềng gần nhất là nhỏ nhất (giả định cập nhật độc
  nằm xa tâm).
- **Median / Trimmed-Mean** (Yin 2018): trung vị / trung bình sau khi cắt $f$
  cực trị theo từng toạ độ.
- **Bulyan** (Mhamdi 2018): chọn lõi bằng Multi-Krum rồi trimmed-mean.
- **FLTrust** (Cao 2021): máy chủ giữ một **tập gốc sạch** nhỏ, tự tính cập nhật
  tham chiếu $u_0$; trọng số tin cậy của máy khách là $\text{ReLU}(\cos(u_i,
  u_0))$, mỗi cập nhật được chuẩn hoá về $\lVert u_0\rVert$ trước khi trung bình.
- **FLAME** (Nguyen 2022): phân cụm theo khoảng cách cosine để loại nhóm thiểu số
  bất thường, cắt chuẩn về trung vị, rồi cộng nhiễu Gaussian. *(Luận văn dùng bản
  thay HDBSCAN bằng lựa chọn lõi-đa-số theo cosine để không phụ thuộc thư viện —
  giữ nguyên ba trụ lọc-cosine + cắt-chuẩn + nhiễu.)*

## 1.5. Các độ đo đánh giá

- **ASR (Attack Success Rate):** tỉ lệ mẫu (không thuộc lớp mục tiêu) gắn trigger
  bị phân loại thành lớp mục tiêu.
- **Main Accuracy (MA):** độ chính xác trên tập kiểm thử sạch (tính tàng hình).
- **Evasion Rate:** tỉ lệ (%) cập nhật độc được luật tổng hợp **chấp nhận** (không
  bị lọc) mỗi vòng. Với phòng thủ theo toạ độ (Median) — không loại theo client —
  quy ước 100%.
- **Durability (retention):** $100\times \text{ASR}_{\text{cuối}}/
  \text{ASR}_{\text{tại lúc rời}}$ — phần backdoor còn lại sau khi kẻ tấn công
  ngừng tham gia.

---

# CHƯƠNG 2. KHẢO SÁT THỰC TRẠNG VÀ LỖ HỔNG NGHIÊN CỨU

## 2.1. Thực trạng các kỹ thuật tấn công

- **LIE** (Baruch 2019): gửi $\mu_{\text{benign}} - z\,\sigma_{\text{benign}}$ —
  "một chút là đủ" để qua mặt bộ lọc thống kê.
- **Min-Max / DnC** (Shejwalkar & Houmansadr 2021) và **Fang** (2020): tấn công
  **thích nghi theo luật tổng hợp (AGR-tailored)** — chuẩn vàng để đánh giá.
- **Model Replacement** (Bagdasaryan 2020): nhân tỉ lệ cập nhật để thay thế mô
  hình; **DBA** (Xie 2020): phân tán trigger.
- **Neurotoxin** (Zhang 2022): tăng độ bền backdoor bằng cách cấy vào tham số ít
  được cập nhật.
- **3DFed** (Li 2023): khung thích nghi black-box dùng chỉ báo + mô hình mồi nhử.

## 2.2. Thực trạng phòng thủ và điểm yếu "đơn đặc trưng"

Mỗi họ phòng thủ chủ yếu kiểm tra **một** đặc trưng: độ lớn (Krum, Trimmed-Mean),
hướng (FLTrust), hay phân cụm + nhiễu (FLAME). Hệ quả: một tấn công vượt được lớp
này thường gãy trước lớp khác. Chưa có khung tấn công **đa ràng buộc** đánh giá
trên **tổ hợp** nhiều phòng thủ.

## 2.3. Phi-IID: thách thức hội tụ hay rủi ro an ninh?

Phi-IID làm tăng phương sai tự nhiên của cập nhật benign, khiến ranh giới giữa
"lệch do dữ liệu đặc thù" và "lệch do độc hại" mờ đi. Vai trò của phi-IID **như
một yếu tố an ninh** (chứ không chỉ là thách thức hội tụ) chưa được khảo sát định
lượng qua quét $\alpha$ một cách hệ thống.

## 2.4. Định vị đóng góp của luận văn

| Năng lực | Model Repl. | LIE/Min-Max | Neurotoxin | 3DFed | **GeoTox** |
|---|---|---|---|---|---|
| Vượt phòng thủ khoảng cách | Kém | TB | Tốt | Tốt | Có |
| Vượt phòng thủ định hướng | Kém | Kém | Kém | Tốt | Có |
| Thích nghi theo AGR | Không | Một phần | Không | Có | **Có** |
| Độ bền backdoor | Thấp | Thấp | **Rất cao** | TB | Có (tùy chọn) |
| Khảo sát hệ thống theo $\alpha$ | Không | Không | Không | Kém | **Có** |
| Đặc trưng hóa trade-off | Không | Không | Không | Không | **Có** |

*Phân biệt trùng tên:* "Chameleon" (Dai 2023) dùng học đối lập trong không gian
đặc trưng — khác hoàn toàn; luận văn dùng tên **GeoTox**.

## 2.5. Phát biểu bài toán và giả thuyết nghiên cứu

- **RQ1:** Quan hệ đánh đổi giữa độ tàng hình (Evasion) và hiệu quả (ASR) của một
  tấn công đa ràng buộc dưới các họ phòng thủ là gì?
- **RQ2:** Mức phi-IID ($\alpha$) dịch chuyển ranh giới đó về phía kẻ tấn công ra
  sao?
- **RQ3:** Backdoor tồn tại bao lâu sau khi kẻ tấn công rời mạng?

---

# CHƯƠNG 3. KỸ THUẬT ĐỀ XUẤT: GEOTOX VÀ GEOTOX-ADAPTIVE

## 3.1. Ý tưởng tổng quát

GeoTox là một kỹ thuật **model poisoning đa ràng buộc**: nhào nặn cập nhật độc
(đã mang tín hiệu backdoor qua huấn luyện trên dữ liệu có trigger) sao cho đồng
thời thoả mãn nhiều ràng buộc tàng hình, qua đó né nhiều họ phòng thủ cùng lúc.

## 3.2. Tấn công trong không gian cập nhật

Mỗi máy khách độc huấn luyện cục bộ trên dữ liệu nhiễm trigger để thu được cập
nhật thô $u_{\text{raw}} = w_{\text{local}} - w^t$ (mang tín hiệu backdoor), rồi
biến đổi $u_{\text{raw}}$ theo các bước dưới đây trước khi gửi.

## 3.3. Ràng buộc tàng hình hướng (núm đánh đổi $\tau$)

Gọi $\mu_b$ là cập nhật trung bình của nhóm benign (ước lượng được), $\hat\mu =
\mu_b/\lVert\mu_b\rVert$. Đặt $d = u_{\text{raw}}/\lVert u_{\text{raw}}\rVert$ và
$c = \langle d, \hat\mu\rangle$. Ta tìm hệ số hoà trộn nhỏ nhất
$\lambda\in[0,1]$ sao cho vector $v=\lambda\hat\mu+(1-\lambda)d$ đạt
$\cos(v,\hat\mu)\ge\tau$. Vì $\cos(v,\hat\mu)$ tăng đơn điệu theo $\lambda$
(từ $c$ tới $1$), $\lambda$ được tìm bằng **tìm kiếm nhị phân**. Tham số
$\tau\in[0,1]$ chính là **núm điều khiển đánh đổi**: $\tau$ thấp giữ nguyên hướng
backdoor (hiệu quả cao, dễ bị lộ); $\tau$ cao ép update giống benign (tàng hình
cao, backdoor yếu đi).

## 3.4. Ràng buộc tàng hình độ lớn

Cập nhật cuối được tái tỉ lệ về **trung vị chuẩn của benign** $B=\text{median}_i
\lVert u_i\rVert$: $\;u_{\text{send}} = B\cdot v/\lVert v\rVert$. Nhờ đó update
độc nằm trong "đám mây" benign, né các cơ chế cắt chuẩn (Krum, Trimmed-Mean, bước
cắt chuẩn của FLAME).

## 3.5. Cơ chế bền vững tuỳ chọn (mặt nạ kiểu Neurotoxin) và đánh đổi với ASR

Tuỳ chọn: giữ backdoor ở các toạ độ benign **ít biến động** nhất (mặt nạ theo
$|\mu_b|$, tham số `mask_ratio`). **Quan sát thực nghiệm quan trọng:** mặt nạ quá
mạnh (giữ 70% toạ độ, zero 30% quan trọng nhất) **làm sụp ASR** (vd từ ~84% xuống
~5% khi không phòng thủ) vì xoá luôn tín hiệu backdoor. Do đó mặc định **tắt mặt
nạ** ($\text{mask\_ratio}=1.0$); đây là một **đánh đổi độ-bền↔hiệu-quả** cần quét
có chủ đích, không bật mặc định.

## 3.6. GeoTox-Adaptive (white-box, cận trên toàn tri)

Khi biết luật phòng thủ, sau khi tạo hình GeoTox, kẻ tấn công **tìm kiếm nhị phân
hệ số khuếch đại** $s\in[1, s_{\max}]$ lớn nhất mà phòng thủ vẫn **chấp nhận** mọi
client độc (mô phỏng chính luật tổng hợp trên toàn bộ cập nhật), rồi áp dụng — tức
hoạt động ngay tại **biên chấp nhận** để tối đa cường độ backdoor.

## 3.7. Giả định tấn công phối hợp (colluding)

Các client độc gửi cập nhật **đồng nhất** sau khi tạo hình. Giả định này mô hình
hoá kẻ tấn công phối hợp; nó tạo một cụm cosine dày, có thể đánh lừa phòng thủ
phân cụm (FLAME). Luận văn trình bày minh bạch giả định này như một **điểm khai
thác** đối với phòng thủ phân cụm, đồng thời thảo luận tác động của nó tới phương
sai kết quả (Chương 4).

## 3.8. Độ phức tạp và tính khả thi

Các phép chiếu/hoà trộn và tìm kiếm nhị phân có chi phí $O(d)$–$O(\log)$; bước
adaptive lặp lại luật tổng hợp ~15 lần/vòng. Toàn bộ chạy được trên CPU với MNIST
/ Fashion-MNIST + LeNet.

---

# CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1. Thiết lập

- **Dữ liệu/mô hình:** MNIST, LeNet (mở rộng Fashion-MNIST).
- **Cấu hình FL:** 20 máy khách, 10 tham gia/vòng, 20% độc ($m=4$), 30 vòng.
- **Phi-IID:** Dirichlet $\alpha\in\{0.1,0.3,0.5,1.0\}$ và IID.
- **Phòng thủ:** FedAvg, Krum, FLTrust, FLAME (mở rộng Median/Bulyan/Trimmed/
  Norm-clip/Multi-Krum).
- **Seed:** 3 (báo cáo mean±std). **Hạ tầng:** CPU.

## 4.2. Giao thức và baseline

Mỗi cấu hình có baseline **không tấn công** (đo MA) và được so với **các tấn công
nền** LIE / Min-Max / Model-Replacement dưới cùng phòng thủ (Mục 4.5).

## 4.3. RQ1 — Đường cong đánh đổi Evasion↔ASR (MNIST, $\alpha=0.5$, 3 seed)

| Phòng thủ | $\tau{=}0.0$ | $\tau{=}0.3$ | $\tau{=}0.6$ | $\tau{=}0.9$ | Evas@0.9 | MA |
|---|---|---|---|---|---|---|
| FedAvg | 37.5±37.1 | 21.3±24.8 | 2.5±2.3 | 0.3±0.0 | 100% | 98.7 |
| **Krum** | 2.4±1.9 | 2.3±2.0 | 44.2±35.7 | **97.1±1.3** | 47% | 97.6 |
| **FLTrust** | 0.3±0.1 | 0.4±0.1 | 0.4±0.1 | 0.3±0.1 | 64% | 97.5 |
| FLAME | 38.2±44.4 | 42.3±47.9 | 23.7±19.8 | 0.4±0.1 | 100% | 98.6 |

**Nhận xét.** (i) **Krum** thể hiện đường cong đánh đổi rõ và **ổn định**: phải
nâng $\tau$ để lọt bộ lọc (Evasion 4%→47%), tại $\tau=0.9$ đạt **ASR 97.1±1.3%**
(độ lệch nhỏ) mà MA vẫn 97.6% — minh chứng mạnh cho RQ1. (ii) **FLTrust** giữ ASR
ở 0.3–0.4% tại mọi $\tau$: để khớp hướng tham chiếu *sạch* thì update phải lành
tính, triệt tiêu backdoor — **GeoTox không phá được FLTrust**. (iii) **FLAME**
phương sai rất lớn (±40–48): tuỳ seed GeoTox đôi khi xuyên (~90%), đôi khi bị
chặn (~1%) — phản ánh tính bất ổn định của phòng thủ phân cụm trước tấn công phối
hợp; **không** kết luận "đánh bại FLAME".

## 4.4. RQ2 — Độ nhạy theo phi-IID (FLAME, $\tau=0.5$, 3 seed)

| $\alpha$ | 0.1 | 0.3 | 0.5 | 1.0 | IID |
|---|---|---|---|---|---|
| ASR (%) | 54.7±40.6 | 31.6±30.4 | 28.7±22.5 | 6.9±9.6 | 16.8±21.6 |

ASR **giảm dần khi $\alpha$ tăng** (0.1→1.0), *ủng hộ định hướng* "phi-IID càng
mạnh, kẻ tấn công càng dễ ẩn náu". Tuy phương sai lớn và điểm IID chưa khớp đơn
điệu nên đây là **bằng chứng định hướng**, cần thêm seed/vòng để khẳng định.

## 4.5. So sánh GeoTox với các tấn công nền (best ASR theo phòng thủ)

> Chạy `python3 run_baselines.py` rồi `python3 analyze_results.py` (bảng *Attack
> comparison*) để điền. Mẫu bảng:

| Phòng thủ | LIE | Min-Max | Model-Repl. | **GeoTox (best τ)** | **GeoTox-Adaptive** |
|---|---|---|---|---|---|
| FedAvg | `[CHỜ]` | `[CHỜ]` | `[CHỜ]` | 37.5±37.1 | – |
| Krum | `[CHỜ]` | `[CHỜ]` | `[CHỜ]` | **97.1±1.3** | – |
| FLTrust | `[CHỜ]` | `[CHỜ]` | `[CHỜ]` | 0.4±0.1 | – |
| FLAME | `[CHỜ]` | `[CHỜ]` | `[CHỜ]` | ~42 (±lớn) | 21.3±36.3 |

*Mục tiêu lập luận:* cho thấy GeoTox (đặc biệt ở $\tau$ phù hợp / Adaptive) đạt
ASR **cao hơn hoặc tàng hình hơn** so với LIE/Min-Max/Model-Replacement dưới cùng
phòng thủ — hoặc nêu trung thực nếu một tấn công nền mạnh hơn ở tình huống nào đó.

## 4.6. RQ3 — Độ bền sau khi kẻ tấn công rời mạng (vòng 15)

Độ bền chỉ có ý nghĩa khi backdoor thực sự được cấy. Ở lần GeoTox-Adaptive xuyên
FLAME (ASR@stop 92.7%), backdoor giữ **retention ~68%** (ASR 92.7%→63.3%); các
lần không cấy được thì retention không đại diện. Kết quả gợi ý backdoor đã cấy có
độ bền đáng kể, song cần ổn định hoá bằng thêm seed.

## 4.7. Thảo luận và mối đe doạ tới tính hợp lệ (validity)

- *Internal:* phương sai cao ở $\tau$ thấp / FLAME do lấy mẫu client ngẫu nhiên +
  30 vòng + 3 seed → cần tăng vòng và seed.
- *External:* mới MNIST; cần Fashion-MNIST/CIFAR để khái quát.
- *Construct:* FLAME bản dùng là *xấp xỉ* (không HDBSCAN gốc) — cần nêu rõ.
- **Kết quả vững nhất:** đường cong đánh đổi trên Krum và khả năng phòng thủ của
  FLTrust (đều có độ lệch chuẩn nhỏ, lặp lại được).

---

# CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1. Tóm tắt đóng góp

1. **GeoTox / GeoTox-Adaptive** — kỹ thuật model poisoning đa ràng buộc (tàng
   hình hướng + độ lớn, tuỳ chọn bền vững, biến thể thích nghi white-box).
2. **Đặc trưng hóa đường cong đánh đổi** Evasion↔ASR — chứng minh rõ trên Krum.
3. **Khảo sát vai trò an ninh của phi-IID** qua quét $\alpha$.
4. **Khung benchmark tái lập được** (PyTorch, 25+ kiểm thử, script phân tích).

## 5.2. Hàm ý cho phòng thủ thế hệ sau

FLTrust (tham chiếu sạch) tỏ ra bền vững nhất trước GeoTox; trong khi phòng thủ
phân cụm (FLAME) dễ bị tấn công phối hợp đánh lừa dưới phi-IID. Gợi ý: kết hợp
tín hiệu **tham chiếu sạch + lịch sử + đa-cụm** để phân biệt client lệch-dữ-liệu
lương thiện với client độc.

## 5.3. Hạn chế

Quy mô thực nghiệm (1 dataset, 3 seed, 30 vòng); một số kết quả (FLAME, RQ2) còn
phương sai cao; FLAME là bản xấp xỉ; giả định client độc đồng nhất.

## 5.4. Hướng phát triển

- Tăng seed/vòng, thêm Fashion-MNIST/CIFAR-10 và mô hình lớn hơn.
- Thêm phòng thủ theo lịch sử (FLDetector) và phòng thủ đề xuất *heterogeneity-
  aware*.
- Mở rộng sang **secure aggregation** và **FL fine-tuning mô hình nền/LLM (LoRA)**.

---

# TÀI LIỆU THAM KHẢO

[1] McMahan và cộng sự, *Communication-Efficient Learning of Deep Networks from
Decentralized Data*, AISTATS 2017.
[2] Blanchard và cộng sự, *Krum: Byzantine-Tolerant Gradient Descent*, NeurIPS 2017.
[3] Yin và cộng sự, *Byzantine-Robust Distributed Learning (Median/Trimmed-Mean)*,
ICML 2018.
[4] El Mhamdi và cộng sự, *Bulyan*, ICML 2018.
[5] Baruch và cộng sự, *A Little Is Enough*, NeurIPS 2019.
[6] Bagdasaryan và cộng sự, *How to Backdoor Federated Learning*, AISTATS 2020.
[7] Xie và cộng sự, *DBA: Distributed Backdoor Attacks*, ICLR 2020.
[8] Fang và cộng sự, *Local Model Poisoning Attacks*, USENIX Security 2020.
[9] Shejwalkar & Houmansadr, *Manipulating the Byzantine (Min-Max/DnC)*, NDSS 2021.
[10] Cao và cộng sự, *FLTrust*, NDSS 2021.
[11] Nguyen và cộng sự, *FLAME*, USENIX Security 2022.
[12] Zhang và cộng sự, *FLDetector*, KDD 2022.
[13] Zhang và cộng sự, *Neurotoxin: Durable Backdoors in FL*, ICML 2022.
[14] Li và cộng sự, *3DFed*, IEEE S&P 2023.
[15] Dai và cộng sự, *Chameleon*, ICML 2023.
[16] Wang và cộng sự, *Attack of the Tails: Edge-case Backdoors*, NeurIPS 2020.
