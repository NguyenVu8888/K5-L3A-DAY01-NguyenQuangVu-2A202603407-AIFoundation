# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603407
- **Họ và tên:** Nguyễn Quang Vũ
- **SEED:** 550

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

Bảng so sánh tổng hợp 4 thế hệ AI trên bài toán lọc SMS Spam (SEED = 550):

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.945 | 0.895 | 0.919 | 0.900 | 0.45s | 18,500 | 0.04 ms | 0.08 ms | $0.0000 |
| **2. BoW + MLP (Deep Learning)** | 0.938 | 0.903 | 0.920 | 0.900 | 2.80s | 6,200 | 0.15 ms | 0.28 ms | $0.0000 |
| **3a. Encoder (MiniLM) + kNN** | 0.966 | 0.887 | 0.925 | 0.923 | 4.20s | 480 | 12.50 ms | 18.20 ms | $0.0000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.981 | 0.935 | 0.957 | 0.975 | 4.60s | 520 | 11.80 ms | 17.50 ms | $0.0000 |
| **3c. Encoder (MiniLM) + MLP head**| 0.972 | 0.935 | 0.953 | 0.975 | 5.40s | 490 | 12.20 ms | 18.00 ms | $0.0000 |
| **4a. LLM nhỏ zero-shot (`gpt-6-luna`)** | 0.952 | 1.000 | 0.976 | 0.976 | 0.00s | 1.65 | 580.00 ms | 950.00 ms | $0.0105 |
| **4b. LLM lớn zero-shot (`gpt-6.1-terra`)**| 1.000 | 1.000 | 1.000 | 1.000 | 0.00s | 0.85 | 1150.00 ms | 1800.00 ms | $0.1050 |
| **4c. LLM nhỏ few-shot (`gpt-6-luna`)** | 0.952 | 1.000 | 0.976 | 0.976 | 0.00s | 1.25 | 750.00 ms | 1200.00 ms | $0.0380 |

*Ghi chú:*
- Cột `F1 (SUB 40)` dùng để so sánh công bằng cả 8 mô hình (vì LLM chỉ chạy trên tập con SUB 40 tin để tối ưu chi phí API).
- Độ trễ inference đo từ văn bản thô đến nhãn cuối cùng (bao gồm cả tokenize / vector hoá / gọi API).

---

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất so với kết quả thực tế:

1. **Dự đoán `dl_curve` (Hiệu năng của Deep Learning MLP so với Logistic Regression):**
   - *Đoán trước:* MLP có lớp ẩn phi tuyến sẽ học được các tương tác từ phức tạp, vượt trội hơn LogReg từ 0.03 - 0.05 F1.
   - *Thực tế:* F1 test của MLP (0.920) gần như bằng phẳng so với LogReg (0.919), trong khi tham số gấp 64 lần (128.129 vs 2.001) và bắt đầu overfit mạnh từ epoch 7.
   - *Vì sao lệch:* BoW đã biến câu thành vector nhị phân thưa thớt, làm mất trật tự ngữ pháp và quan hệ cú pháp. Trên tập dữ liệu nhỏ (~3.600 tin train), biên phân cách tối ưu vốn dĩ gần như tuyến tính; việc ép thêm 64 nơ-ron chỉ làm tăng nguy cơ ghi nhớ nhiễu (overfitting) chứ không khai thác thêm được thông tin mới.

2. **Dự đoán `llm_fewshot` (Tác động của Few-shot đối với LLM):**
   - *Đoán trước:* Thêm 4 ví dụ mẫu lấy từ tập train vào prompt sẽ giúp LLM định hình ngữ cảnh tốt hơn và cải thiện F1 ít nhất +0.02 - 0.04.
   - *Thực tế:* F1 trên tập SUB không hề tăng (0.976 bằng hệt bản zero-shot), trong khi chi phí token tăng gấp ~3.6 lần ($0.0380 vs $0.0105 / 1k tin) và độ trễ p95 tăng từ 950 ms lên 1200 ms.
   - *Vì sao lệch:* Mô hình ngôn ngữ lớn hiện đại (`gpt-6-luna`) đã có năng lực Zero-shot cực mạnh nhờ pre-train trên kho ngữ liệu khổng lồ. Nhiệm vụ phân loại tin nhắn rác vốn có định nghĩa nhãn rất rõ ràng, nên 4 ví dụ đơn giản không cung cấp thêm tri thức mới mà chỉ làm phình độ dài prompt (token input).

3. **Dự đoán `emb_search` (Số từ của câu query có trong từ điển BoW):**
   - *Đoán trước:* Câu query ngụy trang sẽ có ít nhất 70% từ nằm trong từ điển BoW vì đều là các từ tiếng Anh thông dụng.
   - *Thực tế:* Chỉ có 3/13 từ của câu xuất hiện trong top 2000 từ phổ biến nhất của tập train (các từ cốt lõi như 'complimentary', 'holiday' hoàn toàn vắng bóng).
   - *Vì sao lệch:* Tập train SMS Spam có kích thước hữu hạn và thiên lệch về một số từ khóa đặc thù (call, txt, win, free). Khi kẻ lừa đảo dùng từ đồng nghĩa trang trọng, BoW lập tức rơi vào trạng thái Out-of-Vocabulary (OOV) mù tịt, trong khi Embedding vẫn bắt trọn vẹn ngữ nghĩa nhờ không gian vector liên tục.

---

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** Hai mô hình dựa trên Bag-of-Words là **BoW + LogReg** và **BoW + MLP** tụt dốc thảm hại nhất, F1 từ ~0.92 ở test thường rơi xuống chỉ còn ~0.46 - 0.52 trên 50 case khó (độ chính xác chỉ đạt ~50-54%, tương đương tung đồng xu ngẫu nhiên).
- **Lý do điểm test thường quá lạc quan:** Tập test thường được lấy cùng phân phối (i.i.d) từ tập dữ liệu UCI Spam gốc, vốn chứa nhiều mẫu spam truyền thống lặp đi lặp lại cùng các từ khóa quen thuộc. Ngược lại, bộ 50 case khó được thiết kế đối kháng (adversarial) để tấn công trực tiếp vào các điểm mù của mô hình.
- **Nhóm case gây sụp đổ:** Nhóm *Né từ khoá (Obfuscation)* và nhóm *Tưởng spam nhưng không phải (Ham looks like spam)*.

### Hai case cụ thể và cơ chế khiến mô hình sai:

1. **Case #29 (Nhóm: Né từ khoá):**
   - *Văn bản:* `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"` (Nhãn thật: `spam`).
   - *Kết quả:* BoW đoán là `ham` (SAI); Embedding và LLM đoán là `spam` (ĐÚNG).
   - *Cơ chế sai:* Kẻ tấn công thay ký tự bằng số (`0` thay `o`, `4` thay `a`, `1` thay `i`). Bộ tokenizer BoW coi `c0ngr4tulat10ns` và `cl4im` là các chuỗi ký tự lạ không tồn tại trong từ điển (OOV). Tổng trọng số spam của câu bằng 0, mô hình mặc định gán nhãn ham. Trong khi đó, subword tokenizer và mô hình ngữ nghĩa (Transformer) vẫn nhận diện được sự tương đồng về biểu diễn với lời mời nhận thẻ quà tặng.

2. **Case #2 (Nhóm: Tưởng spam nhưng không phải):**
   - *Văn bản:* `"FREE pizza in the office kitchen right now, come quick before it's gone"` (Nhãn thật: `ham`).
   - *Kết quả:* BoW đoán là `spam` (SAI); Embedding + LogReg đoán là `ham` (ĐÚNG).
   - *Cơ chế sai:* Tin nhắn chứa từ khóa `"FREE"` viết hoa cùng tính cấp bách `"now", "quick"`. Trong BoW, từ `"free"` có trọng số dương đẩy về spam lớn nhất trong toàn bộ từ điển. Mô hình BoW chỉ cộng dồn trọng số từ độc lập mà không hiểu được ngữ cảnh đồng nghiệp mời ăn pizza ở văn phòng, dẫn đến chặn nhầm (False Positive).

---

## 5. Thử thách cá nhân

- **SEED:** 550 $\rightarrow$ `MY_CHALLENGE = 550 % 6 = 4` (Đề thử thách #5).
- **Nội dung đề:** *Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.*
- **Cách triển khai (Code thực hiện):**
  ```python
  c_cases = [
      ("Special exclusive proposition: Collect your brand-new smartphone at zero charge now", 1),
      ("Notice: Your subscription package perks run out today, reply YES to prolong", 1),
      ("Are you free this weekend to win that tennis tournament together?", 0),
      ("The company won the prize for best workplace, free coffee tomorrow for all!", 0)
  ]
  c_txt = [t for t, _ in c_cases]; c_true = np.array([l for _, l in c_cases])
  p_b = (predict_proba(vectorize(c_txt, vocab), w, b) >= 0.5).astype(int)
  p_e = (predict_proba((emb_model.encode(c_txt) - mu) / sd, w_e, b_e) >= 0.5).astype(int)
  ```
- **Kết quả đo được:**
  - Tin 1 (Spam né từ): `"Special exclusive proposition: Collect your brand-new smartphone at zero charge now"`
    $\rightarrow$ BoW đoán **ham** (✗), Embedding đoán **spam** (✓).
  - Tin 2 (Spam né từ): `"Notice: Your subscription package perks run out today, reply YES to prolong"`
    $\rightarrow$ BoW đoán **ham** (✗), Embedding đoán **spam** (✓).
  - Tin 3 (Ham có từ khóa): `"Are you free this weekend to win that tennis tournament together?"`
    $\rightarrow$ BoW đoán **spam** (✗), Embedding đoán **ham** (✓).
  - Tin 4 (Ham có từ khóa): `"The company won the prize for best workplace, free coffee tomorrow for all!"`
    $\rightarrow$ BoW đoán **spam** (✗), Embedding đoán **ham** (✓).
- **Tổng kết:** BoW đúng 0/4 tin (thất bại hoàn toàn); Embedding + LogReg đúng 4/4 tin (chính xác tuyệt đối).
- **Giải thích cơ chế:**
  - Với 2 tin spam né từ: Kẻ lừa đảo thay `free` bằng `zero charge`, thay `claim` bằng `collect`. BoW bị vô hiệu hóa vì các từ này không nằm trong danh sách từ khóa spam trọng số cao. Ngược lại, Embedding nhận diện ngữ nghĩa tổng thể là chiêu trò lừa tặng điện thoại và gia hạn dịch vụ để chiếm đoạt tiền.
  - Với 2 tin ham: Có chứa cả `free`, `win`, `prize` nhưng là rủ bạn bè đánh tennis hoặc thông báo nội bộ công ty. BoW thấy từ khóa là kích hoạt nhãn spam, còn Embedding nhận diện được văn phong trò chuyện tự nhiên và môi trường công sở nên phân loại đúng là ham.

---

## 6. Phương án cho sản phẩm thật

### Đặt bài toán:
Nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình, cao điểm ~500 tin/giây), yêu cầu nghiêm ngặt **p95 < 50 ms/tin**.

### 1. Phương án lựa chọn và đối chiếu số liệu:
- **Lựa chọn tối ưu:** **3b. Encoder (`all-MiniLM-L6-v2`) + Logistic Regression**.
- **Dẫn chứng số liệu:**
  - **Độ trễ:** p95 khi gọi từng tin là **17.50 ms** (đáp ứng xuất sắc SLA < 50 ms, biên an toàn gấp gần 3 lần).
  - **Throughput:** ~520 tin/s trên 1 worker CPU tiêu chuẩn, dễ dàng chịu tải 116 tin/s trung bình chỉ với 1-2 instance.
  - **Chất lượng:** Đạt F1 rất cao (**0.957 trên toàn bộ test** và **0.975 trên SUB 40**), vượt trội hơn BoW (+0.04 F1) và chống chọi tốt trước các tin nhắn ngụy trang.
  - **Chi phí API:** **0 USD**. Chạy hoàn toàn on-premise / edge trên CPU sẵn có, không phụ thuộc kết nối Internet ra bên ngoài, không lo lộ lọt thông tin cá nhân của khách hàng.

### 2. Vì sao KHÔNG dùng LLM làm giải pháp độc lập duy nhất?
- **Độ trễ vi phạm SLA trầm trọng:** LLM nhỏ (`gpt-6-luna`) có độ trễ p95 lên tới **950 ms** (chậm gấp 19 lần mức cho phép); LLM lớn (`gpt-6.1-terra`) lên tới **1.800 ms** (chậm gấp 36 lần mức cho phép). Hệ thống viễn thông sẽ bị nghẽn nghẹt lập tức (queueing explosion).
- **Chi phí khổng lồ:**
  - Dùng `gpt-6-luna`: $10.50/1 triệu tin $\rightarrow$ 10 triệu tin/ngày tốn **$105/ngày** $\approx$ **$3.150/tháng** (~80 triệu VNĐ/tháng).
  - Dùng `gpt-6.1-terra`: $105/1 triệu tin $\rightarrow$ tốn **$1.050/ngày** $\approx$ **$31.500/tháng** (~800 triệu VNĐ/tháng).

### 3. Thiết kế hệ thống lai tối ưu trong thực tế (Router / Cascade Architecture):
Để vừa bảo đảm SLA thời gian thực, chi phí rẻ, vừa tận dụng trí tuệ vượt trội của LLM cho các ca lừa đảo tinh vi:
- **Tầng 1 (Fast Filter - 95% traffic):** Mô hình **3b (Encoder + LogReg)** tiếp nhận 100% tin nhắn.
  - Nếu xác suất $p(spam) < 0.15$: Chốt nhãn `ham` ngay lập tức (Latency ~15 ms, Cost = $0).
  - Nếu xác suất $p(spam) > 0.85$: Chốt nhãn `spam` ngay lập tức (Latency ~15 ms, Cost = $0).
- **Tầng 2 (Deep Inspection - 5% traffic lưỡng lự):** Chỉ 5% tin nhắn rơi vào vùng không chắc chắn $p \in [0.15, 0.85]$ mới được đẩy vào hàng đợi bất đồng bộ (async queue) gửi tới **LLM với prompt phòng thủ**.
  - Độ trễ tổng thể p95 của toàn hệ thống vẫn giữ nguyên ở mức **< 20 ms**.
  - Chi phí LLM giảm 95%, chỉ còn khoảng **$150/tháng**, một con số hoàn toàn chấp nhận được cho bài toán an ninh viễn thông.

---

## 7. AI log và điều đã học

### Bảng khai báo sử dụng AI:

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời / viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 1** | *Giải thích công thức cập nhật gradient descent cho bài toán Binary Cross Entropy với hồi quy Logistic* | AI đưa ra công thức toán $grad\_w = \frac{1}{n} X^T (p - y)$ và giải thích ý nghĩa sai số $(p - y)$. | Tự code vector hoá ma trận bằng Numpy thuần, kiểm chứng chiều ma trận $(n, d) \times (d,) \rightarrow (n,)$ và chạy qua test tự động `1.3 train_logreg`. |
| **Phần 2** | *Nguyên nhân gây bùng nổ gradient trong vòng lặp PyTorch và vì sao phải gọi zero_grad?* | AI giải thích cơ chế autograd của PyTorch mặc định cộng dồn `param.grad += grad_new`. | Viết lại bằng văn phong của mình vào mục Giải thích 2, đối chiếu với code vòng lặp 5 dòng và test `2.2 train_mlp`. |
| **Phần 4** | *Viết regex an toàn để parse nhãn 'spam' / 'ham' tránh bị match nhầm câu 'Not spam'* | AI gợi ý kiểm tra cụm `'not spam'` trước, sau đó dùng `re.search(r'\bspam\b')`. | Tự hoàn thiện hàm `parse_label`, bổ sung xử lý tiền tố `'Label:'`, strip dấu câu và kiểm tra vượt qua cả 7 test case khó của test `4.2–4.3`. |

### Điều hiểu ra hôm nay mà trước đó chưa hiểu:
1. **Biểu diễn đặc trưng (Representation) quyết định giới hạn của mô hình, không phải thuật toán phức tạp:** Tôi từng nghĩ cứ đưa Deep Learning (MLP) vào là kết quả sẽ tự động nhảy vọt. Nhưng thực tế trên Bag-of-Words thưa thớt, MLP không hề vượt trội hơn Logistic Regression mà chỉ tăng tham số và nguy cơ overfit. Ngược lại, khi chuyển sang biểu diễn Embedding (MiniLM) phản ánh đúng ngữ nghĩa, chỉ cần một lớp Logistic Regression tuyến tính đơn giản cũng đủ đạt F1 trên 0.95.
2. **Độ chính xác (Accuracy) là chỉ số cực kỳ nguy hiểm trong dữ liệu lệch:** Với dữ liệu chỉ có ~13% spam, một mô hình hoàn toàn mù tịt đoán "luôn là ham" đã tự động có 87% accuracy. Trong thực tế an ninh mạng, việc tối ưu hóa F1-score (cân bằng Precision và Recall) là bắt buộc.
3. **Đánh đổi đa chiều trong sản phẩm thực tế:** LLM rất thông minh và không cần train, nhưng trong môi trường production với hàng chục triệu người dùng, rào cản chí tử là độ trễ mạng (p95 hàng nghìn ms) và chi phí API tích lũy. Mô hình Router phân tầng (kết hợp ML/Embedding nhỏ chạy tại biên và LLM cho ca khó) là bài học kiến trúc giá trị nhất mà tôi học được hôm nay.
