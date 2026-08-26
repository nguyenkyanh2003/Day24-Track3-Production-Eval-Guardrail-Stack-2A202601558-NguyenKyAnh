# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Kỳ Anh<br>
**Ngày:** 26/08/2026<br>
**Judge model:** `gemini-3.5-flash-lite`<br>
**Calibration set:** 10 câu có nhãn người chấm

## 1. Pairwise và swap-and-average

Answer A là model answer trong calibration set; Answer B là authoritative reference. Pass 2 hoán đổi vị trí rồi được quy đổi lại về không gian A/B ban đầu.

| # | Question tóm tắt | Pass 1 | Pass 2 (đã quy đổi) | Final | Consistent |
|---:|---|:---:|:---:|:---:|:---:|
| 1 | Nghỉ kết hôn | B | B | B | Yes |
| 2 | Phê duyệt thiết bị 55 triệu | B | B | B | Yes |
| 3 | Thưởng Tết tối thiểu | B | B | B | Yes |
| 4 | Senior 9 năm: phép và lương | B | B | B | Yes |
| 5 | Hoàn trả khóa học | B | B | B | Yes |
| 6 | Tạm ứng 8 triệu quá hạn | B | B | B | Yes |
| 7 | Manager 12 năm: phụ cấp và phép | B | B | B | Yes |
| 8 | Phép năm hiện hành | B | B | B | Yes |
| 9 | Phép năm khi thử việc | A | B | tie | **No** |
| 10 | Dùng VPN cá nhân khi WFH | B | B | B | Yes |

Judge ưu tiên reference ở 9/10 case vì reference đầy đủ hơn hoặc sửa đúng lỗi policy/version của model answer. Case #9 bộc lộ position bias: judge chọn answer xuất hiện trước ở cả hai lượt, nên swap-and-average hạ kết quả cuối thành `tie` thay vì tạo một quyết định thiếu ổn định.

**Position bias:** 1/10 = **10%**.

## 2. Cohen’s κ calibration

| Question ID | Human | Judge | Agree |
|---:|---:|---:|:---:|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

- Agreement rate: **100% (10/10)**
- Cohen’s κ: **1.0000**
- Interpretation: **almost perfect agreement**; vượt điều kiện bonus κ > 0.6.

Judge nhận đúng cả lỗi thiếu người phê duyệt, tính sai phí phạt, dùng policy nghỉ phép v2023 và cho phép VPN cá nhân trái quy định. Tuy nhiên, 10 mẫu là tập calibration nhỏ; κ cao chưa đủ để khẳng định khả năng tổng quát trên dữ liệu production.

## 3. Verbosity bias

Trong 9 case có winner rõ ràng:

- A thắng và A dài hơn B: **0/9**
- B thắng và B dài hơn A: **9/9**
- Verbosity correlation: **100%**

Con số này là tín hiệu tương quan, không tự chứng minh quan hệ nhân quả: Answer B vừa dài hơn vừa là authoritative reference nên thường chính xác và đầy đủ hơn. Dù vậy, production judge vẫn có thể đánh đồng độ dài với độ đầy đủ. Cần rubric tách riêng factual correctness/completeness/conciseness, giới hạn độ dài hai answer tương đương khi calibration, và theo dõi accuracy theo bucket độ dài.

## 4. Kết luận vận hành

Judge đạt κ 1.0 và position bias chỉ 10%, vì vậy đủ tốt để làm quality gate có giám sát. Swap-and-average hữu ích vì đã phát hiện đúng một quyết định phụ thuộc vị trí. Khi triển khai production, nên giữ temperature 0, schema JSON bắt buộc, hoán đổi mọi cặp, cache kết quả, sampling human review định kỳ và không dùng judge làm nguồn chân lý duy nhất. Cần mở rộng calibration set và cân bằng độ dài answer trước khi đặt ngưỡng tự động chặn release.
