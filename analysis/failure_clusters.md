# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Kỳ Anh<br>
**Ngày:** 26/08/2026<br>
**Phạm vi:** 50 câu, 3 distributions, 4 RAGAS metrics

## 1. Aggregate RAGAS scores

| Metric | Factual (20) | Multi-hop (20) | Adversarial (10) |
|---|---:|---:|---:|
| Faithfulness | 0.9750 | 0.7400 | 0.8667 |
| Answer relevancy | 0.8776 | 0.6978 | 0.8495 |
| Context precision | 0.9250 | 0.6917 | 0.9500 |
| Context recall | 1.0000 | 0.7875 | 0.7667 |
| **Average score** | **0.9444** | **0.7292** | **0.8582** |

Điểm trung bình toàn bộ tập là **0.8411**. Multi-hop là distribution yếu nhất; factual mạnh nhất.

## 2. Bottom 10 questions

| Rank | ID | Distribution | Question | Average | Worst metric |
|---:|---:|---|---|---:|---|
| 1 | 37 | multi_hop | Tự xóa malware rồi chia sẻ sự cố trên Slack vi phạm gì? | 0.2697 | faithfulness |
| 2 | 24 | multi_hop | Tạm ứng 15 triệu, thanh toán sau 20 ngày, bị phạt bao nhiêu? | 0.3750 | faithfulness |
| 3 | 21 | multi_hop | Senior 9 năm có bao nhiêu phép và khoảng lương nào? | 0.3750 | answer_relevancy |
| 4 | 30 | multi_hop | So sánh bảo hiểm thử việc và chính thức. | 0.3750 | answer_relevancy |
| 5 | 35 | multi_hop | Junior P1 thử việc nhận lương và phụ cấp tháng đầu ra sao? | 0.5850 | context_recall |
| 6 | 33 | multi_hop | Manager 12 năm có tổng phụ cấp và phép năm bao nhiêu? | 0.6034 | context_precision |
| 7 | 31 | multi_hop | Công tác 2 ngày, khách sạn 1,5 triệu/đêm được trả tối đa bao nhiêu? | 0.6809 | faithfulness |
| 8 | 27 | multi_hop | Thâm niên 7 năm và bị trừ 4 ngày ốm thì còn bao nhiêu phép? | 0.7105 | context_precision |
| 9 | 39 | multi_hop | So sánh policy mật khẩu v1.0 và v2.0. | 0.7291 | context_precision |
| 10 | 3 | factual | Phụ cấp ăn trưa hàng tháng là bao nhiêu? | 0.7441 | context_precision |

## 3. Failure cluster matrix

Mỗi câu đóng góp một lần vào metric thấp nhất của chính câu đó.

| Worst metric | Factual | Multi-hop | Adversarial | Total |
|---|---:|---:|---:|---:|
| Faithfulness | 1 | 7 | 3 | 11 |
| Answer relevancy | 17 | 7 | 2 | 26 |
| Context precision | 2 | 3 | 1 | 6 |
| Context recall | 0 | 3 | 4 | 7 |

## 4. Dominant failure analysis

**Dominant distribution:** `multi_hop` (average 0.7292)<br>
**Dominant metric by cluster count:** `answer_relevancy` (26/50)

Multi-hop chiếm 9/10 câu cuối bảng vì một câu phải truy xuất nhiều policy rồi tính toán hoặc so sánh. Chỉ cần thiếu một chunk hoặc dùng nhầm phiên bản là câu trả lời vừa thiếu ý vừa dễ sai số. Answer relevancy thường là metric thấp nhất, nhưng với factual đây chủ yếu là mức thấp tương đối trong một nhóm vốn có điểm cao; điểm tuyệt đối 0.8776 vẫn tốt. Tín hiệu nghiêm trọng hơn là faithfulness multi-hop 0.7400 và context precision multi-hop 0.6917.

## 5. Root causes và fixes

| Metric yếu | Root cause quan sát được | Cải tiến đề xuất |
|---|---|---|
| Faithfulness | Mô hình suy diễn hoặc tính toán vượt quá bằng chứng truy xuất | Buộc trích nguồn theo từng mệnh đề, temperature 0, hậu kiểm số học và từ chối khi thiếu dữ kiện |
| Context recall | Một truy vấn chứa nhiều policy nhưng top-k không phủ đủ | Tách truy vấn thành sub-query, hybrid BM25+dense, mở rộng parent chunk và hợp nhất theo source |
| Context precision | Nhiều phiên bản hoặc chunk gần nghĩa lọt vào top-k | Rerank shortlist, lọc metadata `effective_date/version`, ưu tiên policy mới nhất trừ khi người dùng hỏi bản cũ |
| Answer relevancy | Câu trả lời không bao phủ đủ từng vế | Prompt có checklist theo vế câu hỏi và bước tự kiểm trước khi trả lời |

## 6. Adversarial distribution

Adversarial đạt 0.8582, thấp hơn factual 0.9444 nên kết quả phản ánh đúng độ khó của version conflict và negation trap, đồng thời đủ điều kiện bonus Phase A. Context recall là điểm yếu rõ nhất của nhóm này (0.7667): pipeline có thể lấy đúng một phiên bản nhưng bỏ sót tài liệu đối chiếu. Không có adversarial case trong bottom 10 vì các câu multi-hop còn khó hơn; điều này cho thấy cơ chế source label và hướng dẫn ưu tiên policy hiện hành đã giảm đáng kể nhầm lẫn phiên bản, nhưng chưa loại bỏ hoàn toàn rủi ro.
