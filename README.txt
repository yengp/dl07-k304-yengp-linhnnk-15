# HƯỚNG DẪN SỬ DỤNG ỨNG DỤNG PHÂN TÍCH CẢM XÚC VÀ PHÂN CỤM THÔNG TIN

## TRUY CẬP ONLINE & MÃ NGUỒN
- Xem ứng dụng trực tuyến: https://dl07-k304-yengp-linhnnk-15.streamlit.app/
- Xem mã nguồn trên GitHub: https://github.com/yengp/dl07-k304-yengp-linhnnk-15

## 📋 TỔNG QUAN

Ứng dụng này được phát triển để phân tích cảm xúc và phân cụm thông tin từ các đánh giá về công ty IT trên nền tảng ITviec.com. Ứng dụng cung cấp 3 chức năng chính:

1. **Phân tích vấn đề kinh doanh** - Hiểu rõ bài toán và mục tiêu
2. **Xây dựng mô hình** - Tìm hiểu quy trình Data Science đầy đủ
3. **Phân tích công ty** - Sử dụng mô hình để phân tích và dự đoán

## 🚀 HƯỚNG DẪN CÀI ĐẶT

### Yêu cầu hệ thống:
- Python 3.7+
- Ít nhất 2GB RAM
- Kết nối internet (để tải thư viện)

### Các bước cài đặt:

1. **Tải về mã nguồn:**
   - Giải nén tất cả file vào một thư mục
   - Đảm bảo cấu trúc thư mục như sau:
     ```
     Thư mục chính/
     ├── streamlit_app.py
     ├── predictor.py
     ├── requirements.txt
     ├── data/
     ├── models/
     ├── img/
     └── output/
     ```

2. **Cài đặt thư viện:**
   Mở Command Prompt/Terminal và chạy:
   ```
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng:**
   ```
   streamlit run streamlit_app.py
   ```

4. **Truy cập ứng dụng:**
   - Ứng dụng sẽ tự động mở trình duyệt
   - Hoặc truy cập: http://localhost:8501

## 📖 HƯỚNG DẪN SỬ DỤNG CHI TIẾT

### 🔍 1. PHÂN TÍCH VẤN đề KINH DOANH

**Mục đích:** Hiểu rõ bài toán và giá trị ứng dụng

**Nội dung:**
- Phân tích cảm xúc từ review nhân viên
- Phân cụm thông tin đánh giá công ty
- Ứng dụng thực tế trong tuyển dụng

**Cách sử dụng:**
- Chọn "💼 Phân tích nghiệp vụ" trong menu
- Đọc thông tin về 2 vấn đề chính
- Xem ví dụ ứng dụng trong thực tế

### 🛠️ 2. XÂY DỰNG MÔ HÌNH

**Mục đích:** Tìm hiểu quy trình Data Science đầy đủ

**Các bước:**

#### Bước 1: Business Understanding
- Mục tiêu dự án rõ ràng
- Tiêu chí thành công
- Bài toán kinh doanh

#### Bước 2: Data Understanding  
- Khám phá 8,417 reviews từ 200+ công ty IT
- Hiểu cấu trúc 13 cột dữ liệu ban đầu
- Đánh giá chất lượng dữ liệu

#### Bước 3: Data Preparation
- Xử lý văn bản tiếng Việt
- Feature Engineering
- Giảm chiều dữ liệu

#### Bước 4: Modeling
- So sánh 6 thuật toán Sentiment Analysis
- So sánh 3 thuật toán Clustering
- Chọn mô hình tối ưu

#### Bước 5: Evaluation
- Đánh giá hiệu suất model
- Phân tích lỗi chi tiết
- Kiểm tra độ tin cậy

#### Bước 6: Deployment
- Triển khai production
- Giám sát hiệu suất
- Bảo trì hệ thống

**Cách sử dụng:**
- Chọn "🛠️ Xây dựng mô hình"
- Khám phá từng tab theo thứ tự
- Xem biểu đồ và bảng phân tích

### 📊 3. PHÂN TÍCH CÔNG TY

**Mục đích:** Ứng dụng mô hình để phân tích thực tế

#### 3.1 Tổng quan công ty

**Chức năng:**
- Chọn công ty từ danh sách
- Xem thông tin cụm (cluster)
- Phân tích phân bố cảm xúc
- WordCloud cho từng loại cảm xúc

**Cách sử dụng:**
1. Chọn tab "🏢 Tổng quan công ty"
2. Chọn công ty từ dropdown
3. Xem thông tin chi tiết:
   - Cluster và đặc điểm
   - Biểu đồ cảm xúc tương tác
   - WordCloud tích cực/tiêu cực
   - Bảng review chi tiết

**Tính năng tương tác:**
- Hover vào biểu đồ để xem chi tiết
- Click vào các phần tử để zoom
- Scroll để xem WordCloud rõ hơn

#### 3.2 Phân tích cảm xúc

**Chức năng:**
- Dự đoán cảm xúc từ text
- Phân tích batch từ file Excel
- Đánh giá recommendation

**Cách sử dụng:**

**A. Dự đoán đơn lẻ:**
1. Nhập nội dung tích cực
2. Nhập góp ý cải thiện  
3. Nhấn "Dự đoán cảm xúc"
4. Xem kết quả và độ tin cậy

**B. Phân tích batch:**
1. Tải file Excel mẫu
2. Chuẩn bị file theo format:
   - Cột "What I liked"
   - Cột "Suggestions for improvement"
3. Upload file
4. Xem kết quả phân tích
5. Tải về file kết quả

**C. Đánh giá recommendation:**
1. Chọn mức rating (1-5)
2. Xem khuyến nghị dựa trên rating
3. Hiểu logic đưa ra khuyến nghị

## 🎯 CÁC TÍNH NĂNG NỔI BẬT

### ✨ Giao diện thân thiện
- Menu rõ ràng, dễ điều hướng
- Màu sắc phù hợp với loại cảm xúc
- Logo và thông tin nhóm phát triển

### 📊 Biểu đồ tương tác
- Hover để xem thông tin chi tiết
- Zoom và pan trên biểu đồ
- Màu sắc nhất quán: 
  * Xanh lá = Positive
  * Đỏ = Negative  
  * Vàng = Neutral

### 🌤️ WordCloud động
- Tự động tạo từ dữ liệu thực
- Màu sắc phù hợp với cảm xúc
- Kích thước từ thể hiện tần suất

### 📈 Phân tích đa chiều
- Sentiment Analysis với 87.2% accuracy
- Clustering với 5 nhóm rõ ràng
- Recommendation engine thông minh

## 🔧 XỬ LÝ SỰ CỐ

### ❌ Lỗi thường gặp:

**1. Lỗi import thư viện:**
```
ImportError: No module named 'streamlit'
```
**Giải quyết:** Chạy lại `pip install -r requirements.txt`

**2. Lỗi không tìm thấy file:**
```
FileNotFoundError: No such file or directory: 'data/Reviews.xlsx'
```
**Giải quyết:** Kiểm tra cấu trúc thư mục, đảm bảo tất cả file ở đúng vị trí

**3. Lỗi model không load được:**
```
Error loading model components
```
**Giải quyết:** Kiểm tra thư mục models/ có đầy đủ file .pkl

**4. WordCloud không hiển thị:**
- Kiểm tra có dữ liệu text không
- Đảm bảo công ty được chọn có review

**5. Biểu đồ không tương tác:**
- Đảm bảo plotly đã được cài đặt
- Thử refresh trang

### 🛠️ Debug tips:
- Mở Developer Tools (F12) để xem lỗi JavaScript
- Kiểm tra terminal/command prompt cho lỗi Python
- Thử khởi động lại ứng dụng: Ctrl+C rồi chạy lại

## 📞 HỖ TRỢ

### 👥 Thông tin liên hệ:
- **Giảng viên hướng dẫn:** Ms. Khuất Thùy Phương
- **Học viên thực hiện:**
  * Ms. Giang Phi Yến - Email: yengp96@gmail.com
  * Ms. Nguyễn Ngọc Khánh Linh - Email: nnkl1517000@gmail.com

### 🎓 Lớp: DL07_K304 - Data Science & Machine Learning

### 📋 Yêu cầu hỗ trợ:
Khi gặp vấn đề, vui lòng cung cấp:
1. Mô tả chi tiết lỗi
2. Screenshot màn hình  
3. Thông tin hệ thống (Windows/Mac, Python version)
4. File log lỗi (nếu có)

## 🚀 TÍNH NĂNG NÂNG CAO

### 📊 Xuất báo cáo
- Tải về kết quả phân tích Excel
- Export biểu đồ dạng PNG
- Báo cáo tổng hợp PDF (tính năng tương lai)

### 🔄 Cập nhật model
- Model có thể được cập nhật với dữ liệu mới
- Tự động backup phiên bản cũ
- A/B testing cho model mới

### 🌐 Triển khai web
- Có thể deploy lên Heroku
- Chia sẻ link cho người khác sử dụng
- Tích hợp API cho hệ thống khác

## 📝 CHANGELOG

### Version 1.0 (Current)
- ✅ Sentiment Analysis với XGBoost (87.2% accuracy)
- ✅ Clustering với K-Means (5 clusters)
- ✅ Giao diện Streamlit đầy đủ
- ✅ WordCloud tương tác
- ✅ Biểu đồ Plotly interactive
- ✅ Upload/Download Excel
- ✅ Recommendation engine

### Planned features:
- 🔮 Real-time data update
- 🔮 Advanced visualization
- 🔮 Mobile responsive
- 🔮 Multi-language support

## 🎯 TIPS SỬ DỤNG HIỆU QUẢ

### 💡 Cho người dùng cuối:
1. **Khám phá từ từ:** Bắt đầu với "Phân tích nghiệp vụ" để hiểu bối cảnh
2. **Tương tác với biểu đồ:** Hover và click để xem thông tin chi tiết
3. **Thử nghiệm:** Upload file Excel nhỏ trước để test
4. **So sánh công ty:** Chọn nhiều công ty khác nhau để so sánh

### 🔬 Cho researcher/analyst:
1. **Study methodology:** Xem chi tiết "Xây dựng mô hình" 
2. **Validation:** Kiểm tra phần Evaluation để hiểu độ tin cậy
3. **Export data:** Sử dụng chức năng xuất Excel cho phân tích sâu
4. **Cross-validation:** So sánh kết quả với tools khác

### 🏢 Cho HR/Company:
1. **Monitor competitors:** Xem cluster của các công ty đối thủ
2. **Track sentiment:** Theo dõi xu hướng cảm xúc
3. **Action planning:** Dựa vào recommendation để cải thiện
4. **Regular review:** Sử dụng định kỳ để đánh giá thay đổi

---

🎉 **Chúc bạn sử dụng ứng dụng hiệu quả!**

*Để cập nhật mới nhất, vui lòng theo dõi repository hoặc liên hệ team phát triển.*
