import type { Citation, Critique, Finding, SubTask } from "@/types";

export const PRESETS = [
  "So sánh chiến lược AI giữa VNG và FPT trong 2024",
  "Phân tích thị trường EV Việt Nam 6 tháng đầu 2025",
  "How does Anthropic's Constitutional AI compare to RLHF?",
  "Tác động của Bitcoin halving 2024 đến thị trường",
  "Compare Apple Vision Pro vs Meta Quest 3 ecosystem",
];

export interface HistoryItem {
  id: string;
  query: string;
  time: string;
  iters: number;
  score: number;
}

export const SESSION_HISTORY: HistoryItem[] = [
  { id: "s1", query: "So sánh chiến lược AI giữa VNG và FPT trong 2024", time: "10 phút trước", iters: 2, score: 0.88 },
  { id: "s2", query: "Open source LLMs comparison Q4 2024", time: "2 giờ trước", iters: 3, score: 0.79 },
  { id: "s3", query: "Phân tích tác động lạm phát đến PMI Việt Nam", time: "Hôm qua", iters: 1, score: 0.91 },
  { id: "s4", query: "EV market trends Southeast Asia 2025", time: "2 ngày trước", iters: 2, score: 0.85 },
];

export const TASKS_BASE: SubTask[] = [
  { id: "t1", question: "Sản phẩm AI và đầu tư R&D của VNG trong 2024", rationale: "Cần baseline về danh mục sản phẩm AI hiện hữu (Zalo AI, ZaloPay fraud, ZingMP3 recsys) và quy mô đầu tư.", dependencies: [] },
  { id: "t2", question: "Đối tác chiến lược và memo nội bộ AI của VNG", rationale: "Xác định liên minh hạ tầng (cloud/GPU) và định hướng dài hạn từ leadership.", dependencies: ["t1"] },
  { id: "t3", question: "Lộ trình AI và tăng trưởng FPT.AI 2024", rationale: "Hiểu trục enterprise của FPT, đặc biệt FPT AI Factory hợp tác NVIDIA.", dependencies: [] },
  { id: "t4", question: "Doanh thu AI và các deal enterprise của FPT 2024", rationale: "Định lượng kết quả thương mại — số liệu doanh thu, hợp đồng Nhật/Mỹ.", dependencies: ["t3"] },
];

export const FINDINGS_BASE: Record<string, Finding> = {
  t1: {
    task_id: "t1",
    content: "",
    confidence: 0.85,
    tool_calls: 8,
    sources: [
      "https://vneconomy.vn/vng-cong-bo-chien-luoc-ai-2024.htm",
      "https://www.vng.com.vn/vi/news/ai-products-2024",
      "https://vnexpress.net/zalo-ai-100-trieu-nguoi-dung-4759123.html",
      "https://techcrunch.com/2024/southeast-asia-ai-vietnam",
    ],
    tools: { web_search: 4, fetch_url: 3, vector_search: 1, python_exec: 0 },
    excerpt: "VNG công bố chiến lược AI-first vào Q2/2024 với ba mảng ưu tiên: Zalo AI Assistant (12M MAU vào 11/2024), ZaloPay fraud detection (giảm 47% giả mạo), và recommendation engine ZingMP3. Đầu tư R&D AI ước ~800 tỷ VND năm 2024, phần lớn vào hạ tầng GPU on-shore.",
  },
  t2: {
    task_id: "t2",
    content: "",
    confidence: 0.6,
    tool_calls: 10,
    sources: ["https://vng.com.vn/blog/strategy-2024", "https://nikkei.com/article/vng-cloud-2024"],
    tools: { web_search: 5, fetch_url: 2, vector_search: 3, python_exec: 0 },
    excerpt: "VNG ký MOU với một số nhà cung cấp cloud nội địa nhưng chi tiết tài chính không công bố. Memo từ CEO Lê Hồng Minh nhấn mạnh \"AI là defensibility layer, không phải product line riêng\".",
  },
  t3: {
    task_id: "t3",
    content: "",
    confidence: 0.9,
    tool_calls: 14,
    sources: [
      "https://fpt.com.vn/vi/tin-tuc/cong-nghe/fpt-ai-2024",
      "https://nvidianews.nvidia.com/fpt-ai-factory",
      "https://vneconomy.vn/fpt-ai-doanh-thu-tang-truong",
      "https://nikkei.com/article/fpt-ai-japan-expansion",
      "https://reuters.com/technology/fpt-vietnam-ai-2024",
    ],
    tools: { web_search: 6, fetch_url: 5, vector_search: 2, python_exec: 1 },
    excerpt: "FPT AI Factory hợp tác NVIDIA, cam kết triển khai 7,000+ GPU H100 giai đoạn 2024–2025 — cụm AI thương mại lớn nhất Đông Nam Á. Lộ trình 2024 tập trung 3 trục: AI services, AI infrastructure-as-a-service, và AI Engineer training (5,000 kỹ sư đến 2026).",
  },
  t4: {
    task_id: "t4",
    content: "",
    confidence: 0.45,
    tool_calls: 5,
    sources: ["https://fpt.com.vn/quan-he-co-dong/bao-cao-tai-chinh"],
    tools: { web_search: 4, fetch_url: 1, vector_search: 0, python_exec: 0 },
    excerpt: "Tìm thấy báo cáo tài chính tổng hợp nhưng không tách biệt rõ doanh thu mảng AI. Tool fetch_url thất bại 2 lần với trang quan hệ cổ đông. Cần thêm vòng research.",
  },
};

export const FINDINGS_ITER2: Record<string, Finding> = {
  t4: {
    task_id: "t4",
    content: "",
    confidence: 0.88,
    tool_calls: 14,
    sources: [
      "https://fpt.com.vn/quan-he-co-dong/bao-cao-tai-chinh",
      "https://vneconomy.vn/fpt-ai-doanh-thu-145-trieu-usd",
      "https://nikkei.com/article/fpt-japan-23-deals-2024",
      "https://dantri.com.vn/fpt-ai-tang-truong-67-phan-tram",
    ],
    tools: { web_search: 7, fetch_url: 4, vector_search: 1, python_exec: 2 },
    excerpt: "FPT.AI đạt doanh thu ~145M USD trong 2024 (+67% YoY). Ký 23 hợp đồng AI enterprise tại Nhật, tổng giá trị >90M USD. Mảng dịch vụ AI cho khách hàng Mỹ tăng 2.3x.",
  },
};

export const CRITIC_ITER1: Critique = {
  is_complete: false,
  quality_score: 0.72,
  threshold: 0.85,
  scores: { completeness: 0.62, evidence: 0.7, depth: 0.78, accuracy: 0.82, structure: 0.7 },
  missing_info: [
    "Doanh thu cụ thể từ AI products của FPT (task t4 confidence chỉ 0.45)",
    "Số liệu so sánh ngân sách R&D định lượng giữa hai bên",
    "VNG B2C vs FPT B2B chưa được structure rõ",
  ],
  suggestions: [
    "Replan: thêm sub-task chuyên biệt về tài chính FPT.AI",
    "Tăng confidence threshold cho synthesis step",
    "Cross-reference với ít nhất 3 nguồn cho mỗi claim định lượng",
  ],
};

export const CRITIC_ITER2: Critique = {
  is_complete: true,
  quality_score: 0.89,
  threshold: 0.75,
  scores: { completeness: 0.9, evidence: 0.92, depth: 0.85, accuracy: 0.92, structure: 0.86 },
  missing_info: [],
  suggestions: [],
};

export const FINAL_REPORT = `# So sánh chiến lược AI: VNG vs FPT trong năm 2024

## Tổng quan

Năm 2024 đánh dấu bước ngoặt trong chiến lược AI của hai tập đoàn công nghệ hàng đầu Việt Nam. VNG tập trung **nhúng AI sâu vào hệ sinh thái sản phẩm tiêu dùng nội địa** [1], trong khi FPT theo đuổi mô hình **mở rộng dịch vụ AI cho khách hàng doanh nghiệp toàn cầu** [4][6]. Hai hướng đi phản ánh hai DNA khác biệt và sẽ tạo ra hai loại risk profile rất khác nhau cho nhà đầu tư.

## VNG: Tập trung sản phẩm và hạ tầng nội địa

VNG công bố chiến lược "AI-first" vào Q2/2024, ưu tiên ba mảng: trợ lý ảo Zalo AI, công cụ chống gian lận cho ZaloPay, và recommendation engine cho ZingMP3 [1][2]. Theo các nguồn nội bộ, công ty đầu tư khoảng **800 tỷ VND cho R&D AI** trong năm, tập trung phần lớn vào hạ tầng GPU đặt tại Việt Nam [3].

Zalo AI Assistant đạt **12 triệu MAU** vào tháng 11/2024, tăng 4x so với đầu năm [3]. Tuy nhiên, VNG không công bố doanh thu cụ thể từ các sản phẩm AI — định hướng "AI là defensibility layer, không phải product line" của CEO Lê Hồng Minh cho thấy công ty muốn dùng AI để khóa user thay vì monetize trực tiếp.

## FPT: Mở rộng enterprise AI quốc tế

FPT đi hướng ngược lại: bán dịch vụ AI cho doanh nghiệp, đặc biệt tại Nhật Bản và Hoa Kỳ [4][6]. **FPT.AI đạt doanh thu khoảng 145 triệu USD trong 2024**, tăng 67% so với 2023 [5]. Nikkei Asia ghi nhận FPT đã ký kết **23 hợp đồng AI quy mô doanh nghiệp tại Nhật**, tổng giá trị vượt 90 triệu USD [6].

Đáng chú ý nhất là **FPT AI Factory** hợp tác với NVIDIA — cam kết triển khai hơn 7,000 GPU H100 trong giai đoạn 2024–2025 [4]. Đây là cụm tính toán AI thương mại lớn nhất Đông Nam Á, cho phép FPT vừa phục vụ khách hàng vừa cho thuê compute.

## So sánh điểm mấu chốt

- **Mô hình kinh doanh**: VNG B2C, FPT B2B. VNG monetize gián tiếp qua engagement; FPT monetize trực tiếp qua hợp đồng dịch vụ.
- **Thị trường mục tiêu**: VNG ưu tiên 100 triệu user Việt; FPT nhắm enterprise toàn cầu, đặc biệt trục Nhật–Mỹ [6][7].
- **Đầu tư hạ tầng**: Cả hai đều đầu tư GPU on-shore, nhưng quy mô FPT lớn hơn đáng kể (7,000+ vs ước tính ~500 GPU cho VNG) [4].
- **Nhân sự**: FPT công bố tuyển 5,000 kỹ sư AI đến 2026; VNG tập trung tuyển senior researchers số lượng nhỏ hơn [8].
- **Khả năng đo ROI**: FPT minh bạch hơn nhiều — VNG gần như không tách doanh thu AI ra báo cáo riêng.

## Rủi ro & triển vọng

VNG đối mặt rủi ro **không chứng minh được ROI rõ ràng** trong ngắn hạn, đặc biệt nếu áp lực IPO quay lại. Lợi thế của họ là kho dữ liệu hành vi 100M+ user — nếu khai thác đúng, sẽ tạo defensibility khó copy. FPT thì rủi ro **phụ thuộc khách hàng Nhật** (concentration risk), nhưng có dòng tiền ổn định và AI Factory đã đặt nền cho mảng compute-as-a-service dài hạn [7].

## Kết luận

Hai chiến lược không loại trừ nhau — VNG vẫn là công ty internet tiêu dùng tăng cường bằng AI, trong khi FPT đang tự định vị thành nhà cung cấp dịch vụ AI cấp doanh nghiệp khu vực. Đối với nhà đầu tư, FPT có khả năng hiển thị doanh thu AI rõ ràng hơn trong ngắn hạn, trong khi VNG hứa hẹn defensibility cao hơn nếu thành công khóa được người dùng [7][8].`;

export const CITATIONS: Citation[] = [
  { n: 1, url: "https://vneconomy.vn/vng-cong-bo-chien-luoc-ai-2024.htm", title: "VNG công bố chiến lược AI-first 2024", domain: "vneconomy.vn" },
  { n: 2, url: "https://www.vng.com.vn/vi/news/ai-products-2024", title: "VNG AI Products Roadmap 2024", domain: "vng.com.vn" },
  { n: 3, url: "https://vnexpress.net/zalo-ai-100-trieu-nguoi-dung-4759123.html", title: "Zalo AI cán mốc 12 triệu người dùng hoạt động", domain: "vnexpress.net" },
  { n: 4, url: "https://nvidianews.nvidia.com/fpt-ai-factory", title: "FPT AI Factory: 7,000 H100 GPUs Partnership with NVIDIA", domain: "nvidianews.nvidia.com" },
  { n: 5, url: "https://vneconomy.vn/fpt-ai-doanh-thu-145-trieu-usd", title: "FPT.AI doanh thu 145 triệu USD năm 2024", domain: "vneconomy.vn" },
  { n: 6, url: "https://nikkei.com/article/fpt-japan-23-deals-2024", title: "FPT signs 23 enterprise AI deals in Japan worth $90M+", domain: "nikkei.com" },
  { n: 7, url: "https://reuters.com/technology/fpt-vietnam-ai-2024", title: "Vietnam AI race heats up: FPT vs VNG strategies diverge", domain: "reuters.com" },
  { n: 8, url: "https://dantri.com.vn/fpt-ai-tang-truong-67-phan-tram", title: "FPT.AI tăng trưởng 67% năm 2024, tuyển 5,000 kỹ sư", domain: "dantri.com.vn" },
];
