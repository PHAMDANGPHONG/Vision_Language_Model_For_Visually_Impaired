# Setup trên Windows 10/11 (8GB RAM, Intel Iris Xe, no CUDA)

## 1. Cài Python 3.10 hoặc 3.11
Tải từ [python.org](https://www.python.org/downloads/windows/). Khi cài bật **"Add Python to PATH"**.

```powershell
python --version   # >= 3.10, < 3.12
```

## 2. Tạo virtual environment
```powershell
cd D:\Github_Project\Vision_Language_Model_For_Visually_Impaired
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

## 3. Cài đặt `llama-cpp-python` (CPU build)

Bản pre-built thường đủ với CPU thông thường:
```powershell
pip install llama-cpp-python>=0.2.90
```

### Tùy chọn: Vulkan backend cho Intel Iris Xe
Yêu cầu **MSVC Build Tools + Vulkan SDK** đã cài đặt. Sau đó build từ source:
```powershell
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
pip install llama-cpp-python --no-binary llama-cpp-python --force-reinstall --upgrade
```
> Lợi ích trên Iris Xe khá nhỏ (~10-25%). Chỉ thử khi đã có baseline CPU.

## 4. Cài đặt dependencies
```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt   # nếu cần notebook + lint + test
```

## 5. PyAudio trên Windows (chỉ khi cần)
Nếu `pip install pyaudio` lỗi, dùng wheel:
```powershell
pip install pipwin
pipwin install pyaudio
```
Hoặc bỏ qua — `sounddevice` đã đủ cho dự án này.

## 6. Tải mô hình
```powershell
python scripts/download_models.py --all
```
Cần khoảng **~2.5 GB** ổ cứng cho Moondream + YOLO + Vosk + Piper.

## 7. Kiểm tra hệ thống
```powershell
python -m vision_assistant.main doctor
```
Output kỳ vọng:
```
  python                   3.11.x
  os                       Windows 10
  cpu_count_physical       4 hoặc 6
  cpu_count_logical        8 hoặc 12
  ram_total_gb             8.0
  ram_available_gb         3-5
  process_rss_mb           <300
```

## 8. Chạy demo
```powershell
python scripts/run_demo.py --query "Mô tả những gì bạn nhìn thấy."
```

## 9. Chạy unit tests
```powershell
pytest -v
```

---

## Khắc phục sự cố

| Vấn đề | Giải pháp |
|---|---|
| `ImportError: DLL load failed` khi import OpenCV | Cài [Visual C++ Redistributable 2015-2022](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `llama-cpp-python` chậm hơn dự kiến | Đặt `n_threads = số core vật lý` (không phải logical). Tắt browser/IDE khi inference. |
| Camera không mở | Mặc định dùng `CAP_DSHOW`; thử đổi `CAMERA_INDEX=1` trong `.env` |
| Vosk yêu cầu microphone | Kiểm tra Privacy Settings của Windows → Microphone → Allow desktop apps |
| Piper TTS không có giọng tiếng Việt | Tải thủ công từ [piper voices](https://github.com/rhasspy/piper/blob/master/VOICES.md), đặt vào `data/models/` |
| RAM peak > 7GB | Đóng tab Chrome, giảm `n_ctx` xuống 1024, set `LLAMA_N_BATCH=128` |
