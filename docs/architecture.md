# Kiến trúc Code Map

> Tham chiếu chéo với README §5 (Kiến trúc hệ thống) và §6 (Lựa chọn mô hình).

## Cây thư mục

```
Vision_Language_Model_For_Visually_Impaired/
├── README.md                              # Đề cương chính
├── LICENSE                                # MIT + disclaimer accessibility
├── requirements.txt                       # Runtime deps
├── requirements-dev.txt                   # +eval, +lint, +test
├── pyproject.toml                         # Project config
├── .gitignore                             # Exclude models/datasets
├── .env.example                           # Template ENV vars
│
├── configs/
│   ├── default.yaml                       # App + pipeline + hardware
│   ├── models.yaml                        # Model paths & hyperparams
│   ├── thresholds.yaml                    # Calibration thresholds
│   └── prompts/
│       ├── vlm_system_prompt.txt          # System prompt for VLM
│       └── vi_formatter_templates.yaml    # VI natural language templates
│
├── src/vision_assistant/
│   ├── __init__.py
│   ├── main.py                            # CLI entrypoint
│   ├── pipeline.py                        # Triple-Layer orchestrator
│   ├── schemas.py                         # Shared dataclasses
│   │
│   ├── layer1_input/                      # § 5.1
│   │   ├── blur_detector.py               # Laplacian variance
│   │   ├── exposure_check.py              # Luminance histogram
│   │   ├── scene_change.py                # SSIM comparison
│   │   └── frame_filter.py                # Composite
│   │
│   ├── layer2_vlm/                        # § 5.2
│   │   ├── vlm_engine.py                  # Moondream / Qwen2-VL adapters
│   │   ├── confidence_estimator.py        # 3-signal combiner ★ contribution
│   │   ├── reacquisition.py               # Camera-action hint
│   │   └── prompts.py                     # Prompt loader
│   │
│   ├── layer3_output/                     # § 5.3
│   │   ├── spatial_grounding.py           # VLM + YOLO merge
│   │   ├── priority_filter.py             # Safety > Goal > Context
│   │   ├── translator.py                  # EN → VI fallback
│   │   └── vi_formatter.py                # Template-based VI speech
│   │
│   ├── perception/
│   │   └── yolo_detector.py               # YOLOv8n via Ultralytics
│   │
│   ├── audio/
│   │   ├── stt_vosk.py                    # Vietnamese STT
│   │   └── tts_piper.py                   # Vietnamese TTS streaming
│   │
│   ├── camera/
│   │   └── capture.py                     # OpenCV camera stream
│   │
│   └── utils/
│       ├── config_loader.py               # YAML + ENV merge
│       ├── logger.py                      # Loguru setup
│       └── metrics.py                     # Latency + RAM monitor
│
├── eval/
│   ├── run_benchmark.py                   # Latency/RAM benchmark
│   └── (TODO Tuần 15) demo_analysis.py    # Phân loại hit / near-miss / critical
│
├── scripts/
│   ├── download_models.py                 # One-shot downloader
│   └── run_demo.py                        # End-to-end single-shot demo
│
├── tests/
│   ├── conftest.py
│   ├── test_layer1_input_filter.py
│   └── test_confidence_estimator.py
│
├── docs/
│   ├── setup_windows.md                   # Windows 8GB setup guide
│   └── architecture.md                    # (this file)
│
└── data/                                  # ⚠ NOT committed
    ├── models/                            # GGUF, ONNX, PT weights
    ├── datasets/                          # VQAv2, VizWiz, COCO, POPE
    └── outputs/                           # Logs, predictions
```

## Sự phụ thuộc giữa các module

```
                                  pipeline.py
                                       │
        ┌───────────────────┬──────────┼──────────┬───────────────────┐
        ▼                   ▼          ▼          ▼                   ▼
  layer1_input/      layer2_vlm/   perception/  layer3_output/   audio/
        │                   │          │              │            │
        └─→ schemas ◄───────┴──────────┴──────────────┴────────────┘
                                       │
                                       ▼
                                  utils/ (config, logger, metrics)
```

- Layer trên KHÔNG được import từ layer dưới (đảm bảo có thể test/ablation độc lập).
- Tất cả I/O dataclass nằm trong `schemas.py` để tránh circular import.
- Config flow: `configs/*.yaml` + `.env` → `utils.config_loader` → factory → modules.

## Ablation mapping

Cấu hình ablation (README §7.5) bật/tắt từng lớp:

| Config | Layer1 | Layer2 verifier | Layer3 grounding |
|---|---|---|---|
| **Baseline** | ✗ | ✗ | ✗ (raw VLM only) |
| **+L1** | ✓ | ✗ | ✗ |
| **+L1+L2** | ✓ | ✓ | ✗ |
| **Full** | ✓ | ✓ | ✓ |

Bật/tắt qua flag `--ablation {baseline,l1,l1l2,full}` trong CLI demo (TODO Week 14).
