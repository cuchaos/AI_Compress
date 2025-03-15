專案目標  
===
本專案旨在 降低大語言模型（LLM）推理時的內存消耗，透過 小模型 (DistilBERT) 生成*高效指令向量*，並讓 大模型 (Mistral-7B) 解析這些指令來執行計算，從而顯著減少記憶體需求。
該方法不依賴傳統的*人類語言輸入*，而是利用 純數值向量（256 維）作為模型間的溝通橋樑，以最小化大模型的計算負擔。

主要特點  
===
* ✅ 指令壓縮傳遞：小模型生成 256 維向量指令，無需完整 Token 輸入，減少 KV Cache 負擔。  
* ⚠ 動態權重調整：大模型僅根據指令 更新 q_proj 的部分權重，而非全模型微調，內存消耗更低。  
* ✅ 支持 4-bit 量化：可搭配 bitsandbytes 進行 8-bit / 4-bit 量化，大幅降低權重佔用。
* ✅ Flash Attention 加速：啟用 FlashAttention，減少 Attention 計算內存。
* ✅ LoRA 適配：進一步減少內存需求，使 Mistral-7B 適合 12GB 內存運行。

| 方法 | 內存消耗 |
| :--: | :--: |
| 標準 Mistral-7B（FP16）  | 18GB+ |
| 本專案（未優化）  | 14-15GB |
| 本專案（4-bit 量化 + LoRA + Flash Attention）(now)  | 4-5GB |  

✨ 目標：可以在 RTX 4060 Ti（12GB）或更低 VRAM 的 GPU 上運行大模型！

 核心技術
 ===
🔹 256 維壓縮指令 → 小模型 (DistilBERT) 生成指令，大模型 (Mistral-7B) 解析並執行  
🔹 局部權重更新 → 只修改 q_proj（1024×1024 → 512×512），避免全模型調整  
🔹 低比特量化（4-bit） → bitsandbytes 降低權重內存占用  
🔹 FlashAttention → 減少 Attention 機制的內存負擔  
🔹 LoRA 適配 → 進一步減少計算需求  

🔗 結果
 ===
💡 因為小模型的存在反而讓內存占用增加，只使用LoRA和量化的壓縮表現就已經很好。
💡 小模型 (DistilBERT) 生成指令反而會增加記憶體的負擔。


硬體配置及環境
===
=== System Information ===
- OS: Windows 10 (10.0.19045)
- Machine: AMD64
- CPU Cores (Total): 16
- Total RAM: 31.10 GB
- CUDA Available: True
- CUDA Version: 12.4
- Device Count: 1
- Device 0: NVIDIA GeForce RTX 4070 Ti SUPER
  - Compute Capability: (8, 9)
  - Total Memory: 15.99 GB

=== Software Environment ===
- Python Version: 3.12.9
- PyTorch Version: 2.6.0+cu124
- Transformers Version: 4.49.0
- Bitsandbytes: 0.45.3
- Xformers: 0.0.29.post3
- Peft: 0.14.0
- Torchvision: 0.21.0+cu124
- Torchaudio: 2.6.0+cu124
- Nltk: 3.9.1
