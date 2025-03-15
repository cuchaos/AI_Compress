專案目標  
===
本專案旨在 降低大語言模型（LLM）推理時的內存消耗，透過 小模型 (DistilBERT) 生成*高效指令向量*，並讓 大模型 (Mistral-7B) 解析這些指令來執行計算，從而顯著減少記憶體需求。
該方法不依賴傳統的*人類語言輸入*，而是利用 純數值向量（256 維）作為模型間的溝通橋樑，以最小化大模型的計算負擔。

主要特點  
===
* ✅ 指令壓縮傳遞：小模型生成 256 維向量指令，無需完整 Token 輸入，減少 KV Cache 負擔。  
* ✅ 動態權重調整：大模型僅根據指令 更新 q_proj 的部分權重，而非全模型微調，內存消耗更低。  
* ❌ 支持 4-bit 量化：可搭配 bitsandbytes 進行 8-bit / 4-bit 量化，大幅降低權重佔用。 (todo)  
* ❌ Flash Attention 加速：啟用 FlashAttention，減少 Attention 計算內存。 (todo)  
* ❌ LoRA 適配：進一步減少內存需求，使 Mistral-7B 適合 12GB 內存運行。 (todo)

| 方法 | 內存消耗 |
| :--: | :--: |
| 標準 Mistral-7B（FP16）  | 18GB+ |
| 本專案（未優化）(now)  | 14-15GB |
| 本專案（4-bit 量化 + LoRA + Flash Attention）  | 10-12GB |  

✨ 目標：可以在 RTX 4060 Ti（12GB）或更低 VRAM 的 GPU 上運行大模型！

 核心技術
 ===
🔹 256 維壓縮指令 → 小模型 (DistilBERT) 生成指令，大模型 (Mistral-7B) 解析並執行  
🔹 局部權重更新 → 只修改 q_proj（1024×1024 → 512×512），避免全模型調整  
🔹 低比特量化（4-bit） → bitsandbytes 降低權重內存占用  
🔹 FlashAttention → 減少 Attention 機制的內存負擔  
🔹 LoRA 適配 → 進一步減少計算需求  

🔗 應用場景
 ===
💡 低內存推理加速：讓大模型適用於 12GB 內存的 GPU（如 RTX 4060 Ti、A6000 16GB）  
💡 邊緣 AI 設備適配：減少記憶體需求，適用於嵌入式 AI 設備  
💡 高效大模型壓縮：類似 LoRA / Prompt Tuning，但更極端，完全拋棄 Token-based 語言輸入  

硬體配置及環境
===
- PyTorch version: 2.5.1
- CUDA available: True
- CUDA version: 12.1
- Using device: cuda
- GPU : NVIDIA GeForce RTX 4070 Ti SUPER
- RAM : 32GB DDR5
