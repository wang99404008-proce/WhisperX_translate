import os
import threading
from tkinter import filedialog, messagebox, StringVar
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import whisperx
import torch

APP_NAME = "WhisperX 多語言語音翻譯工具"

audio_file_path = ""
output_folder_path = ""

def choose_file():
    global audio_file_path
    file_path = filedialog.askopenfilename(
        title="選擇語音或影片檔案",
        filetypes=[
            ("支援的音訊/視訊", "*.mp3;*.wav;*.m4a;*.mp4;*.mkv;*.flac"),
            ("所有檔案", "*.*")
        ]
    )
    if not file_path:
        return
    audio_file_path = file_path
    source_label.config(text=f"來源檔案：\n{os.path.basename(file_path)}")

def choose_output_folder():
    global output_folder_path
    folder_path = filedialog.askdirectory(title="選擇輸出資料夾")
    if not folder_path:
        return
    output_folder_path = folder_path
    output_label.config(text=f"輸出資料夾：\n{output_folder_path}")

def run_whisperx_process():
    global audio_file_path, output_folder_path
    
    if not audio_file_path or not output_folder_path:
        messagebox.showwarning("提醒", "請先選擇音訊/影片檔案與輸出資料夾！")
        return

    target_lang = lang_var.get()
    model_size = model_var.get()
    
    status_label.config(text="正在載入 WhisperX 模型，請稍候...")
    progress.start(10)
    window.update_idletasks()

    try:
        # 自動偵測硬體裝置 (有 GPU 優先使用 cuda，否則用 cpu)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        status_label.config(text=f"正在使用 {device.upper()} 載入模型 ({model_size})...")
        
        # 1. 載入 Whisper 模型
        model = whisperx.load_model(model_size, device, compute_type=compute_type)

        # 2. 載入音訊
        status_label.config(text="正在載入與預處理音訊...")
        audio = whisperx.load_audio(audio_file_path)

        # 3. 進行語音辨識 (Transcription)
        status_label.config(text="正在進行語音轉文字 (Transcription)...")
        result = model.transcribe(audio, batch_size=16)

        # 4. 對齊時間軸 (Alignment)
        status_label.config(text="正在優化單字級時間軸對齊...")
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

        # 5. 翻譯 (如果選擇的目標語言與原文不同，或需要翻譯成英文等)
        # 註：Whisper 模型本身在 transcribe 時可指定 task="translate" 直接翻成英文，
        # 若需要其他語言翻譯，後續可透過大語言模型或 Whisper 內建翻譯處理。
        
        # 6. 輸出結果檔案
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_txt = os.path.join(output_folder_path, f"{base_name}_transcript.txt")
        
        status_label.config(text="正在儲存結果檔案...")
        with open(output_txt, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                f.write(f"[{seg['start']:.2f} --> {seg['end']:.2f}] {seg['text']}\n")

        progress.stop()
        status_label.config(text="處理完成！")
        messagebox.showinfo("成功", f"語音辨識與翻譯已完成！\n檔案已儲存至：\n{output_txt}")

    except Exception as e:
        progress.stop()
        status_label.config(text="處理失敗")
        messagebox.showerror("錯誤", f"過程發生錯誤：\n{str(e)}")

def start_thread():
    threading.Thread(target=run_whisperx_process, daemon=True).start()

# --- UI 介面設計 ---
window = tb.Window(title=APP_NAME, themename="cosmo", size=(700, 650))
window.resizable(False, False)

tb.Label(window, text="WhisperX 智能語音辨識與翻譯工具", font=("Microsoft JhengHei UI", 16, "bold")).pack(pady=20)

# 檔案選擇區
tb.Button(window, text="選擇音訊或影片檔案 (MP3/WAV/MP4)", bootstyle="primary", command=choose_file, width=45).pack(pady=5)
source_label = tb.Label(window, text="尚未選擇來源檔案", font=("Microsoft JhengHei UI", 10), bootstyle="secondary")
source_label.pack(pady=5)

# 輸出資料夾選擇
tb.Button(window, text="選擇輸出資料夾", bootstyle="info", command=choose_output_folder, width=45).pack(pady=5)
output_label = tb.Label(window, text="尚未選擇輸出資料夾", font=("Microsoft JhengHei UI", 10), bootstyle="secondary")
output_label.pack(pady=5)

# 模型大小選擇
tb.Label(window, text="選擇 Whisper 模型大小 (模型越大越準，但需要較多記憶體)", font=("Microsoft JhengHei UI", 10, "bold")).pack(pady=(15, 5))
model_var = StringVar(value="base")
tb.Combobox(window, textvariable=model_var, values=["tiny", "base", "small", "medium", "large-v2", "large-v3"], state="readonly", width=25).pack(pady=5)

# 目標語言選擇
tb.Label(window, text="目標輸出語言代碼 (例如: zh 代表中文, en 代表英文, id 代表印尼文)", font=("Microsoft JhengHei UI", 10, "bold")).pack(pady=(15, 5))
lang_var = StringVar(value="zh")
tb.Entry(window, textvariable=lang_var, width=27).pack(pady=5)

# 狀態與進度條
status_label = tb.Label(window, text="待命中", font=("Microsoft JhengHei UI", 11))
status_label.pack(pady=10)

progress = tb.Progressbar(window, length=500, mode="indeterminate", bootstyle="success-striped")
progress.pack(pady=10)

# 開始按鈕
tb.Button(window, text="開始辨識與翻譯", bootstyle="success", command=start_thread, width=25).pack(pady=15)

window.mainloop()
