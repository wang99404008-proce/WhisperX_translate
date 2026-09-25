import os
# 【關鍵設定】強制關閉 Hugging Face 符號連結，徹底解決 Windows [WinError 1314] 用戶端沒有這項特殊權限問題
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
import threading
import torch
from faster_whisper import WhisperModel
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, StringVar

APP_NAME = "Whisper 離線影音智慧辨識與時間碼工具"

audio_file_path = ""
output_folder_path = ""

def format_timecode(seconds):
    """將秒數轉換為清晰的時間碼格式 (HH:MM:SS)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_model_path(model_size):
    """優先讀取 .exe 旁 models 資料夾中的離線模型，實現 100% 斷網運行"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 對應 faster-whisper 的快取資料夾命名規則
    folder_name = f"models--Systran--faster-whisper-{model_size}"
    local_model_dir = os.path.join(base_dir, "models", folder_name)
    
    # 如果本機有離線模型資料夾，直接回傳該路徑
    if os.path.exists(local_model_dir):
        return local_model_dir
    
    # 若無則回傳預設名稱（在連網電腦上會自動下載，在無網電腦若無 models 會提示）
    return model_size

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

def run_process():
    global audio_file_path, output_folder_path
    
    if not audio_file_path or not output_folder_path:
        messagebox.showwarning("提醒", "請先選擇音訊/影片檔案與輸出資料夾！")
        return

    model_size = model_var.get()
    
    status_label.config(text="正在載入語音辨識模型，請稍候...")
    progress.start(10)
    window.update_idletasks()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        # 取得模型路徑（自動判斷是否使用離線本機模型）
        model_path_or_name = get_model_path(model_size)
        status_label.config(text=f"正在載入模型中...")
        
        model = WhisperModel(model_path_or_name, device=device, compute_type=compute_type)

        status_label.config(text="正在進行語音轉文字與時間碼對齊...")
        segments, info = model.transcribe(audio_file_path, beam_size=5)

        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_txt = os.path.join(output_folder_path, f"{base_name}_transcript.txt")
        
        status_label.config(text="正在產生帶時間碼的文字檔...")
        with open(output_txt, "w", encoding="utf-8") as f:
            for segment in segments:
                start_str = format_timecode(segment.start)
                end_str = format_timecode(segment.end)
                text = segment.text.strip()
                f.write(f"[{start_str} --> {end_str}] {text}\n")

        progress.stop()
        status_label.config(text="處理完成！")
        messagebox.showinfo("成功", f"語音辨識完成！\n帶時間碼的 TXT 檔案已儲存至：\n{output_txt}")

    except Exception as e:
        progress.stop()
        status_label.config(text="處理失敗")
        messagebox.showerror("錯誤", f"過程發生錯誤（若在無網環境，請確認 models 資料夾是否完整）：\n{str(e)}")

def start_thread():
    threading.Thread(target=run_process, daemon=True).start()

# --- UI 介面設計 ---
window = tb.Window(title=APP_NAME, themename="cosmo", size=(700, 620))
window.resizable(False, False)

tb.Label(window, text="Whisper 離線影音智慧辨識工具", font=("Microsoft JhengHei UI", 15, "bold")).pack(pady=20)

tb.Button(window, text="選擇音訊或影片檔案 (MP3/WAV/MP4)", bootstyle="primary", command=choose_file, width=45).pack(pady=5)
source_label = tb.Label(window, text="尚未選擇來源檔案", font=("Microsoft JhengHei UI", 10), bootstyle="secondary")
source_label.pack(pady=5)

tb.Button(window, text="選擇輸出資料夾", bootstyle="info", command=choose_output_folder, width=45).pack(pady=5)
output_label = tb.Label(window, text="尚未選擇輸出資料夾", font=("Microsoft JhengHei UI", 10), bootstyle="secondary")
output_label.pack(pady=5)

tb.Label(window, text="選擇模型大小 (需與 models 內的快取資料夾名稱相符)", font=("Microsoft JhengHei UI", 10, "bold")).pack(pady=(15, 5))
model_var = StringVar(value="base")
tb.Combobox(window, textvariable=model_var, values=["tiny", "base", "small", "medium", "large-v3"], state="readonly", width=25).pack(pady=5)

status_label = tb.Label(window, text="待命中", font=("Microsoft JhengHei UI", 11))
status_label.pack(pady=10)

progress = tb.Progressbar(window, length=500, mode="indeterminate", bootstyle="success-striped")
progress.pack(pady=10)

tb.Button(window, text="開始辨識並產出帶時間碼 TXT", bootstyle="success", command=start_thread, width=30).pack(pady=15)

window.mainloop()
