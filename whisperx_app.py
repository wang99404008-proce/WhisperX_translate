import os
import sys

# 【終極對策】直接強制把 Hugging Face 的快取資料夾指向我們 `.exe` 旁邊的 models 資料夾！
if getattr(sys, 'frozen', False):
    current_base_dir = os.path.dirname(sys.executable)
else:
    current_base_dir = os.path.dirname(os.path.abspath(__file__))

offline_cache_dir = os.path.join(current_base_dir, "models")
os.environ["HF_HOME"] = offline_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = offline_cache_dir
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFERS_OFFLINE"] = "1"

import sys
import threading
import torch
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar

APP_NAME = "Whisper 離線影音智慧辨識工具 (極速相容版)"

audio_file_path = ""
output_folder_path = ""

def format_timecode(seconds):
    """將秒數轉換為清晰的時間碼格式 (HH:MM:SS)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_model_path(model_size):
    """強制鎖定 .exe 旁絕對路徑的 models 資料夾"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 直接指向固定結構
    local_model_dir = os.path.join(base_dir, "models", f"models--Systran--faster-whisper-{model_size}")
    
    # 嚴格檢查，如果找不到直接跳出警告視窗，絕不聯網
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"找不到離線模型資料夾！\n請確認此路徑是否存在：\n{local_model_dir}")
    
    return local_model_dir
    
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
    source_label.config(text=f"來源檔案：{os.path.basename(file_path)}")

def choose_output_folder():
    global output_folder_path
    folder_path = filedialog.askdirectory(title="選擇輸出資料夾")
    if not folder_path:
        return
    output_folder_path = folder_path
    output_label.config(text=f"輸出資料夾：{output_folder_path}")

def run_process():
    global audio_file_path, output_folder_path
    
    if not audio_file_path or not output_folder_path:
        messagebox.showwarning("提醒", "請先選擇音訊/影片檔案與輸出資料夾！")
        return

    model_size = model_var.get()
    selected_lang = lang_var.get().strip()
    language_param = None if selected_lang in ["auto", "自動偵測", ""] else selected_lang

    status_label.config(text="正在載入語音辨識模型，請稍候...")
    window.update_idletasks()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model_path_or_name = get_model_path(model_size)
        status_label.config(text=f"載入模型中 (路徑: {model_path_or_name})...")
        window.update_idletasks()

        # 因為我們已經把 HF_HOME 指向了 models，它會自動去裡面找 models--Systran--faster-whisper-base
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        status_label.config(text="正在進行語音轉文字與時間碼對齊...")
        window.update_idletasks()
        
        segments, info = model.transcribe(
            audio_file_path, 
            beam_size=5, 
            language=language_param
        )

        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_txt = os.path.join(output_folder_path, f"{base_name}_transcript.txt")
        
        status_label.config(text="正在產生帶時間碼的文字檔...")
        window.update_idletasks()
        
        with open(output_txt, "w", encoding="utf-8") as f:
            for segment in segments:
                start_str = format_timecode(segment.start)
                end_str = format_timecode(segment.end)
                text = segment.text.strip()
                f.write(f"[{start_str} --> {end_str}] {text}\n")

        status_label.config(text="處理完成！")
        messagebox.showinfo("成功", f"語音辨識完成！\n檢測語言: {info.language}\n帶時間碼的 TXT 檔案已儲存至：\n{output_txt}")

    except Exception as e:
        status_label.config(text="處理失敗")
        messagebox.showerror("錯誤發生", f"詳細錯誤訊息：\n{str(e)}")

def start_thread():
    threading.Thread(target=run_process, daemon=True).start()

# --- 純 Tkinter 介面設計 (極速、無相容性問題) ---
window = tk.Tk()
window.title(APP_NAME)
window.geometry("600x520")
window.resizable(False, False)

# 標題
title_label = tk.Label(window, text="Whisper 離線影音智慧辨識工具", font=("Microsoft JhengHei UI", 13, "bold"))
title_label.pack(pady=15)

# 檔案選擇
btn_file = tk.Button(window, text="選擇音訊或影片檔案 (MP3/WAV/MP4)", command=choose_file, width=40, bg="#f0f0f0", font=("Microsoft JhengHei UI", 10))
btn_file.pack(pady=5)
source_label = tk.Label(window, text="尚未選擇來源檔案", font=("Microsoft JhengHei UI", 9), fg="gray")
source_label.pack(pady=2)

# 輸出資料夾
btn_folder = tk.Button(window, text="選擇輸出資料夾", command=choose_output_folder, width=40, bg="#f0f0f0", font=("Microsoft JhengHei UI", 10))
btn_folder.pack(pady=10)
output_label = tk.Label(window, text="尚未選擇輸出資料夾", font=("Microsoft JhengHei UI", 9), fg="gray")
output_label.pack(pady=2)

# 模型選擇
lbl_model = tk.Label(window, text="選擇模型大小 (需與 models 內的資料夾名稱相符):", font=("Microsoft JhengHei UI", 9, "bold"))
lbl_model.pack(pady=(15, 2))
model_var = StringVar(value="base")
model_menu = tk.OptionMenu(window, model_var, "tiny", "base", "small", "medium", "large-v3")
model_menu.config(width=15, font=("Microsoft JhengHei UI", 9))
model_menu.pack(pady=2)

# 語言設定
lbl_lang = tk.Label(window, text="目標語言代碼 (例如: zh 代表中文, en 代表英文, auto 自動偵測):", font=("Microsoft JhengHei UI", 9, "bold"))
lbl_lang.pack(pady=(10, 2))
lang_var = StringVar(value="zh")
lang_entry = tk.Entry(window, textvariable=lang_var, width=20, font=("Microsoft JhengHei UI", 10), justify="center")
lang_entry.pack(pady=2)

# 狀態顯示
status_label = tk.Label(window, text="待命中", font=("Microsoft JhengHei UI", 11, "bold"), fg="blue")
status_label.pack(pady=15)

# 開始按鈕
btn_start = tk.Button(window, text="開始辨識並產出帶時間碼 TXT", command=start_thread, bg="#4CAF50", fg="white", font=("Microsoft JhengHei UI", 11, "bold"), width=35, height=2)
btn_start.pack(pady=10)

window.mainloop()
