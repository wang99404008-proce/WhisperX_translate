import os
# 【關鍵設定】強制關閉 Hugging Face 符號連結，徹底解決 Windows [WinError 1314] 權限錯誤
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
import threading
import torch
from faster_whisper import WhisperModel
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, StringVar

APP_NAME = "Whisper 離線影音智慧辨識與多語言工具"

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
    
    if os.path.exists(local_model_dir):
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
    selected_lang = lang_var.get().strip()
    
    # 如果輸入「自動偵測」或留白，則傳入 None 讓模型自己判斷
    language_param = None if selected_lang in ["auto", "自動偵測", ""] else selected_lang

    status_label.config(text="正在載入語音辨識模型，請稍候...")
    progress.start(10)
    window.update_idletasks()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model_path_or_name = get_model_path(model_size)
        status_label.config(text=f"正在載入模型中...")
        
        model = WhisperModel(model_path_or_name, device=device, compute_type=compute_type)

        status_label.config(text="正在進行語音轉文字與時間碼對齊...")
        
        # 帶入指定的語言參數 (language)
        segments, info = model.transcribe(
            audio_file_path, 
            beam_size=5, 
            language=language_param
        )

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
        messagebox.showinfo("成功", f"語音辨識完成！\n檢測語言: {info.language} (機率: {info.language_probability:.2f})\n帶時間碼的 TXT 檔案已儲存至：\n{output_txt}")

    except Exception as e:
        progress.stop()
        status_label.config(text="處理失敗")
        messagebox.showerror("錯誤", f"過程發生錯誤：\n{str(e)}")

def start_thread():
    threading.Thread(target=run_process, daemon=True).start()

# --- UI 介面設計 ---
window = tb.Window(title=APP_NAME, themename="cosmo", size=(700, 680))
window.resizable(False, False)

tb.Label(window, text="Whisper 離線影音智慧辨識與多語言工具", font=("Microsoft JhengHei UI", 14, "bold")).pack(pady=15)

# 檔案選擇
tb.Button(window, text="選擇音訊或影片檔案 (MP3/WAV/MP4)", bootstyle="primary", command=choose_file, width=45).pack(pady=5)
source_label = tb.Label(window, text="尚未選擇來源檔案", font=("Microsoft JhengHei UI", 9), bootstyle="secondary")
source_label.pack(pady=2)

# 輸出資料夾
tb.Button(window, text="選擇輸出資料夾", bootstyle="info", command=choose_output_folder, width=45).pack(pady=5)
output_label = tb.Label(window, text="尚未選擇輸出資料夾", font=("Microsoft JhengHei UI", 9), bootstyle="secondary")
output_label.pack(pady=2)

# 模型選擇
tb.Label(window, text="選擇模型大小 (需與 models 內的快取資料夾名稱相符)", font=("Microsoft JhengHei UI", 10, "bold")).pack(pady=(10, 2))
model_var = StringVar(value="base")
tb.Combobox(window, textvariable=model_var, values=["tiny", "base", "small", "medium", "large-v3"], state="readonly", width=25).pack(pady=2)

# 語言選擇設定
tb.Label(window, text="目標語言代碼 (例如: zh 代表中文, en 代表英文, 留白或填 auto 則自動偵測)", font=("Microsoft JhengHei UI", 10, "bold")).pack(pady=(10, 2))
lang_var = StringVar(value="zh")  # 預設為中文
tb.Entry(window, textvariable=lang_var, width=27).pack(pady=2)

# 狀態與進度條
status_label = tb.Label(window, text="待命中", font=("Microsoft JhengHei UI", 11))
status_label.pack(pady=8)

progress = tb.Progressbar(window, length=500, mode="indeterminate", bootstyle="success-striped")
progress.pack(pady=5)

# 開始按鈕
tb.Button(window, text="開始辨識並產出帶時間碼 TXT", bootstyle="success", command=start_thread, width=30).pack(pady=12)

window.mainloop()
