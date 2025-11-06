# ignore depreciation error
import warnings
warnings.filterwarnings("ignore")


import os
import time
from datetime import timedelta
from faster_whisper import WhisperModel
from colorama import init, Fore, Style

init(autoreset=True)

# ===== USER BENCHMARK =====
# Update this after your first benchmark run
BENCHMARK_THROUGHPUT = 5.80   # seconds_of_video / second_wall

input_folder = "videos_in"
output_folder = "transcripts_out"
os.makedirs(output_folder, exist_ok=True)

# Verify with  might be because of gpu passthrough?
'''
python -c "import ctranslate2 as ct; print(ct.get_supported_compute_types('cuda'))" C:\v\whisper_transcribe\Lib\site-packages\ctranslate2\__init__.py:8: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81. import pkg_resources {'float16', 'int8_float32', 'int8', 'int8_float16', 'bfloat16', 'int8_bfloat16', 'float32'}
'''
model = WhisperModel("large-v3", device="cuda", compute_type="float16") # 5.8~ on RTX 3080, engages tensor cores
#model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16") # 3.5x~ fallback?
#model = WhisperModel("large-v3", device="cuda", compute_type="int8") # 3.2x~ fallback?

def fmt(t):
    return str(timedelta(seconds=int(t)))

def render_bar(pct, width=50):
    filled = int(pct * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

for file in os.listdir(input_folder):
    if file.lower().endswith(".mp4"):
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file + ".txt")

        print(Fore.CYAN + "\nProcessing: " + file)
        start = time.time()

        segments_generator, info = model.transcribe(in_path, beam_size=1)
        total = info.duration  # seconds of video

        predicted_runtime = total / BENCHMARK_THROUGHPUT
        print(Fore.MAGENTA + f"Predicted Runtime: {fmt(predicted_runtime)}")
        print(Fore.MAGENTA + f"Video Duration:    {fmt(total)}")

        seg_list = []
        last_pct = -1

        for s in segments_generator:
            seg_list.append(s)
            pct = min(s.end / total, 1.0)
            ipct = int(pct * 100)

            if ipct != last_pct:
                last_pct = ipct
                elapsed = time.time() - start
                eta = predicted_runtime - elapsed
                bar = render_bar(pct)
                print(
                    Fore.GREEN +
                    f"\r{bar} {ipct:3d}%  ETA: {fmt(eta)}  Elapsed: {fmt(elapsed)}",
                    end="", flush=True
                )

        print(Fore.GREEN + "\nWriting transcript...")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in seg_list:
                f.write(f"{s.start:.2f} --> {s.end:.2f} {s.text}\n")

        end = time.time()
        proc = end - start
        throughput = total / proc  # update suggestion

        print(Fore.YELLOW + "-----------------------------------------")
        print(Fore.YELLOW + f"Video Duration:      {fmt(total)}")
        print(Fore.YELLOW + f"Processing Time:     {fmt(proc)}")
        print(Fore.YELLOW + f"Throughput:          {throughput:.2f}x (video sec per wall sec)")
        print(Fore.YELLOW + f"Benchmark Suggest:   Use BENCHMARK_THROUGHPUT={throughput:.2f}")
        print(Fore.YELLOW + f"Completed:           {time.ctime(end)}")
        print(Fore.YELLOW + "-----------------------------------------")

print(Style.BRIGHT + Fore.MAGENTA + "\nAll transcriptions complete.\n")
