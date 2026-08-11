#!/usr/bin/env python3
"""
Локальный реалтайм-переводчик английский -> русский для звонков (Zoom и т.п.).

Всё считается на твоём Mac, звук никуда не уходит в интернет.
Пайплайн: BlackHole (звук собеседника) -> VAD -> скользящее окно whisper
          -> перевод (локальный LLM qwen3 через Ollama, стримингом; либо argos)
          -> плавающее окно / терминал с субтитрами.

Потоки разнесены: whisper даёт «живой» английский сразу, а перевод LLM
приходит отдельным потоком и стримится по словам, поэтому не «висит».

Запуск:
    python realtime_translator.py                 # окно, перевод через Ollama (qwen3)
    python realtime_translator.py --argos         # перевод офлайн-движком argos
    python realtime_translator.py --ollama-model qwen2.5:7b   # модель полегче/быстрее
    python realtime_translator.py --model medium.en           # точнее распознавание
    python realtime_translator.py --console       # вывод в терминал
    python realtime_translator.py --no-original   # только перевод, без английского
    python realtime_translator.py --list-devices  # показать аудиоустройства
"""

import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.request

import numpy as np
import sounddevice as sd
import webrtcvad

# ----------------------------- Настройки -----------------------------
SAMPLE_RATE = 16000            # Гц, обязателен для whisper и webrtcvad
FRAME_MS = 30                  # длина одного аудиокадра для VAD (10/20/30)
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480 сэмплов

REFRESH_MS = 700               # как часто обновляем «живой» английский во время речи
FINALIZE_SILENCE_MS = 450      # столько тишины = конец куска -> фиксируем перевод
MAX_CHUNK_MS = 4500            # принудительная фиксация в непрерывной речи
MIN_SPEECH_MS = 250            # короче — считаем шумом, игнорируем
VAD_AGGRESSIVENESS = 2         # 0..3, выше = агрессивнее режет тишину/шум

RU_HISTORY = 2                 # сколько последних русских кусков держать на экране

DEFAULT_MODEL = "small.en"     # base.en (быстрее) / small.en / medium.en (точнее)
DEFAULT_OLLAMA_MODEL = "qwen3:14b"
FROM_CODE = "en"
TO_CODE = "ru"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_KEEP_ALIVE = "30m"      # держим модель прогретой в памяти между фразами
# /no_think надёжно выключает «рассуждения» qwen3 независимо от версии Ollama
LLM_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a professional interpreter for a job interview. "
    "Translate the user's English text into natural, fluent Russian. "
    "Keep technical terms, tool names and product names recognizable. "
    "Output ONLY the Russian translation — no quotes, no notes, no English."
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean_think(s: str) -> str:
    return _THINK_RE.sub("", s).strip()


# ----------------------------- Перевод: LLM (Ollama, стриминг) -----------------------------
def _ollama_payload(model: str, text: str, stream: bool) -> bytes:
    return json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": stream,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.2, "num_predict": 256},
    }).encode("utf-8")


def make_llm_streamer(model: str):
    """Возвращает генератор: по мере генерации отдаёт куски русского текста."""
    def gen(text: str):
        req = urllib.request.Request(
            OLLAMA_URL, data=_ollama_payload(model, text, stream=True),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break
    return gen


def llm_oneshot(model: str, text: str) -> str:
    """Разовый (нестриминговый) перевод — для проверки связи и прогрева."""
    req = urllib.request.Request(
        OLLAMA_URL, data=_ollama_payload(model, text, stream=False),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return clean_think(resp.get("message", {}).get("content", ""))


# ----------------------------- Перевод: argos (офлайн, fallback) -----------------------------
def ensure_argos(from_code: str, to_code: str) -> None:
    import argostranslate.package as pkg
    import argostranslate.translate as tr

    installed = {l.code for l in tr.get_installed_languages()}
    if from_code in installed and to_code in installed:
        return
    print(f"Скачиваю офлайн-пакет перевода {from_code}->{to_code} (один раз)...")
    pkg.update_package_index()
    available = pkg.get_available_packages()
    match = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if match is None:
        sys.exit(f"Не нашёл пакет перевода {from_code}->{to_code} в индексе argos.")
    pkg.install_from_path(match.download())
    print("Пакет перевода установлен.")


def make_argos_translator(from_code: str, to_code: str):
    import argostranslate.translate as tr

    langs = tr.get_installed_languages()
    src = next(l for l in langs if l.code == from_code)
    dst = next(l for l in langs if l.code == to_code)
    translation = src.get_translation(dst)
    return lambda text: translation.translate(text)


def make_stream_translator(llm_streamer):
    """Стриминговый перевод: LLM по кускам, argos — резерв (одним куском)."""
    argos_box = {"fn": None}

    def argos():
        if argos_box["fn"] is None:
            ensure_argos(FROM_CODE, TO_CODE)
            argos_box["fn"] = make_argos_translator(FROM_CODE, TO_CODE)
        return argos_box["fn"]

    def stream(text: str):
        if llm_streamer is not None:
            try:
                produced = False
                for piece in llm_streamer(text):
                    produced = True
                    yield piece
                if produced:
                    return
            except Exception as e:  # Ollama упала/не запущена
                print(f"[LLM недоступен: {e} — перехожу на argos]", file=sys.stderr)
        yield argos()(text)

    return stream


# ----------------------------- Вспомогательное -----------------------------
class LatestSlot:
    """Один слот на последнее значение: новые перезаписывают старые (для interim)."""

    def __init__(self):
        self._v = None
        self._lock = threading.Lock()

    def set(self, v):
        with self._lock:
            self._v = v

    def take(self):
        with self._lock:
            v, self._v = self._v, None
            return v


def transcribe(model, raw: bytes) -> str:
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


def list_devices() -> None:
    print(sd.query_devices())


def find_input_device(name_substr: str):
    for idx, dev in enumerate(sd.query_devices()):
        if name_substr.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx, dev["name"]
    return None, None


# ----------------------------- Движок: VAD + буфер -----------------------------
def engine(audio_q, final_q, interim_slot: LatestSlot, stop, want_interim: bool):
    """Копит речь в буфер, отдаёт «финалы» на паузе/по таймауту и interim-снимки."""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    silence_limit = FINALIZE_SILENCE_MS // FRAME_MS
    min_speech = MIN_SPEECH_MS // FRAME_MS
    max_frames = MAX_CHUNK_MS // FRAME_MS
    refresh_frames = REFRESH_MS // FRAME_MS

    buf = bytearray()
    speech_frames = silence_frames = frames_in_buf = since_refresh = 0
    have_speech = False

    def reset():
        nonlocal buf, speech_frames, silence_frames, frames_in_buf, since_refresh, have_speech
        buf = bytearray()
        speech_frames = silence_frames = frames_in_buf = since_refresh = 0
        have_speech = False

    while not stop.is_set():
        try:
            frame = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        if is_speech:
            buf.extend(frame); frames_in_buf += 1; since_refresh += 1
            speech_frames += 1; silence_frames = 0; have_speech = True
        elif have_speech:
            buf.extend(frame); frames_in_buf += 1; since_refresh += 1
            silence_frames += 1

        if have_speech and speech_frames >= min_speech and (
            silence_frames >= silence_limit or frames_in_buf >= max_frames
        ):
            final_q.put(bytes(buf))       # зафиксировать этот кусок
            reset()
            continue

        if (want_interim and have_speech and speech_frames >= min_speech
                and since_refresh >= refresh_frames):
            interim_slot.set(bytes(buf))  # обновить «живой» английский
            since_refresh = 0


# ----------------------------- Поток STT: распознавание -----------------------------
def stt_worker(final_q, interim_slot: LatestSlot, translate_q, display_q, model, stop):
    """Финалы -> в очередь перевода; interim -> live-строка английского."""
    while not stop.is_set():
        try:
            raw = final_q.get_nowait()
        except queue.Empty:
            raw = None

        if raw is not None:
            en = transcribe(model, raw)
            if en:
                translate_q.put(en)                 # перевод сделает отдельный поток
            continue

        snap = interim_slot.take()
        if snap is not None:
            en = transcribe(model, snap)
            if en:
                display_q.put(("interim", en, None))
        else:
            time.sleep(0.04)


# ----------------------------- Поток перевода: LLM-стриминг -----------------------------
def translate_worker(translate_q, display_q, stream_fn, stop):
    while not stop.is_set():
        try:
            en = translate_q.get(timeout=0.3)
        except queue.Empty:
            continue
        raw_acc = ""
        last_shown = None
        for piece in stream_fn(en):
            raw_acc += piece
            # ждём закрытия <think>, если модель всё же начала «думать»
            if "<think>" in raw_acc and "</think>" not in raw_acc:
                continue
            cleaned = clean_think(raw_acc)
            if cleaned and cleaned != last_shown:
                last_shown = cleaned
                display_q.put(("partial", en, cleaned))   # живой русский по словам
        final_ru = clean_think(raw_acc)
        if final_ru:
            display_q.put(("final", en, final_ru))         # зафиксировать кусок


# ----------------------------- Вывод: терминал -----------------------------
def run_console(display_q, stop, show_original: bool):
    print("Режим терминала. Ctrl+C — выход.\n")
    try:
        while not stop.is_set():
            try:
                kind, en, ru = display_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if kind == "interim":
                if show_original:
                    sys.stdout.write("\r\033[K  " + en[-100:])
                    sys.stdout.flush()
            elif kind == "final":
                if show_original:
                    sys.stdout.write("\r\033[K")
                    print(f"  {en}")
                print(f"РУ: {ru}\n")
            # 'partial' в терминале не печатаем, чтобы не засорять вывод
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


# ----------------------------- Вывод: окно -----------------------------
def run_ui(display_q, stop, show_original: bool):
    import tkinter as tk

    root = tk.Tk()
    root.title("RU перевод")
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.88)
    except tk.TclError:
        pass
    root.configure(bg="black")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = int(sw * 0.6), 180
    x, y = (sw - w) // 2, sh - h - 90
    root.geometry(f"{w}x{h}+{x}+{y}")

    ru_var = tk.StringVar(value="Слушаю…")
    en_var = tk.StringVar(value="")
    committed = []          # зафиксированные русские куски
    partial = {"txt": ""}   # текущий стримящийся кусок

    if show_original:  # оригинал — сверху, мелким приглушённым шрифтом (живой)
        tk.Label(
            root, textvariable=en_var, fg="#9aa0a6", bg="black",
            font=("Helvetica", 15), wraplength=w - 40, justify="center",
        ).pack(side="top", fill="x", padx=12, pady=(12, 0))

    tk.Label(  # перевод — снизу, крупно
        root, textvariable=ru_var, fg="white", bg="black",
        font=("Helvetica", 26, "bold"), wraplength=w - 40, justify="center",
    ).pack(side="bottom", expand=True, fill="both", padx=12, pady=(4, 12))

    def render_ru():
        tail = committed[-(RU_HISTORY - 1):] if RU_HISTORY > 1 else []
        parts = tail + ([partial["txt"]] if partial["txt"] else [])
        ru_var.set(" ".join(parts) if parts else " ".join(committed[-RU_HISTORY:]))

    def start_move(e):
        root._dx, root._dy = e.x, e.y

    def do_move(e):
        root.geometry(f"+{e.x_root - root._dx}+{e.y_root - root._dy}")

    root.bind("<Button-1>", start_move)
    root.bind("<B1-Motion>", do_move)

    def poll():
        try:
            while True:
                kind, en, ru = display_q.get_nowait()
                if kind == "interim":
                    en_var.set(en)
                elif kind == "partial":
                    en_var.set(en)
                    partial["txt"] = ru
                    render_ru()
                else:  # final
                    en_var.set(en)
                    partial["txt"] = ""
                    committed.append(ru)
                    del committed[:-max(RU_HISTORY, 4)]
                    render_ru()
        except queue.Empty:
            pass
        if stop.is_set():
            root.destroy()
            return
        root.after(60, poll)

    root.after(60, poll)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop.set(), root.destroy()))
    root.mainloop()
    stop.set()


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Локальный переводчик англ->рус для звонков")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="модель whisper: base.en / small.en / medium.en")
    ap.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL,
                    help="модель Ollama для перевода (по умолчанию qwen3:14b)")
    ap.add_argument("--argos", action="store_true",
                    help="переводить офлайн-движком argos вместо Ollama")
    ap.add_argument("--device-name", default="BlackHole",
                    help="часть имени входного аудиоустройства (по умолчанию BlackHole)")
    ap.add_argument("--no-original", action="store_true",
                    help="показывать только перевод, без английского оригинала")
    ap.add_argument("--console", action="store_true",
                    help="вывод в терминал вместо окна (не требует Tk)")
    ap.add_argument("--list-devices", action="store_true",
                    help="показать аудиоустройства и выйти")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    dev_idx, dev_name = find_input_device(args.device_name)
    if dev_idx is None:
        print(f"Не нашёл входное устройство «{args.device_name}».")
        print("Проверь, что BlackHole установлен. Список устройств:\n")
        list_devices()
        sys.exit(1)
    print(f"Слушаю устройство: [{dev_idx}] {dev_name}")

    show_original = not args.no_original

    # переводчик
    llm_streamer = None if args.argos else make_llm_streamer(args.ollama_model)
    stream_fn = make_stream_translator(llm_streamer)
    if llm_streamer is not None:
        print(f"Проверяю Ollama ({args.ollama_model}) и прогреваю модель...")
        try:
            sample = llm_oneshot(args.ollama_model, "Tell me about your experience.")
            print(f"Ollama работает. Пример: {sample[:70]}")
        except Exception as e:
            print(f"[!] Ollama не отвечает: {e}")
            print("    Запусти сервер: ollama serve   (или добавь флаг --argos)")
    else:
        ensure_argos(FROM_CODE, TO_CODE)

    print(f"Загружаю модель распознавания «{args.model}» (int8)...")
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print("Готово. Говори в Zoom — перевод появится в окне.\n")

    audio_q: queue.Queue = queue.Queue()
    final_q: queue.Queue = queue.Queue()
    translate_q: queue.Queue = queue.Queue()
    display_q: queue.Queue = queue.Queue()
    interim_slot = LatestSlot()
    stop = threading.Event()

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(bytes(indata))

    threading.Thread(
        target=engine, args=(audio_q, final_q, interim_slot, stop, show_original),
        daemon=True,
    ).start()
    threading.Thread(
        target=stt_worker,
        args=(final_q, interim_slot, translate_q, display_q, model, stop),
        daemon=True,
    ).start()
    threading.Thread(
        target=translate_worker, args=(translate_q, display_q, stream_fn, stop),
        daemon=True,
    ).start()

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=FRAME_SAMPLES, device=dev_idx, callback=audio_cb,
    )
    with stream:
        try:
            if args.console:
                run_console(display_q, stop, show_original)
            else:
                try:
                    run_ui(display_q, stop, show_original)
                except ModuleNotFoundError:
                    print("Tk не найден — переключаюсь в режим терминала.\n")
                    run_console(display_q, stop, show_original)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
    print("Остановлено.")


if __name__ == "__main__":
    main()
