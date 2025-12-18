#!/usr/bin/env python3
"""
text_video_xtts_pipeline_fixed.py

Merged / upgraded version (CPU-only, XTTS-ready) combining:
 - your original pipeline (synthesis, subtitles, SRT, BG ducking)
 - cinematic image->video section from the working script (KenBurns, vignette, grain)
Features:
 - captions ON/OFF via control.json ("caption_on": true/false)
 - uses assets/images for images, bg_music folder for bg audio
 - robust multi-image selection (no single-image bug)
 - creates timestamped output folder
 - CPU-friendly 
  (threads limited)
 - cleanups temporary images
Top of file includes minimal README & pip install commands

Mapping notes (if you compare with your original files)
 - Replaced create_video_from_captions(...) with create_video_from_segments_and_images(...)
 - Added helpers: select_images_for_duration, process_image_for_cinematic, make_kenburns_clip (from working code)
"""

# ----------------- SHORT README / DEPENDENCIES -----------------
# pip installs (run in your venv)
# pip install --upgrade pip
# pip install moviepy pillow pydub soundfile numpy TTS symspellpy language-tool-python opencv-python-headless
# NOTE: opencv-python-headless is optional; if not installed vignette will still work via PIL method
#
# Hardware notes:
# - Designed for CPU-only machines (Intel i3 with 8GB RAM). Keep moviepy threads small in control.json (default 2).
# - If 'sox' is required for tempo changes install sox on system (apt install sox) or set use_sox_for_tempo to False.
#
# Usage:
# 1) Edit control.json (sample provided below) — ensure assets/images has images and bg_music folder contains tracks (optional)
# 2) Place your script in scripts/input.txt (or pass --test_sample)
# 3) python3 text_video_xtts_pipeline_fixed.py --config control.json

import os
import sys
import json
import time
import shutil
import random
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from tempfile import mkstemp, gettempdir, NamedTemporaryFile
from collections import Counter
import math
import re
import inspect
import difflib
import numpy as np
import re



# OpenCV optional (for advanced vignette) - not required
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# Limit CPU threads for safety on low-RAM machines
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# audio/video libs
from pydub import AudioSegment, silence, effects
import soundfile as sf
from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageOps, ImageEnhance
from moviepy.editor import (
    ImageClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips, ColorClip
)

# TTS
try:
    from TTS.api import TTS
except Exception as e:
    print("ERROR: TTS import failed. Install Coqui TTS in your venv: pip install TTS")
    raise

# optional helpers
try:
    import language_tool_python
    LANGTOOL_AVAILABLE = True
except Exception:
    LANGTOOL_AVAILABLE = False

try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except Exception:
    SYMSPELL_AVAILABLE = False

# ---------- BATCH PROCESSING + MEMORY HELPERS (ADD THESE IMPORTS) ----------
import gc
from typing import List, Optional
try:
    import psutil
except Exception:
    psutil = None

# torch is optional but we will attempt safe cleanup if present
try:
    import torch
except Exception:
    torch = None
# ---------- END IMPORTS ----------


# logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
LOG = logging.getLogger("xtts_pipeline_fixed")

# ---------------- default control (written if missing) ----------------
DEFAULT_CONTROL = {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "local_xtts_path": "",
    "speaker": "p225",
    "use_speaker_wav": True,
    "speaker_wav_path": "sample.wav",
    "force_single_speaker": False,
    "fallback_single_speaker_model": None,
    "voices_folder": "voices",
    "voices_map": {},
    "voice_defaults": {},
    "scripts_folder": "scripts",
    "input_filename": "input.txt",
    "bg_music_folder": "bg_music",
    "output_folder": "output",
    "normalize_hindi": True,
    "normalizer_options": {"ellipsis_style": "ellipsis", "use_langtool_if_available": False},
    "tts": {"sampling_rate": 24000, "device": "cpu", "speaker_params": {}},
    "movie": {
        "orientation": "vertical",
        "vertical": {"width": 720, "height": 1280},
        "horizontal": {"width": 1280, "height": 720},
        "fps": 30,
        "bitrate": "3500k",
        "preset": "medium",
        "font_path": "assets/fonts/NotoSansDevanagari-Regular.ttf",
        "font_size": 56,
        "caption_color": "#FFD54F",
        "caption_shadow": "black",
        "caption_margin_bottom": 170,
        "max_width_padding": 160,
        "zoom_strength": 0.02,
        "vignette_strength": 0.18,
        "typing_fps": 24,
        "cinematic": {"vignette": True, "grain": True, "letterbox": True, "background_blur": 0.0, "soft_glow": False}
    },
    "bg": {
        "bg_reduce_db": 14,
        "bg_lowpass_hz": 9000,
        "bg_slow_tempo": 0.97,
        "use_sox_for_tempo": True,
        "bg_duck_db": 10,
        "bg_duck_threshold_db": -42.0,
        "bg_duck_window_ms": 200,
        "bg_music_level": 0.18
    },
    "runtime": {"torch_num_threads": 2, "moviepy_threads": 2, "temp_dir": None, "cleanup_tmp": True},
    # caption on/off (this is the new toggle)
    "caption_on": True,
    # image selection mode: false = sequential (default), true = random per-segment
    "image_random": False,
}

def write_default_control(path: Path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(DEFAULT_CONTROL, f, indent=2, ensure_ascii=False)
    LOG.info("Default control.json written to %s - edit it and re-run.", path)

def load_control(path: Path):
    if not path.exists():
        write_default_control(path)
        sys.exit(0)
    cfg = json.loads(path.read_text(encoding="utf8"))
    for k, v in DEFAULT_CONTROL.items():
        if k not in cfg:
            cfg[k] = v
    if "movie" not in cfg:
        cfg["movie"] = DEFAULT_CONTROL["movie"]
    if "bg" not in cfg:
        cfg["bg"] = DEFAULT_CONTROL["bg"]
    if "runtime" not in cfg:
        cfg["runtime"] = DEFAULT_CONTROL["runtime"]
    return cfg

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def make_timestamped_output(base_folder: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_folder) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

# -------------------- Text cleaning (same as yours) --------------------
def unicode_normalize(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", text)

def replace_smart_quotes(text: str) -> str:
    smart_map = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u00A0': ' '
    }
    for k, v in smart_map.items():
        text = text.replace(k, v)
    return text

def re_sub(pattern, repl, text, flags=0):
    return re.sub(pattern, repl, text, flags=flags)

def re_sub_func(pattern, func, text, flags=0):
    return re.sub(pattern, func, text, flags=flags)

def collapse_repeated_punctuation(text: str, ellipsis_style: str = "ellipsis"):
    changes = Counter()
    new = text
    tmp = re_sub(r'।{2,}', '।', new)
    if tmp != new:
        changes['double_danda'] += 1
    new = tmp
    tmp = re_sub(r'!{2,}', '!', new)
    if tmp != new:
        changes['exclaim'] += 1
    new = tmp
    tmp = re_sub(r'\?{2,}', '?', new)
    if tmp != new:
        changes['question'] += 1
    new = tmp
    if ellipsis_style == "ellipsis":
        tmp = re_sub(r'(\.\s*){3,}', '…', new)
        tmp = re_sub(r'\.{3,}', '…', tmp)
        if tmp != new:
            changes['ellipsis_to_char'] += 1
        new = tmp
    else:
        tmp = re_sub(r'\.{2,}', '.', new)
        if tmp != new:
            changes['ellipsis_to_dot'] += 1
        new = tmp

    def _replace_mixed(match):
        seq = match.group(0)
        return seq[-1]

    tmp = re_sub_func(r'([!?.]{2,})', _replace_mixed, new)
    if tmp != new:
        changes['mixed_punct'] += 1
    new = tmp
    return new, changes

def fix_punctuation_spacing(text: str):
    changes = Counter()
    new = re_sub(r'\s+([,।.!?…])', r'\1', text)
    if new != text:
        changes['space_before_punct'] += 1
    text = new
    new = re_sub(r'([,।.!?…])([^\s\n])', r'\1 \2', text)
    if new != text:
        changes['space_after_punct'] += 1
    text = new
    new = re_sub(r'[ \t]{2,}', ' ', text)
    if new != text:
        changes['collapse_spaces'] += 1
    text = new
    return text, changes

def collapse_blank_lines(text: str):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    while lines and lines[0].strip() == '':
        lines.pop(0)
    while lines and lines[-1].strip() == '':
        lines.pop()
    out_lines = []
    collapsed = 0
    blank_run = 0
    for ln in lines:
        if ln.strip() == '':
            blank_run += 1
            if blank_run == 1:
                out_lines.append('')
            else:
                collapsed += 1
        else:
            blank_run = 0
            out_lines.append(ln.rstrip())
    return '\n'.join(out_lines) + '\n', collapsed

def simple_conservative_fixes(text: str):
    changes = Counter()
    new = re_sub(r'(?<!\n) {2,}', ' ', text)
    if new != text:
        changes['collapse_interior_spaces'] += 1
    text = new
    new = text.replace('\u00A0', ' ')
    if new != text:
        changes['nbsp_fix'] += 1
    text = new
    return text, changes

def apply_language_tool_if_available(text: str):
    if not LANGTOOL_AVAILABLE:
        return text, 0
    try:
        tool = language_tool_python.LanguageTool('hi')
        matches = tool.check(text)
        if not matches:
            return text, 0
        corrected = language_tool_python.utils.correct(text, matches)
        return corrected, len(matches)
    except Exception:
        return text, 0

def clean_text_pipeline(text: str, ellipsis_style: str = "ellipsis", use_langtool: bool = False):
    stats = Counter()
    text = unicode_normalize(text)
    text = replace_smart_quotes(text)
    c1_text, c1 = simple_conservative_fixes(text)
    text = c1_text
    stats.update(c1)
    c2_text, c2 = collapse_repeated_punctuation(text, ellipsis_style=ellipsis_style)
    text = c2_text
    stats.update(c2)
    c3_text, c3 = fix_punctuation_spacing(text)
    text = c3_text
    stats.update(c3)
    text, collapsed_blanks = collapse_blank_lines(text)
    if collapsed_blanks:
        stats['collapsed_blank_lines'] = collapsed_blanks
    lines = [ln.rstrip() for ln in text.split('\n')]
    text = '\n'.join([ln.strip() for ln in lines]).strip() + '\n'
    lang_changes = 0
    if use_langtool and LANGTOOL_AVAILABLE:
        text, lang_changes = apply_language_tool_if_available(text)
        stats['langtool_matches'] = lang_changes
    stats['cleaned_chars'] = len(text)
    stats['cleaned_lines'] = text.count('\n') + (0 if text.endswith('\n') else 1)
    return text, stats
def strict_xtts_clean(text: str) -> str:
    """
    Strict cleaner per user's rules:
      - Keep only Devanagari letters (\u0900-\u097F plus extended block \uA8E0-\uA8FF),
        space, and allowed punctuation: ';' '?' '।' (U+0964).
      - Remove all English/Latin letters/numbers and any other symbols.
      - Collapse whitespace/newlines to single spaces.
      - Insert one '।' after every 8 Devanagari words (attached to the 8th word).
      - Return a single-line string (no trailing newline).
    """
    import unicodedata
    import re

    if not isinstance(text, str):
        text = str(text)

    # 1) Normalize
    text = unicodedata.normalize("NFC", text)

    # Allowed unicode ranges/chars:
    # Devanagari: U+0900..U+097F ; Extended Devanagari A: U+A8E0..U+A8FF (rare)
    DEVANAGARI_RANGES = r"\u0900-\u097F\uA8E0-\uA8FF"
    # Allowed punctuation characters (kept as-is)
    allowed_punct = ";\u0964?"  # ; , purnaviram (।), ?

    # 2) Replace any character that is NOT (Devanagari OR whitespace OR allowed punctuation) with a space
    # This removes Latin letters, digits, banned punctuation, emojis, etc.
    pattern_keep = f"[{DEVANAGARI_RANGES}\\s{re.escape(allowed_punct)}]"
    # Equivalent: keep only those; replace everything else with space
    cleaned_chars = []
    for ch in text:
        if re.match(pattern_keep, ch):
            cleaned_chars.append(ch)
        else:
            # convert disallowed char to space (so words separate cleanly)
            cleaned_chars.append(" ")
    s = "".join(cleaned_chars)

    # 3) Collapse all whitespace (spaces/newlines/tabs) into single spaces and strip ends
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return ""

    # 4) Split into tokens by spaces (tokens will contain only Devanagari letters and allowed punct)
    tokens = [t for t in s.split(" ") if t != ""]

    # 5) Build output, inserting one '।' after every 8 Devanagari words.
    out_tokens = []
    dev_word_count = 0
    # regex to detect if token contains at least one Devanagari letter
    dev_re = re.compile(f"[{DEVANAGARI_RANGES}]")

    for tok in tokens:
        # keep token as-is (it will only contain Devanagari or allowed punctuation)
        out_tokens.append(tok)
        # count as a "word" only if it contains at least one Devanagari letter
        if dev_re.search(tok):
            dev_word_count += 1
            if dev_word_count % 8 == 0:
                # attach purnaviram to the last token (no space before it)
                # remove any trailing allowed punctuation before appending to avoid duplicates
                last = out_tokens[-1].rstrip(" " + allowed_punct)
                out_tokens[-1] = last + "।"

    # 6) Join with single spaces
    out = " ".join(out_tokens)

    # 7) Ensure no space before allowed punctuation ; ? ।  (tighten spacing)
    out = re.sub(r"\s+([;?\u0964])", r"\1", out)

    # 8) Final collapse of multiple spaces (just in case) and trim
    out = re.sub(r"\s+", " ", out).strip()

    return out

# ----------------- Voice selection and TTS helpers (kept from original) -----------------
def select_voice_sample_from_config(cfg: dict, voices_root: Path):
    voice_name_raw = (cfg.get("voice_name") or "").strip()
    voice_name = voice_name_raw.lower()
    voices_map = cfg.get("voices_map", {}) or {}
    voice_defaults = cfg.get("voice_defaults", {}) or {}
    default_params = voice_defaults.get("default", {"pitch_scale": 1.0, "speed_scale": 1.0, "formant_shift": 0.0})

    def check_in_folder(candidate_fname, folder_name=None):
        if folder_name:
            p = voices_root / folder_name / candidate_fname
            if p.exists():
                return p
            p2 = voices_root / folder_name / "sample.wav"
            if p2.exists():
                return p2
        proot = voices_root / candidate_fname
        if proot.exists():
            return proot
        return None

    if voice_name in voices_map:
        fname = voices_map[voice_name]
        if "/" in fname or "\\" in fname:
            cand = voices_root / fname
            if cand.exists():
                return cand, voice_defaults.get(voice_name, default_params)
        p = check_in_folder(fname, folder_name=voice_name)
        if p:
            return p, voice_defaults.get(voice_name, default_params)
        p2 = check_in_folder(fname, folder_name=None)
        if p2:
            return p2, voice_defaults.get(voice_name, default_params)

    p_folder = voices_root / voice_name / "sample.wav"
    if p_folder.exists():
        return p_folder, voice_defaults.get(voice_name, default_params)

    p_direct = voices_root / voice_name
    if p_direct.exists() and p_direct.is_file():
        return p_direct, default_params

    if voices_map:
        keys = list(voices_map.keys())
        matches = difflib.get_close_matches(voice_name, keys, n=1, cutoff=0.6)
        if matches:
            k = matches[0]
            fname = voices_map.get(k)
            p = check_in_folder(fname, folder_name=k)
            if p:
                return p, voice_defaults.get(k, default_params)
            p2 = check_in_folder(fname, folder_name=None)
            if p2:
                return p2, voice_defaults.get(k, default_params)

    try:
        entries = [p for p in voices_root.iterdir() if p.exists()]
    except Exception:
        entries = []
    cand_names = []
    mapping = {}
    for e in entries:
        if e.is_dir():
            cand_names.append(e.name.lower())
            mapping[e.name.lower()] = e / "sample.wav"
        elif e.is_file():
            cand_names.append(e.stem.lower())
            mapping[e.stem.lower()] = e
    if cand_names:
        matches = difflib.get_close_matches(voice_name, cand_names, n=1, cutoff=0.6)
        if matches:
            m = matches[0]
            p = mapping.get(m)
            if p and p.exists():
                return p, voice_defaults.get(m, default_params)

    default_fname = voices_map.get("default")
    if default_fname:
        p = check_in_folder(default_fname, folder_name=None)
        if p:
            return p, voice_defaults.get("default", default_params)

    try:
        for e in entries:
            if e.is_dir():
                cand = e / "sample.wav"
                if cand.exists():
                    return cand, default_params
        for e in entries:
            if e.is_file() and e.suffix.lower() in [".wav", ".flac", ".mp3"]:
                return e, default_params
    except Exception:
        pass

    return None, default_params

# ----------------- BG helpers (kept) -----------------
def _apply_low_pass(seg: AudioSegment, cutoff_hz: int = 9000):
    try:
        return seg.low_pass_filter(cutoff_hz)
    except Exception:
        return seg

def _compress_segment(seg: AudioSegment, threshold_db: float = -20.0, ratio: float = 3.0):
    try:
        from pydub.effects import compress_dynamic_range
        return compress_dynamic_range(seg, threshold=threshold_db, ratio=ratio)
    except Exception:
        return seg

def _slowdown_with_sox(in_path: str, out_path: str, tempo: float = 0.97):
    cmd = ["sox", in_path, out_path, "tempo", str(tempo)]
    subprocess.run(cmd, check=True)

def prepare_bg_music_cinematic(src: Path, target_sr=24000, target_channels=1, target_dur_ms=None,
                               reduce_db_base: int = 20, lowpass_hz: int = 9000, slow_tempo: float = 0.97,
                               use_sox_for_tempo: bool = True):
    seg = AudioSegment.from_file(str(src))
    seg = seg.set_frame_rate(target_sr).set_channels(target_channels).set_sample_width(2)

    if use_sox_for_tempo and shutil.which("sox") is not None and slow_tempo < 0.999:
        tmp_in = Path(gettempdir()) / f"bg_tmp_in_{int(time.time() * 1000)}.wav"
        tmp_out = Path(gettempdir()) / f"bg_tmp_out_{int(time.time() * 1000)}.wav"
        seg.export(str(tmp_in), format="wav")
        try:
            _slowdown_with_sox(str(tmp_in), str(tmp_out), tempo=slow_tempo)
            seg = AudioSegment.from_wav(str(tmp_out)).set_frame_rate(target_sr).set_channels(target_channels)
        except Exception as e:
            LOG.warning("sox tempo failed (%s). Continuing without tempo change.", e)
        finally:
            try:
                tmp_in.unlink()
            except Exception:
                pass
            try:
                tmp_out.unlink()
            except Exception:
                pass
    else:
        if slow_tempo < 0.999:
            new_frame_rate = int(seg.frame_rate * slow_tempo)
            seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
            seg = seg.set_frame_rate(target_sr)

    seg = _apply_low_pass(seg, cutoff_hz=lowpass_hz)
    seg = _compress_segment(seg)

    if target_dur_ms and len(seg) > target_dur_ms:
        seg = seg[:target_dur_ms]
    elif target_dur_ms and len(seg) < target_dur_ms:
        seg = seg + AudioSegment.silent(duration=(target_dur_ms - len(seg)))

    seg = seg - reduce_db_base
    return seg

def duck_bg_under_voice(bg_seg: AudioSegment, voice_seg: AudioSegment,
                       threshold_db: float = -42.0, reduction_db: int = 10, window_ms: int = 200):
    if window_ms <= 0:
        return bg_seg
    if len(bg_seg) < len(voice_seg):
        bg_seg = bg_seg + AudioSegment.silent(duration=(len(voice_seg) - len(bg_seg)))
    else:
        voice_seg = voice_seg + AudioSegment.silent(duration=(len(bg_seg) - len(voice_seg)))

    out = AudioSegment.silent(duration=len(bg_seg))
    num_windows = math.ceil(len(bg_seg) / window_ms)

    for i in range(num_windows):
        start = i * window_ms
        end = min((i + 1) * window_ms, len(bg_seg))
        voice_chunk = voice_seg[start:end]
        bg_chunk = bg_seg[start:end]
        try:
            v_db = voice_chunk.dBFS
        except Exception:
            v_db = -100.0
        if v_db > threshold_db:
            chunk_out = bg_chunk - reduction_db
        else:
            chunk_out = bg_chunk
        out = out.overlay(chunk_out, position=start)

    return out

# ----------------- TTS synthesis helpers (kept) -----------------
def _filter_kwargs_for_callable(fn, kwargs):
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs.copy()
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs.copy()
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def validate_sample_wav(path: Path):
    try:
        info = sf.info(str(path))
        sr = info.samplerate
        frames = info.frames
        ch = info.channels
        dur = frames / sr if sr and frames else 0.0
        msgs = []
        if dur < 3.0:
            msgs.append(f"too short ({dur:.2f}s) - need >= 3s")
        if ch != 1:
            msgs.append(f"channels={ch} - need mono (1)")
        if sr < 22050:
            msgs.append(f"samplerate={sr} - recommend 24000")
        if msgs:
            return False, "; ".join(msgs)
        return True, f"OK (dur={dur:.2f}s, sr={sr}, ch={ch})"
    except Exception as e:
        return False, f"cannot read sample.wav ({e})"

def clean_sample_wav(in_path: Path, out_path: Path):
    seg = AudioSegment.from_file(str(in_path))
    seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    trimmed = silence.strip_silence(seg, silence_len=400, silence_thresh=-50)
    if len(trimmed) < 3000:
        pad = AudioSegment.silent(duration=500)
        trimmed = pad + trimmed + pad
    trimmed.export(str(out_path), format="wav")
    return out_path

def get_model_speakers(tts_obj):
    cand = []
    try:
        sp = getattr(tts_obj, "speakers", None)
        if sp:
            if isinstance(sp, dict):
                cand = list(sp.keys())
            elif isinstance(sp, (list, tuple)):
                cand = list(sp)
    except Exception:
        cand = []
    try:
        if not cand and hasattr(tts_obj, "get_speakers"):
            cand = list(tts_obj.get_speakers())
    except Exception:
        pass
    return cand

def synth_sentence_to_file(tts, text, out_wav: Path, speaker_wav: Path = None, synth_kwargs: dict = None, language="hi"):
    synth_kwargs = synth_kwargs or {}
    if speaker_wav:
        ok, msg = validate_sample_wav(speaker_wav)
        if not ok:
            LOG.warning("speaker_wav validation: %s. Attempting to clean...", msg)
            tmp = speaker_wav.parent / (speaker_wav.stem + "_clean.wav")
            try:
                clean_sample_wav(speaker_wav, tmp)
                speaker_wav = tmp
                ok2, msg2 = validate_sample_wav(speaker_wav)
                if not ok2:
                    LOG.warning("Cleaned sample still not ideal: %s", msg2)
                else:
                    LOG.info("Cleaned sample OK: %s", msg2)
            except Exception as e:
                LOG.warning("Failed to auto-clean sample_wav: %s", e)

        if speaker_wav and hasattr(tts, "tts_with_vc_to_file"):
            try:
                call_kwargs = _filter_kwargs_for_callable(tts.tts_with_vc_to_file, synth_kwargs)
                LOG.info("Trying tts_with_vc_to_file (voice conversion with sample)...")
                tts.tts_with_vc_to_file(text=text, speaker_wav=str(speaker_wav), file_path=str(out_wav), language=language, **call_kwargs)
                LOG.info("tts_with_vc_to_file succeeded.")
                return
            except Exception as e:
                LOG.warning("tts_with_vc_to_file failed: %s", e)

    speakers = get_model_speakers(tts)
    if speakers:
        LOG.info("Model exposes %d speakers; trying a few...", len(speakers))
        for sp in speakers[:12]:
            try:
                call_kwargs = _filter_kwargs_for_callable(tts.tts_to_file, synth_kwargs)
                LOG.info("Trying speaker id/name: %s", sp)
                tts.tts_to_file(text=text, speaker=sp, file_path=str(out_wav), language=language, **call_kwargs)
                LOG.info("Succeeded with speaker %s", sp)
                return
            except Exception as e:
                LOG.debug("speaker %s failed: %s", sp, e)

    candidates = []
    for i in range(0, 80):
        candidates.extend([i, str(i)])
    for i in range(0, 40):
        candidates += [f"p{i}", f"sp{i}", f"speaker_{i}", f"s{i}", f"voice_{i}"]
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    for sp in candidates:
        try:
            call_kwargs = _filter_kwargs_for_callable(tts.tts_to_file, synth_kwargs)
            tts.tts_to_file(text=text, speaker=sp, file_path=str(out_wav), language=language, **call_kwargs)
            LOG.info("Succeeded with candidate speaker: %s", sp)
            return
        except Exception:
            continue

    if speaker_wav:
        try:
            call_kwargs = _filter_kwargs_for_callable(tts.tts_to_file, synth_kwargs)
            LOG.info("Trying final fallback: tts_to_file with speaker_wav...")
            tts.tts_to_file(text=text, file_path=str(out_wav), speaker_wav=str(speaker_wav), language=language, **call_kwargs)
            LOG.info("Final fallback with speaker_wav succeeded.")
            return
        except Exception as e:
            LOG.warning("Final fallback with speaker_wav failed: %s", e)

    raise RuntimeError(
        "TTS model requires a 'speaker' or a valid 'speaker_wav' but none worked.\n"
        "Fixes:\n"
        "  1) Ensure voices/<voice_name>/sample.wav is mono PCM16, >=3s, 24000 Hz\n"
        "  2) Add 'speaker' to control.json with a model speaker id if present.\n"
        "  3) Or set 'force_single_speaker': true in control.json to skip cloning."
    )

# ----------------- SRT writer -----------------
def write_srt(segments, out_path: Path):
    def fmt_time(s):
        td = timedelta(seconds=float(s))
        total = td.total_seconds()
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        secs = int(total % 60)
        msecs = int((total - int(total)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

    lines = []
    for i, (st, ed, txt) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_time(st)} --> {fmt_time(ed)}")
        lines.append(txt)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf8")

# ----------------- Cinematic image helpers (from working script) -----------------
_TEMP_IMAGES = []

def apply_vignette_pil(pil_img, strength=1.6):
    W, H = pil_img.size
    x = np.linspace(-1, 1, W)[None, :]
    y = np.linspace(-1, 1, H)[:, None]
    r = np.sqrt(x * x + y * y)
    mask = (1 - np.clip(r, 0, 1)) ** strength
    arr = np.array(pil_img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=2)
    for c in range(3):
        arr[:, :, c] *= mask
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr)

def add_film_grain_pil(pil_img, amount=0.04):
    W, H = pil_img.size
    noise = np.random.normal(loc=128, scale=24, size=(H, W, 1)).clip(0, 255).astype("uint8")
    noise = np.repeat(noise, 3, axis=2)
    noise_img = Image.fromarray(noise, mode="RGB")
    return Image.blend(pil_img, noise_img, amount)

def process_image_for_cinematic(src_path, target_W, target_H, enable_vignette=True, enable_grain=True, scale_factor=1.15):
    try:
        im = Image.open(src_path).convert("RGB")
    except Exception:
        return src_path
    new_w = int(target_W * scale_factor)
    new_h = int(target_H * scale_factor)
    im.thumbnail((new_w, new_h), Image.LANCZOS)
    try:
        im = ImageEnhance.Color(im).enhance(1.08)
        im = ImageEnhance.Contrast(im).enhance(1.04)
        im = ImageEnhance.Sharpness(im).enhance(1.03)
    except Exception:
        pass
    if enable_vignette:
        try:
            im = apply_vignette_pil(im, strength=1.6)
        except Exception:
            pass
    if enable_grain:
        try:
            im = add_film_grain_pil(im, amount=0.03)
        except Exception:
            pass
    tmpf = NamedTemporaryFile(delete=False, suffix=".jpg", prefix="cin_img_")
    tmpf.close()
    out_path = tmpf.name
    im.save(out_path, quality=95, optimize=True)
    _TEMP_IMAGES.append(out_path)
    return out_path

def select_images_for_duration(all_images, video_duration, per_image_duration=3, max_images_per_video=1000):
    if per_image_duration is None or per_image_duration <= 0:
        per_image_duration = 3
    needed = math.ceil(float(video_duration) / float(per_image_duration))
    needed = int(min(max(1, needed), max_images_per_video))
    imgs = list(all_images)
    if not imgs:
        return []
    random.shuffle(imgs)
    if len(imgs) >= needed:
        return imgs[:needed]
    result = imgs[:]
    more = needed - len(result)
    idx = 0
    while more > 0:
        candidate = imgs[idx % len(imgs)]
        if result and candidate == result[-1]:
            idx += 1
            candidate = imgs[idx % len(imgs)]
        result.append(candidate)
        idx += 1
        more -= 1
    return result

def make_kenburns_clip(image_path, size, duration, zoom_start=1.0, zoom_end=1.08, pan_dx=0.06, pan_dy=0.04, fade=0.7, enable_vignette=True, enable_grain=True):
    W, H = size
    processed_path = process_image_for_cinematic(image_path, W, H, enable_vignette=enable_vignette, enable_grain=enable_grain, scale_factor=1.12)
    clip = ImageClip(processed_path).resize(width=W)
    # since clip.w/clip.h are functions, we create size_at based on original clip size
    iw, ih = clip.w, clip.h
    def size_at(t):
        frac = t / max(1e-6, duration)
        ease = (1 - math.cos(frac * math.pi)) / 2
        scale = zoom_start + (zoom_end - zoom_start) * ease
        return (iw * scale, ih * scale)
    kb = clip.set_duration(duration)
    kb = kb.resize(lambda t: size_at(t))
    kb = kb.set_pos(lambda t: ("center", int((H - kb.h(t)) / 2) + int((math.sin(t * 0.6)) * H * 0.02)))
    kb = kb.fadein(fade).fadeout(fade)
    final = kb.on_color(size=(W, H), color=(0, 0, 0))
    return final

# ----------------- caption rendering helpers (kept) -----------------
def render_hindi_text_image(text: str, font_path: str, font_size: int, color: str, max_width: int = 800, shadow: bool = False) -> str:
    """
    Render Hindi text into a temporary PNG and return its path.
    - Splits into lines so each line <= max_width (measured with the provided font).
    - If shadow=True: draws a soft shadow (blur applied later).
    - Otherwise draws a simple black outline for better readability on busy images.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # helper draw to measure text
    draw_tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw_tmp.textbbox((0, 0), test, font=font)
        wbox = bbox[2] - bbox[0]
        if wbox > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)

    # layout
    line_height = int(font_size * 1.15)
    img_h = line_height * max(1, len(lines)) + 20
    img_w = max_width + 40
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 10

    # convert hex color to RGB tuple (fall back to a warm yellow)
    try:
        hh = color.lstrip("#")
        rgb = tuple(int(hh[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        rgb = (255, 213, 79)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        wbox = bbox[2] - bbox[0]
        x = (img_w - wbox) // 2

        if shadow:
            # draw shadow slightly offset, then main text (later blur applied if requested)
            shadow_fill = (0, 0, 0, 180)  # semi-transparent black
            draw.text((x + 4, y + 4), line, font=font, fill=shadow_fill)
            draw.text((x, y), line, font=font, fill=rgb)
        else:
            # outline: draw black shadows around then main text on top
            outline_offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
            for ox, oy in outline_offsets:
                draw.text((x + ox, y + oy), line, font=font, fill=(0, 0, 0))
            draw.text((x, y), line, font=font, fill=rgb)

        y += line_height

    if shadow:
        # apply subtle blur to shadow/whole image to soften shadow if requested
        try:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        except Exception:
            pass

    fd, tmp_path = mkstemp(suffix=".png")
    os.close(fd)
    # ensure PNG saved with transparency
    img.save(tmp_path, format="PNG")
    return tmp_path

def make_typing_clip(text, font_path, font_size, color, shadow_color, max_width, duration, position, reveal_fps=15, max_words=6):
    """
    Typing effect but showing at most `max_words` words on-screen at any time.
    - `position` is kept for compatibility but clips will be placed at ("center","center") by caller for vertical centering
    - `max_words` defaults to 6 (you can pass different value from config)
    """
    reveal_fps = max(6, int(reveal_fps))
    frames = max(1, int(duration * reveal_fps))
    if frames > 240:
        frames = 240

    L = len(text)
    imgs = []
    shadow_imgs = []

    for i in range(frames):
        frac = (i + 1) / frames
        frac_ease = 1 - (1 - frac) * (1 - frac)
        nch = max(1, min(L, int(round(L * frac_ease))))
        prefix = text[:nch].strip()

        # limit visible words to last `max_words`
        words = prefix.split()
        if len(words) > max_words:
            display = " ".join(words[-max_words:])
        else:
            display = prefix

        # create text image for the display string
        txt_path = render_hindi_text_image(display, font_path, font_size, color, max_width=max_width, shadow=False)
        sh_path = render_hindi_text_image(display, font_path, font_size, shadow_color, max_width=max_width, shadow=True)
        imgs.append(txt_path)
        shadow_imgs.append(sh_path)

    per_dur = duration / frames if frames > 0 else duration
    small_clips = []
    for sh, tx in zip(shadow_imgs, imgs):
        sh_clip = ImageClip(sh).set_duration(per_dur)
        tx_clip = ImageClip(tx).set_duration(per_dur)
        # composite shadow under text
        comp = CompositeVideoClip([sh_clip.set_position(("center","center")), tx_clip.set_position(("center","center"))], size=(tx_clip.w, tx_clip.h))
        comp = comp.set_duration(per_dur)
        small_clips.append(comp)

    if len(small_clips) == 1:
        final = small_clips[0]
    else:
        final = concatenate_videoclips(small_clips, method="compose")

    # Attach temp png list for cleanup
    final._tmp_pngs = imgs + shadow_imgs
    return final

# Add after render_hindi_text_image / make_typing_clip helper functions

from moviepy.video.fx.resize import resize

def make_text_effect_clip(
    text: str,
    font_path: str,
    font_size: int,
    color: str,
    shadow_color: str,
    max_width: int,
    duration: float,
    effect: str = "fade_move_up",
    reveal_fps: int = 15,
    max_words: int = 6,
):
    """
    Create a MoviePy clip for the given text using one of 5 effects:
    - fade_move_up
    - word_by_word
    - subtitle_center
    - zoom_in
    - glow
    Returns a video clip sized to its rendered image (caller will position it).
    """

    # render full text image (we reuse existing helper)
    txt_png = render_hindi_text_image(text, font_path, font_size, color, max_width=max_width, shadow=False)

    # load base clip
    base = ImageClip(txt_png).set_duration(duration)

    # effect: fade_move_up -> fade in + slight translate up
    if effect == "fade_move_up":
        def pos_fn(t):
            # start a bit lower, move up by ~10% of height
            h = base.h
            y0 = int((H - base.h) / 2 + h * 0.06) if 'H' in globals() else ("center")
            # For simplicity, center horizontally, vertical animated by lambda using returned tuple
            # We'll implement vertical shift relative to clip height
            frac = t / max(1e-6, duration)
            dy = int((1.0 - frac) * (base.h * 0.06))
            return ("center", int((H - base.h) / 2 + dy)) if 'H' in globals() else ("center", "center")
        eff = base.set_position(lambda t: ("center", int((0) if True else 0))).fadein(0.5)
        # simpler stable: fadein + small upward resize/position using crossfade
        eff = base.fadein(min(0.5, duration * 0.2)).set_start(0)
        return eff

    # effect: word_by_word -> reveal per word (fade each word)
    if effect == "word_by_word":
        words = text.split()
        # Build short clips for chunks, keep it light: reveal groups of words across duration
        n_steps = min(len(words), max(1, int(duration * (reveal_fps/2))))
        group_size = max(1, math.ceil(len(words) / n_steps))
        parts = []
        for i in range(0, len(words), group_size):
            part_text = " ".join(words[: i + group_size])
            png = render_hindi_text_image(part_text, font_path, font_size, color, max_width=max_width, shadow=False)
            c = ImageClip(png).set_duration(duration / math.ceil(len(words)/group_size))
            c = c.fadein(0.12)
            parts.append(c)
        if parts:
            return concatenate_videoclips(parts, method="compose")
        else:
            return base

    # effect: subtitle_center -> static centered subtitle with small fade
    if effect == "subtitle_center":
        c = base.set_position(("center", "bottom")).fadein(0.25)
        return c

    # effect: zoom_in -> slight zoom in on text
    if effect == "zoom_in":
        # starting slightly smaller, slowly scale to 1.0
        def resize_fn(t):
            start = 0.92
            end = 1.0
            frac = t / max(1e-6, duration)
            scale = start + (end - start) * frac
            return scale
        z = base.resize(lambda t: resize_fn(t))
        return z

    # effect: glow -> duplicate + blur underneath, then show
    if effect == "glow":
        from PIL import ImageFilter
        # create a blurred version for glow
        try:
            glow_png = render_hindi_text_image(text, font_path, font_size, color, max_width=max_width, shadow=True)
            glow_clip = ImageClip(glow_png).set_duration(duration).set_opacity(0.9)
            top_clip = base.set_duration(duration)
            comp = CompositeVideoClip([glow_clip, top_clip], size=(base.w, base.h))
            return comp
        except Exception:
            return base

    # default fallback
    return base

# ----------------- choose_random_bg helper (missing earlier) -----------------
def choose_random_bg(bg_folder: Path):
    if not bg_folder.exists():
        raise FileNotFoundError(f"BG folder not found: {bg_folder}")
    candidates = []
    for ext in ("wav","mp3","m4a","flac","ogg"):
        candidates += list(bg_folder.glob(f"*.{ext}"))
    if not candidates:
        raise FileNotFoundError("No background music files found in bg_folder")
    return random.choice(candidates)
    
def create_vignette(w: int, h: int, strength: float = 0.5) -> str:
    """
    Create a radial vignette PNG (temporary file) sized w x h.
    Returns path to the png file (string).
    Strength: 0.0 (no vignette) .. 1.0+ (strong)
    """
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("PIL not available for create_vignette")

    # clamp strength
    s = float(strength)
    if s < 0:
        s = 0.0
    if s > 3.0:
        s = 3.0

    # create alpha mask
    mask = Image.new("L", (w, h), 0)
    px = mask.load()
    # center coords
    cx = w / 2.0
    cy = h / 2.0
    max_dist = math.sqrt((cx) ** 2 + (cy) ** 2)
    # fill mask with radial values scaled by strength
    for y in range(h):
        for x in range(w):
            dx = (x - cx)
            dy = (y - cy)
            d = math.sqrt(dx * dx + dy * dy) / max_dist  # 0..1
            # compute value: 0 in center -> 255 at edges (adjust with strength)
            val = int(255 * min(1.0, (d ** (1.0 + s))))
            px[x, y] = val

    # make an RGBA image (transparent center, dark edges)
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    rgba.putalpha(mask)

    fd, pth = mkstemp(suffix=".png")
    os.close(fd)
    rgba.save(pth)
    return pth

# ----------------- New video creation function (replaces old image->video section) -----------------
def create_video_from_segments_and_images(segments, voice_wav_path: Path, srt_path: Path, out_video_path: Path, cfg: dict):
    """
    Uses cinematic KenBurns clips from assets/images and overlays per-segment text effects (based on cfg).
    Ensures video duration matches audio length.
    """
    movie_cfg = cfg.get("movie", {})
    fps = int(movie_cfg.get("fps", 30))
    orientation = movie_cfg.get("orientation", "vertical")
    if orientation == "horizontal":
        W = int(movie_cfg.get("horizontal", {}).get("width", 1280))
        H = int(movie_cfg.get("horizontal", {}).get("height", 720))
    else:
        W = int(movie_cfg.get("vertical", {}).get("width", 720))
        H = int(movie_cfg.get("vertical", {}).get("height", 1280))

    font_path = movie_cfg.get("font_path")
    font_size = int(movie_cfg.get("font_size", 56))
    caption_color = movie_cfg.get("caption_color", "#FFD54F")
    caption_shadow = movie_cfg.get("caption_shadow", "black")
    margin_bottom = int(movie_cfg.get("caption_margin_bottom", 170))
    max_width = W - int(movie_cfg.get("max_width_padding", 160))
    zoom_strength = float(movie_cfg.get("zoom_strength", 0.04))
    vignette_strength = float(movie_cfg.get("vignette_strength", 0.18))
    reveal_fps = int(movie_cfg.get("typing_fps", 24))
    vign_path = create_vignette(W, H, strength=vignette_strength)

    # images
    images_folder = Path("assets/images")
    imgs_all = sorted([str(p) for p in images_folder.glob("*.*")]) if images_folder.exists() else []
    if not imgs_all:
        LOG.warning("No images found in assets/images - will use colored backgrounds")

    # compute total audio duration from voice_wav_path
    try:
        info = sf.info(str(voice_wav_path))
        total_time = info.frames / info.samplerate
    except Exception:
        try:
            total_time = AudioSegment.from_file(str(voice_wav_path)).duration_seconds
        except Exception:
            total_time = sum([ed - st for st, ed, _ in segments])

    # determine per-segment images selection: one clip per segment (keeps subtitle sync)
    sel_images = []
    if imgs_all:
        image_random = bool(cfg.get("image_random", False))
        if image_random:
            for i in range(len(segments)):
                sel_images.append(random.choice(imgs_all))
        else:
            for i in range(len(segments)):
                candidate = imgs_all[i % len(imgs_all)]
                if sel_images and candidate == sel_images[-1] and len(imgs_all) > 1:
                    candidate = imgs_all[(i + 1) % len(imgs_all)]
                sel_images.append(candidate)
    else:
        sel_images = [None] * len(segments)

    clips = []
    temp_img_files = []
    cinematic_cfg = movie_cfg.get("cinematic", {}) if movie_cfg.get("cinematic") else {}
    enable_vignette = bool(cinematic_cfg.get("vignette", True))
    enable_grain = bool(cinematic_cfg.get("grain", True))
    enable_letterbox = bool(cinematic_cfg.get("letterbox", True))

    # caption position mode from config: "bottom" (default) or "center"
    caption_position = (movie_cfg.get("caption_position") or "bottom").lower()
    if caption_position not in ("bottom", "center"):
        caption_position = "bottom"

    for idx, (start, end, text) in enumerate(segments):
        dur = max(0.8, end - start)
        img_path = sel_images[idx]
        if img_path:
            clip = make_kenburns_clip(
                img_path,
                (W, H),
                duration=dur,
                zoom_start=1.0,
                zoom_end=1.06,
                enable_vignette=enable_vignette,
                enable_grain=enable_grain,
            )
        else:
            clip = ColorClip(size=(W, H), color=(18, 18, 28), duration=dur)

        typing_clip = None
        if bool(cfg.get("caption_on", True)):
            caption_max_words = int(cfg.get("movie", {}).get("caption_max_words", 6))

            # pick text effect (fallback logic)
            text_effect = cfg.get("text_effect") or None
            if not text_effect:
                te_map = cfg.get("text_effects", {}) or {}
                for k in ["fade_move_up", "word_by_word", "subtitle_center", "zoom_in", "glow"]:
                    if te_map.get(k):
                        text_effect = k
                        break
            if not text_effect:
                text_effect = "fade_move_up"

            typing_clip = make_text_effect_clip(
                text=text,
                font_path=font_path,
                font_size=font_size,
                color=caption_color,
                shadow_color=caption_shadow,
                max_width=max_width,
                duration=dur,
                effect=text_effect,
                reveal_fps=reveal_fps,
                max_words=caption_max_words,
            )

            # determine available vertical space and clip height safely
            top_bar_h = 90 if enable_letterbox else 0
            bot_bar_h = 90 if enable_letterbox else 0
            bottom_margin_px = margin_bottom
            available_height = H - top_bar_h - bot_bar_h - bottom_margin_px - 20
            if available_height < 40:
                available_height = int(H * 0.4)

            try:
                clip_h = int(getattr(typing_clip, "h", getattr(typing_clip, "size", [None, None])[1] or 0))
            except Exception:
                clip_h = 0

            # if too tall, scale down to fit
            if clip_h and clip_h > available_height:
                try:
                    scale = float(available_height) / float(clip_h)
                    typing_clip = typing_clip.resize(scale)
                except Exception:
                    try:
                        typing_clip = typing_clip.resize(width=int(W * 0.9))
                    except Exception:
                        pass
                try:
                    clip_h = int(getattr(typing_clip, "h", getattr(typing_clip, "size", [None, None])[1] or 0))
                except Exception:
                    clip_h = 0

            # set position based on caption_position
            if caption_position == "center":
                try:
                    typing_clip = typing_clip.set_position(("center", "center"))
                except Exception:
                    typing_clip = typing_clip.set_position(("center", int(H * 0.5)))
            else:
                # bottom placement: ensure bottom of clip remains above margin
                def _pos_fn(t):
                    try:
                        ch = clip_h or (typing_clip.h if hasattr(typing_clip, "h") else 0)
                        if ch:
                            y_center = int(H - bottom_margin_px - (ch / 2))
                            min_y = int(top_bar_h + (ch / 2) + 6)
                            if y_center < min_y:
                                y_center = min_y
                            return ("center", y_center)
                        else:
                            return ("center", int(H * 0.56))
                    except Exception:
                        return ("center", int(H * 0.56))

                try:
                    typing_clip = typing_clip.set_position(_pos_fn)
                except Exception:
                    typing_clip = typing_clip.set_position(("center", int(H * 0.56)))

            # vignette + top/bottom bars (these are composited below)
            vign_clip = ImageClip(vign_path).set_duration(dur).set_position(("center", "center")).set_opacity(0.8)
            top_bar = ColorClip(size=(W, 90), color=(0, 0, 0)).set_duration(dur).set_position(("center", "top"))
            bot_bar = ColorClip(size=(W, 90), color=(0, 0, 0)).set_duration(dur).set_position(("center", "bottom"))

            comp = CompositeVideoClip(
                [
                    clip,
                    vign_clip,
                    typing_clip,
                    top_bar,
                    bot_bar,
                ],
                size=(W, H),
            ).set_duration(dur)
        else:
            comp = clip

        # optional top/bottom bars for cinematic look (add extra thin bars if enabled)
        if enable_letterbox:
            top_bar = ColorClip(size=(W, 80), color=(0, 0, 0)).set_duration(dur).set_position(("center", "top"))
            bot_bar = ColorClip(size=(W, 80), color=(0, 0, 0)).set_duration(dur).set_position(("center", "bottom"))
            comp = CompositeVideoClip([comp, top_bar, bot_bar], size=(W, H)).set_duration(dur)

        comp = comp.crossfadein(0.12).crossfadeout(0.12)
        clips.append(comp)

        if typing_clip is not None and hasattr(typing_clip, "_tmp_pngs"):
            temp_img_files.extend(getattr(typing_clip, "_tmp_pngs", []))

    if not clips:
        raise RuntimeError("No video clips to compose")

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = final_clip.set_fps(fps)

    # attach audio (final_out is expected as mixed voice+bg)
    speech_audio = AudioFileClip(str(voice_wav_path))
    final_clip = final_clip.set_audio(speech_audio)

    LOG.info("Exporting video (CPU may take a few minutes)...")
    ffmpeg_params = ["-crf", "18", "-pix_fmt", "yuv420p"]
    final_clip.write_videofile(
        str(out_video_path),
        codec="libx264",
        audio_codec="aac",
        threads=int(cfg.get("runtime", {}).get("moviepy_threads", 2)),
        fps=fps,
        preset=cfg.get("movie", {}).get("preset", "medium"),
        bitrate=cfg.get("movie", {}).get("bitrate", "3500k"),
        ffmpeg_params=ffmpeg_params,
    )
    final_clip.close()

    # cleanup tmp images
    for p in set(temp_img_files + _TEMP_IMAGES):
        try:
            os.remove(p)
        except Exception:
            pass
    _TEMP_IMAGES.clear()
    return out_video_path


# ----------------- (rest of pipeline remains mostly same) main pipeline -----------------
# --- Put these helpers above run_pipeline (e.g. after clean_text_pipeline) ---

# ----------------- Strict XTTS cleaner (paste after clean_text_pipeline) -----------------


# Strict XTTS cleaner (enhanced with 16/18-window rule)
import re

_DEVANAGARI_RE = r"\u0900-\u097F"

def strict_xtts_clean(text: str) -> str:
    """
    Strict cleaner implementing required rules:
    - Keep only Devanagari block, spaces, and allowed symbols: ';' and '।'
    - Merge standalone symbols into previous word (attach to previous word)
    - No two symbols adjacent; if conflict keep the newer symbol
    - Window logic:
        * First window = 16 words; subsequent windows = 18 words.
        * For each window: if any token contains '।' -> do not insert in that window.
          Otherwise insert '।' attached to the 8th word (or last word if fewer than 8).
    - Ensure final script ends with '।' (replace trailing ';' with '।' if needed).
    """
    if not text:
        return ""

    # normalize unicode
    try:
        import unicodedata
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass

    # normalize whitespace/newlines -> spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Remove all chars except Devanagari, whitespace, danda and semicolon
    # NOTE: we intentionally drop question marks and other punctuation
    text = re.sub(rf"[^{_DEVANAGARI_RE}\s।;]", "", text)

    # collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # split into raw tokens (space-separated)
    raw_tokens = [t for t in text.split(" ") if t != ""]

    # merge standalone symbols into previous token (attach them)
    tokens = []
    for tok in raw_tokens:
        if tok in ("।", ";"):
            if tokens:
                tokens[-1] = tokens[-1] + tok
            else:
                # leading symbol without previous word -> discard
                continue
        else:
            # If token contains spaces of symbol internally (shouldn't after split), keep as-is
            tokens.append(tok)

    # helper: does window contain any token with danda '।'?
    def window_has_danda(toks):
        for tt in toks:
            if "।" in tt:
                return True
        return False

    out = []
    i = 0
    n = len(tokens)
    first_window = True

    while i < n:
        window_len = 16 if first_window else 18
        window = tokens[i: min(n, i + window_len)]

        if window_has_danda(window):
            # find first token in window with danda and emit up to and including it
            found_idx = None
            for k, tt in enumerate(window):
                if "।" in tt:
                    found_idx = k
                    break
            if found_idx is None:
                # defensive fallback - append one token
                out.append(tokens[i])
                i += 1
            else:
                for k in range(0, found_idx + 1):
                    tok = tokens[i + k]
                    # ensure no adjacent symbols: if previous endswith symbol and current also has symbol, prefer current
                    if out and (out[-1].endswith("।") or out[-1].endswith(";")) and (tok.endswith("।") or tok.endswith(";")):
                        out[-1] = out[-1].rstrip("।;")
                    out.append(tok)
                i = i + found_idx + 1
        else:
            # no danda in window -> insert danda after 8 words (or at end if fewer)
            insert_offset = 8 - 1  # zero-based index for 8th word
            insert_idx = min(len(window) - 1, insert_offset)
            # append tokens before insert_idx
            for k in range(0, insert_idx):
                out.append(tokens[i + k])
            # handle insert token: attach danda if not present
            tok = tokens[i + insert_idx]
            if tok.endswith("।") or tok.endswith(";"):
                # it already has symbol -> append as-is but ensure previous isn't symboled
                if out and (out[-1].endswith("।") or out[-1].endswith(";")):
                    out[-1] = out[-1].rstrip("।;")
                out.append(tok)
            else:
                # attach danda to this token; remove previous trailing symbol if any
                if out and (out[-1].endswith("।") or out[-1].endswith(";")):
                    out[-1] = out[-1].rstrip("।;")
                out.append(tok + "।")
            # advance to token after insert_idx
            i = i + insert_idx + 1

        first_window = False

    # final pass to ensure no two symbols adjacent across tokens:
    cleaned_tokens = []
    for tok in out:
        if not cleaned_tokens:
            cleaned_tokens.append(tok)
            continue
        prev = cleaned_tokens[-1]
        if (prev.endswith("।") or prev.endswith(";")) and (tok.endswith("।") or tok.endswith(";")):
            # prefer current token's symbol: strip symbol from previous then append current
            cleaned_tokens[-1] = prev.rstrip("।;")
            cleaned_tokens.append(tok)
        else:
            cleaned_tokens.append(tok)

    cleaned = " ".join(cleaned_tokens).strip()

    # ensure last char is danda '।' (replace trailing ';' with '।', or append if missing)
    if cleaned:
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1] + "।"
        elif not cleaned.endswith("।"):
            cleaned = cleaned + "।"

    # collapse spaces defensive
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def ensure_trailing_danda(text: str) -> str:
    """
    Ensure the script text ends with a Devanagari danda '।' attached to the last word,
    unless last punctuation is '?' or ';' in which case we do not append।
    """
    if text is None:
        return ""
    t = text.rstrip()
    if not t:
        return t
    last_char = t[-1]
    if last_char == '।':
        return t
    if last_char in ('?', ';'):
        return t
    # Otherwise append danda (no extra space)
    return t + "।"

# --- Replace your existing run_pipeline with this corrected version ---
def run_pipeline(control_path: Path, test_sample: Path = None):
    LOG.info("Loading control: %s", control_path)
    cfg = load_control(control_path)
    TEMP_DIR = cfg["runtime"].get("temp_dir") or gettempdir()
    cfg["runtime"]["temp_dir"] = TEMP_DIR
    try:
        torch_threads = int(cfg["runtime"].get("torch_num_threads", 2))
    except Exception:
        torch_threads = 2
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    model_name = cfg.get("model_name", DEFAULT_CONTROL["model_name"])
    local_model_path = Path(cfg.get("local_xtts_path") or "") if cfg.get("local_xtts_path") else None
    control_speaker = cfg.get("speaker") or None
    use_speaker_wav = bool(cfg.get("use_speaker_wav", True))
    speaker_wav_path_cfg = Path(cfg.get("speaker_wav_path")) if cfg.get("speaker_wav_path") else None
    force_single = cfg.get("force_single_speaker", False)
    fallback_single = cfg.get("fallback_single_speaker_model", None)

    voices_folder = Path(cfg.get("voices_folder", "voices"))
    bg_folder = Path(cfg.get("bg_music_folder", cfg.get("bg", {}).get("bg_music_folder", "bg_music")))
    scripts_folder = Path(cfg.get("scripts_folder", "scripts"))
    input_filename = cfg.get("input_filename", "input.txt")

    if test_sample:
        LOG.info("Test sample mode: using %s", test_sample)
        raw_text = Path(test_sample).read_text(encoding="utf8").strip()
    else:
        txt_path = scripts_folder / input_filename
        if not txt_path.exists():
            raise FileNotFoundError(f"Input script not found: {txt_path}")
        raw_text = txt_path.read_text(encoding="utf8").strip()

    LOG.info("Original script length: %d chars", len(raw_text))

    # If you want the strict XTTS-only cleaner, enable "use_strict_cleaner": true in control.json (root).
    if cfg.get("use_strict_cleaner", False):
        # strict cleaning per user's prompt (removes English, banned punctuation, inserts । every 8 words)
        cleaned_text = strict_xtts_clean(raw_text)
        LOG.info("Applied strict XTTS cleaner. Result length: %d chars", len(cleaned_text))
    else:
        if cfg.get("normalize_hindi", True):
            opts = cfg.get("normalizer_options", {}) or {}
            cleaned_text, stats = clean_text_pipeline(
                raw_text,
                ellipsis_style=opts.get("ellipsis_style", "ellipsis"),
                use_langtool=opts.get("use_langtool_if_available", False),
            )
            LOG.info("Cleaned text length: %d chars; stats: %s", len(cleaned_text), dict(stats))
        else:
            cleaned_text = raw_text

        # Ensure script ends with a danda '।' attached to last word (user request)
        cleaned_text = ensure_trailing_danda(cleaned_text)

    out_root = Path(cfg.get("output_folder", "output"))
    out_dir = make_timestamped_output(out_root)
    try:
        shutil.copytree("assets", out_dir / "assets", dirs_exist_ok=True)
    except Exception:
        pass

    selected_sample, selected_voice_params = select_voice_sample_from_config(cfg, voices_folder)
    if selected_sample:
        sample_path = selected_sample
    else:
        sample_path = speaker_wav_path_cfg if (speaker_wav_path_cfg and speaker_wav_path_cfg.exists()) else None
        if not sample_path:
            LOG.warning("No voice sample found; model must accept 'speaker' or be single-speaker.")

    if local_model_path and local_model_path.exists():
        model_source = str(local_model_path)
        LOG.info("Using local XTTS model at %s", model_source)
    else:
        model_source = model_name
        LOG.info("Using model name (remote/cached): %s", model_source)

    try:
        LOG.info("Loading TTS model (cpu)...")
        tts = TTS(model_source, progress_bar=True, gpu=False)
    except Exception as e:
        LOG.error("Failed to load TTS model '%s': %s", model_source, e)
        raise

    sentence_seps = re.split(r'(?<=[।\?…])\s+', cleaned_text)
    sentence_seps = [s.strip() for s in sentence_seps if s.strip()]
    if not sentence_seps:
        sentence_seps = [cleaned_text]

    tmp_folder = out_dir / "tmp_segments"
    ensure_dirs(tmp_folder)
    segments = []
    total_time = 0.0

    synth_kwargs_cfg = cfg.get("synth_kwargs", {}) or {}
    if control_speaker:
        LOG.info("Control provided speaker id: %s - will try using it first for all segments.", control_speaker)

    for idx, sent in enumerate(sentence_seps):
        seg_wav = tmp_folder / f"seg_{idx:03d}.wav"
        try:
            if control_speaker:
                try:
                    call_kwargs = _filter_kwargs_for_callable(tts.tts_to_file, synth_kwargs_cfg)
                    LOG.info("Synthesizing (control speaker) sentence %d ...", idx)
                    tts.tts_to_file(text=sent, speaker=str(control_speaker), file_path=str(seg_wav), language="hi", **call_kwargs)
                except Exception as e:
                    LOG.warning("Direct 'speaker' call failed (%s). Falling back to robust synth function.", e)
                    synth_sentence_to_file(tts, sent, seg_wav, speaker_wav=(sample_path if not force_single else None), synth_kwargs=synth_kwargs_cfg, language="hi")
            else:
                synth_sentence_to_file(tts, sent, seg_wav, speaker_wav=(sample_path if not force_single else None), synth_kwargs=synth_kwargs_cfg, language="hi")
        except Exception as e:
            LOG.error("Synthesis failed for sentence %d: %s", idx, e)
            if not force_single and fallback_single:
                try:
                    LOG.info("Retrying with fallback single-speaker model: %s", fallback_single)
                    tts = TTS(fallback_single, progress_bar=True, gpu=False)
                    synth_sentence_to_file(tts, sent, seg_wav, speaker_wav=None, synth_kwargs={})
                except Exception as e2:
                    LOG.exception("Fallback failed: %s", e2)
                    raise RuntimeError(f"Synthesis ultimately failed: {e2}") from e2
            else:
                raise

        try:
            info = sf.info(str(seg_wav))
            dur = info.frames / info.samplerate
        except Exception:
            try:
                dur = len(AudioSegment.from_wav(str(seg_wav))) / 1000.0
            except Exception:
                dur = 0.0
        start = total_time
        end = total_time + dur
        segments.append((start, end, sent))
        total_time += dur
        LOG.info("Synthesized segment %d dur=%.3fs", idx, dur)

    voice_concat = AudioSegment.silent(duration=0)
    for idx in range(len(sentence_seps)):
        p = tmp_folder / f"seg_{idx:03d}.wav"
        voice_concat += AudioSegment.from_wav(str(p))

    voice_raw = out_dir / "voice_only_raw.wav"
    voice_concat.export(str(voice_raw), format="wav")
    LOG.info("Exported voice-only raw -> %s", voice_raw)

    sampling_rate = cfg.get("tts", {}).get("sampling_rate", 24000)
    voice_proc = out_dir / "voice_only.wav"
    try:
        seg = AudioSegment.from_file(str(voice_raw))
        try:
            seg = seg.normalize()
        except Exception:
            pass
        seg += AudioSegment.silent(duration=150)
        seg = seg.set_frame_rate(sampling_rate)
        seg.export(str(voice_proc), format="wav")
        LOG.info("Postprocessed voice written -> %s", voice_proc)
    except Exception as e:
        LOG.warning("Postprocess failed: %s - copying raw", e)
        shutil.copyfile(str(voice_raw), str(voice_proc))

    # BG music selection and processing (robust)
    try:
        bg_src = None
        try:
            bg_src = choose_random_bg(Path(bg_folder))
        except Exception:
            # fallback: try assets/music
            cand = list(Path("assets/music").glob("*.*")) if Path("assets/music").exists() else []
            if cand:
                bg_src = random.choice(cand)
        if bg_src:
            LOG.info("Selected BG: %s", bg_src)
            bg_cfg = cfg.get("bg", {})
            bg_seg = prepare_bg_music_cinematic(bg_src,
                                               target_sr=sampling_rate,
                                               target_channels=1,
                                               target_dur_ms=int(total_time * 1000),
                                               reduce_db_base=int(bg_cfg.get("bg_reduce_db", 14)),
                                               lowpass_hz=int(bg_cfg.get("bg_lowpass_hz", 9000)),
                                               slow_tempo=float(bg_cfg.get("bg_slow_tempo", 0.97)),
                                               use_sox_for_tempo=bool(bg_cfg.get("use_sox_for_tempo", True)))
            bg_out = out_dir / "bg_music.wav"
            bg_seg.export(str(bg_out), format="wav")
            LOG.info("BG processed -> %s", bg_out)
        else:
            raise FileNotFoundError("No BG found")
    except Exception as e:
        LOG.warning("BG selection failed: %s - using silence", e)
        bg_seg = AudioSegment.silent(duration=int(total_time * 1000))
        bg_out = out_dir / "bg_music.wav"
        bg_seg.export(str(bg_out), format="wav")

    duck_db = int(cfg.get("bg", {}).get("bg_duck_db", 10))
    duck_thresh = float(cfg.get("bg", {}).get("bg_duck_threshold_db", -42.0))
    duck_window = int(cfg.get("bg", {}).get("bg_duck_window_ms", 200))
    bg_ducked = duck_bg_under_voice(bg_seg, voice_concat, threshold_db=duck_thresh, reduction_db=duck_db, window_ms=duck_window)
    final_bg = bg_ducked - int(cfg.get("bg", {}).get("bg_reduce_db", 14))

    if len(final_bg) < len(voice_concat):
        final_bg = final_bg + AudioSegment.silent(duration=(len(voice_concat) - len(final_bg)))
    elif len(final_bg) > len(voice_concat):
        final_bg = final_bg[:len(voice_concat)]

    final_mix = voice_concat.overlay(final_bg)
    final_out = out_dir / "output.wav"
    final_mix.export(str(final_out), format="wav")
    LOG.info("Exported final mix -> %s", final_out)

    srt_path = out_dir / "subtitles.srt"
    write_srt(segments, srt_path)
    LOG.info("SRT -> %s", srt_path)

    # create video with cinematic images + captions (caption_on controls overlay)
    try:
        movie_cfg_local = cfg.get("movie", {})
        video_out = out_dir / "final.mp4"
        create_video_from_segments_and_images(segments, final_out, srt_path, video_out, cfg)
        LOG.info("Video created -> %s", video_out)
    except Exception as e:
        LOG.exception("Video creation failed: %s", e)
        video_out = None

    meta = {
        "model_used": model_source,
        "speaker_cfg": str(control_speaker),
        "sample_used": str(sample_path) if sample_path else None,
        "output_audio": str(final_out),
        "output_video": str(video_out) if video_out else None,
        "duration_sec": total_time,
        "segments": len(segments),
        "timestamp": datetime.now().isoformat()
    }
    try:
        (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
    except Exception:
        pass

    if cfg["runtime"].get("cleanup_tmp", True):
        shutil.rmtree(tmp_folder, ignore_errors=True)
    LOG.info("Run complete. Outputs in: %s", out_dir)
        # attempt to explicitly free large objects (tts model etc.)
    try:
        # delete local tts if present to free memory faster
        if 'tts' in locals():
            try:
                del tts
            except Exception:
                pass
    except Exception:
        pass

    # force garbage collection and final memory log
    try:
        gc.collect()
    except Exception:
        pass

    return out_dir
# ---------- BATCH PROCESSING HELPERS (PASTE AFTER run_pipeline) ----------
def _natural_sort_key(s: str):
    """
    Natural sort key (keeps numbers in names in numeric order).
    """
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_script_files(scripts_dir: Path) -> List[Path]:
    """
    Return a naturally-sorted list of .txt files from scripts_dir.
    """
    p = Path(scripts_dir)
    if not p.exists():
        return []
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    files.sort(key=lambda x: _natural_sort_key(str(x.name)))
    return files

def log_memory(tag: str = ""):
    """
    Log simple memory stats if psutil available, otherwise print basic info.
    """
    if psutil:
        vm = psutil.virtual_memory()
        print(f"[MEM] {tag} total={vm.total//1024**2}MB avail={vm.available//1024**2}MB used%={vm.percent}")
    else:
        print(f"[MEM] {tag} psutil not installed (skipping detailed memory log)")

def _safe_close_moviepy_clip(obj):
    """
    Try to close a moviepy clip or audio reader if it exposes close() or reader attributes.
    """
    try:
        if hasattr(obj, "close"):
            try:
                obj.close()
            except Exception:
                pass
        # moviepy AudioFileClip may have reader and audio attributes
        if hasattr(obj, "reader"):
            try:
                r = getattr(obj, "reader")
                if hasattr(r, "close"):
                    r.close()
            except Exception:
                pass
        if hasattr(obj, "audio"):
            try:
                a = getattr(obj, "audio")
                if hasattr(a, "close"):
                    a.close()
            except Exception:
                pass
    except Exception:
        pass

def cleanup_post_run(objects: Optional[List] = None):
    """
    Best-effort cleanup after a pipeline run.
    - Close MoviePy clips/readers if provided.
    - Delete objects, force gc.collect(), try torch cleanup (safe).
    """
    # close provided objects
    if objects:
        for o in objects:
            try:
                _safe_close_moviepy_clip(o)
            except Exception:
                pass
            try:
                del o
            except Exception:
                pass

    # try to close any global MoviePy clips if present in globals (very conservative)
    for name in list(globals().keys()):
        try:
            val = globals().get(name)
            # Avoid deleting important globals (keep minimal)
            if hasattr(val, "__class__") and (val.__class__.__module__.startswith("moviepy") or name.lower().startswith("clip")):
                try:
                    _safe_close_moviepy_clip(val)
                except Exception:
                    pass
        except Exception:
            pass

    # call gc
    gc.collect()

    # torch cleanup (safe: wrapped)
    if torch:
        try:
            # Only attempt CPU-safe calls; torch.cuda.empty_cache is wrapped in try to avoid errors if CUDA not present
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # remove tracking tensors if any pinned to CUDA (best-effort)
            # note: we avoid calling any cuda-specific functions that assume devices present
        except Exception:
            pass

    # final memory log
    log_memory("[after cleanup]")

def generate_video_from_script(script_path: Path, control_path: Path):
    """
    Adapter wrapper that runs the full pipeline for a single script file.
    It uses run_pipeline(control_path, test_sample=script_path) so your existing logic is reused.
    Returns the output directory Path on success, or None on failure.
    """
    try:
        script_path = Path(script_path)
        if not script_path.exists():
            print(f"[BATCH] Script not found: {script_path}")
            return None
        print(f"[BATCH] Starting pipeline for: {script_path.name}")
        log_memory(f"[before run {script_path.name}]")

        # run your existing pipeline using the test_sample hook
        out_dir = run_pipeline(control_path, test_sample=script_path)

        log_memory(f"[completed run {script_path.name}]")
        # Attempt cleanup (best-effort)
        cleanup_post_run(objects=None)
        return out_dir
    except Exception as e:
        print(f"[BATCH] Error processing {script_path}: {e}")
        try:
            cleanup_post_run(objects=None)
        except Exception:
            pass
        return None

def process_all_scripts(scripts_dir: Path, control_path: Path, stop_on_error: bool = False):
    """
    Iterate all .txt files in scripts_dir and generate a video for each, sequentially.
    - Processes files in natural sort order
    - Uses generate_video_from_script() (adapter) for each file
    - stop_on_error: if True, stop at first error
    """
    scripts = list_script_files(Path(scripts_dir))
    if not scripts:
        print(f"[BATCH] No .txt files found in {scripts_dir}")
        return

    print(f"[BATCH] Found {len(scripts)} scripts. Starting sequential processing...")
    for idx, s in enumerate(scripts, start=1):
        print(f"\n[BATCH] ({idx}/{len(scripts)}) -> {s.name}")
        out = generate_video_from_script(s, control_path)
        if out:
            print(f"[BATCH] Completed: {s.name} -> {out}")
        else:
            print(f"[BATCH] Failed: {s.name}")
            if stop_on_error:
                print("[BATCH] Stopping due to stop_on_error=True")
                break
        # tiny pause to give OS scheduler breathing room on low-RAM machines
        time.sleep(0.25)

    print("[BATCH] All done.")
# ---------- END BATCH HELPERS ----------


# ----------------- CLI -----------------
# ----------------- CLI (modified to support batch processing) -----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="control.json", help="Path to control.json")
    ap.add_argument("--test", action="store_true", help="Run built-in test using scripts/sample.txt")
    ap.add_argument("--test_sample", default=None, help="Path to a single sample txt file")
    ap.add_argument("--batch", action="store_true", help="Process all .txt files in scripts folder sequentially")
    ap.add_argument("--stop_on_error", action="store_true", help="Stop the batch on first error")
    args = ap.parse_args()
    cfg_path = Path(args.config)
        # --- Auto-dispatch: if no explicit mode and scripts folder has multiple txt files, run batch ---
    # load config early to get scripts_folder setting
    cfg = None
    try:
        if cfg_path.exists():
            cfg = load_control(cfg_path)
        else:
            # write default and exit (keeps previous behavior)
            write_default_control(cfg_path)
            LOG.info("Wrote default control.json - edit it and re-run")
            sys.exit(0)
    except Exception as e:
        LOG.warning("Could not read control.json early: %s", e)
        cfg = None

    # If user did NOT explicitly request --batch/--test/--test_sample, check scripts folder
    if not (args.batch or args.test or args.test_sample):
        scripts_folder = Path((cfg.get("scripts_folder") if cfg else "scripts"))
        scripts_list = []
        if scripts_folder.exists():
            scripts_list = [p for p in scripts_folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
        if scripts_list:
            # If multiple scripts OR at least one script and no input.txt present, run batch automatically
            input_path = Path(cfg.get("scripts_folder", "scripts")) / cfg.get("input_filename", "input.txt")
            if len(scripts_list) > 1 or not input_path.exists():
                LOG.info("Auto-detected %d scripts in %s — starting batch processing.", len(scripts_list), scripts_folder)
                process_all_scripts(scripts_folder, cfg_path, stop_on_error=args.stop_on_error)
                sys.exit(0)
    # If we get here, fall through to existing behavior (single-run/test/run_pipeline)

    if not cfg_path.exists():
        write_default_control(cfg_path)
        LOG.info("Wrote default control.json - edit it and re-run")
        sys.exit(0)
    try:
        if args.batch:
            # Use scripts folder from control.json (or default 'scripts')
            cfg = load_control(cfg_path)
            scripts_folder = Path(cfg.get("scripts_folder", "scripts"))
            process_all_scripts(scripts_folder, cfg_path, stop_on_error=args.stop_on_error)
        else:
            if args.test or args.test_sample:
                sample = Path(args.test_sample) if args.test_sample else Path("scripts/sample.txt")
                if not sample.exists():
                    LOG.error("Test sample not found: %s", sample)
                    sys.exit(2)
                run_pipeline(cfg_path, test_sample=sample)
            else:
                run_pipeline(cfg_path)
    except Exception as e:
        LOG.exception("Pipeline failed: %s", e)
        sys.exit(1)
