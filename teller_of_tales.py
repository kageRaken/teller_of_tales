
"""
teller_of_tales_local.py
~~~~~~~~~~~~~~~~~~~~~~~~

Turn a plain-text story into an illustrated, narrated video.

Workflow:
    1. Split the story into short fragments.
    2. Generate an image prompt for each fragment.
    3. Render an image with Stable-Diffusion (A1111) or Pollinations.
    4. Synthesize speech with Edge-TTS, ElevenLabs or Kokoro.
    5. Stitch everything into short clips and concatenate into a final MP4.

The script is **project-oriented**: place each story.txt in its own directory
under ``./projects/<project_name>/`` and run this file – everything else is
automatic.

Configuration lives in ``config.ini`` and ``characters_descriptions.ini``.
"""

# if first run then run installer 
# import nltk
# nltk.download()

from __future__ import annotations

import asyncio
import base64
import configparser
import gc
import io
import json
import multiprocessing
import subprocess
import os
import pathlib
import re
import shutil
import time
import sys
from datetime import datetime
from typing import (Dict, List, Tuple)

import edge_tts
import openai
import psutil
import requests
from concurrent.futures import ProcessPoolExecutor
from fake_useragent import UserAgent
from functools import lru_cache
from keybert import KeyBERT
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.fx.all import volumex
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from nltk.tokenize import sent_tokenize, word_tokenize
from ollama import ChatResponse, chat
from PIL import Image, PngImagePlugin

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



# ---------- MoviePy FFMPEG override ----------
import moviepy.config as mpy_cfg

mpy_cfg.change_settings({"FFMPEG_BINARY": "ffmpeg"})

# ---------- Configuration ----------
_CONFIG_PATH = pathlib.Path(__file__).with_name("config.ini")
config = configparser.ConfigParser()
config.read(_CONFIG_PATH, encoding="utf-8")

# GENERAL
DEBUG: bool = config["GENERAL"].getboolean("DEBUG", fallback=False)
SPEED_UP: bool = config["GENERAL"].getboolean("SPEED_UP", fallback=False)
FREE_SWAP_GB: int = int(config["GENERAL"]["FREE_SWAP"])
FPS: int = int(config["GENERAL"]["FPS"])
AUDIO_ONLY: bool = config["GENERAL"].getboolean("AUDIO_ONLY", fallback=False)
COMBINE_ALL_PROJECTS: bool = config["GENERAL"].getboolean("COMBINE_ALL_PROJECTS", fallback=False)


# TEXT
FRAGMENT_LENGTH: int = int(config["TEXT_FRAGMENT"]["FRAGMENT_LENGTH"])

# AUDIO
TTS_PROVIDER: str = config["AUDIO"]["TTS_PROVIDER"]
ELEVENLABS_VOICE_ID: str = config["AUDIO"]["ELEVENLABS_VOICE_ID"]
KOKORO_VOICE_ID: str = config["AUDIO"]["KOKORO_VOICE_ID"]
KOKORO_URL: str = config["AUDIO"]["KOKORO_URL"]
VOICE: str = config["AUDIO"]["VOICE"]
VOICE_SPEED: str = config["AUDIO"]["VOICE_SPEED"]
BG_MUSIC: bool = config["AUDIO"].getboolean("BG_MUSIC")
BG_MUSIC_PATH: pathlib.Path = pathlib.Path(__file__).parent / config["AUDIO"]["BG_MUSIC_PATH"]
MUSIC_VOLUME: float = float(config["AUDIO"]["MUSIC_VOLUME"])

# IMAGE PROMPTS
IMAGE_PROMPT_PROVIDER: str = config["IMAGE_PROMPT"]["IMAGE_PROMPT_PROVIDER"]
OLLAMA_MODEL: str = config["IMAGE_PROMPT"]["OLLAMA_MODEL"]

# STABLE DIFFUSION
POSITIVE_PREFIX: str = config["STABLE_DIFFUSION"]["positive_prompt_prefix"]
POSITIVE_SUFFIX: str = config["STABLE_DIFFUSION"]["positive_prompt_suffix"]
NEGATIVE_PROMPT: str = config["STABLE_DIFFUSION"]["negative_prompt"]
USE_SD_API: str = config["STABLE_DIFFUSION"]["USE_SD_VIA_API"]
SD_URL: str = config["STABLE_DIFFUSION"]["SD_URL"]
SEED: int = int(config["STABLE_DIFFUSION"]["seed"])
IMAGE_WIDTH: int = int(config["STABLE_DIFFUSION"]["image_width"])
IMAGE_HEIGHT: int = int(config["STABLE_DIFFUSION"]["image_height"])

USE_CHAR_DESC: bool = config["STABLE_DIFFUSION"].getboolean("USE_CHARACTERS_DESCRIPTIONS")
CHAR_DESC: Dict[str, str] = {}
if USE_CHAR_DESC:
    _CHAR_DESC_PATH = pathlib.Path(__file__).with_name("characters_descriptions.ini")
    if _CHAR_DESC_PATH.exists():
        _cd = configparser.ConfigParser()
        _cd.read(_CHAR_DESC_PATH, encoding="utf-8")
        CHAR_DESC = dict(_cd["CHARACTERS_DESCRIPTIONS"])

# API keys from environment
if TTS_PROVIDER == "elevenlabs":
    openai.api_key = os.environ["ELEVENLABS_API_KEY"]

if IMAGE_PROMPT_PROVIDER == "chatgpt":
    openai.api_key = os.environ["OPENAI_TOKEN"]

# ---------- Utilities ----------
_TIMESTAMP_FMT = "[%Y-%m-%d %H:%M:%S UTC]"


def _log(msg: str) -> None:
    """Print timestamped message when DEBUG=True."""
    if DEBUG:
        print(f"{datetime.utcnow().strftime(_TIMESTAMP_FMT)}  {msg}")


def _write_text(path: pathlib.Path, text: str) -> None:
    """Atomic write with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text(path: pathlib.Path) -> str:
    """Read UTF-8 file."""
    return path.read_text(encoding="utf-8")


# ---------- Text Processing ----------
def clean_text(text: str) -> str:
    """Normalize punctuation, quotes and dashes."""
    mapping = {
        "é": "e",
        ">": "",
        "<": "",
        "=": "",
        "#": "",
        "..": ".",
        "“": "",
        "”": "",
        "-": " ",
        "–": " ",
        "—": " ",
        "*": "",
        "_": "",
        "~": "",
        "XXXXXX": "",
        "xxxxx": "",
        ".....": ".",
        "....": ".",
        "...": ", ",
        "…": ", ",
        "\n\n\n": "\n",
        "\n\n": "\n",
    }
    for k, v in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(k, v)
    return text


def load_and_split_to_sentences(story_path: pathlib.Path) -> int:
    """
    Split *story.txt* into sentences and write into
    ``text/story_sentences/story_sentence{idx}.txt``.

    Returns the number of sentence files created.
    """
    raw = story_path.read_text(encoding="utf-8")
    raw = clean_text(raw)
    sentences = sent_tokenize(raw)

    # split long sentences at selected puntuations
    punctuation_list = [',', ';', ':']
    new_sentences: List[str] = []
    frag_len = 3*FRAGMENT_LENGTH
    for sent in sentences:
        words = sent.split()
        if len(words) <= FRAGMENT_LENGTH:
            new_sentences.append(sent)
        else:
            part = []
            for word in words:
                part.append(word)
                if word[-1] in punctuation_list and len(part) > frag_len:
                    new_sentences.append(' '.join(part))
                    part = []
            if part:
                new_sentences.append(" ".join(part))

    for idx, sent in enumerate(new_sentences):
        _write_text(story_path.parent / f"text/story_sentences/story_sentence{idx}.txt", sent)

    _log(f"Created {len(new_sentences)} sentence files.")
    return len(new_sentences)


def sentences_to_fragments(num_sentences: int, project_dir: pathlib.Path) -> int:
    """
    Group consecutive sentences into fragments of at least *FRAGMENT_LENGTH* words
    and write to ``text/story_fragments/``.
    """
    fragments: List[str] = []
    current_words: List[str] = []

    for i in range(num_sentences):
        sentence = _read_text(project_dir / f"text/story_sentences/story_sentence{i}.txt")
        current_words.extend(sentence.split())
        if len(current_words) > FRAGMENT_LENGTH:
            fragments.append(" ".join(current_words))
            current_words = []

    if current_words:
        fragments.append(" ".join(current_words))

    for idx, frag in enumerate(fragments):
        _write_text(project_dir / f"text/story_fragments/story_fragment{idx}.txt", frag)

    _log(f"Created {len(fragments)} fragment files.")
    return len(fragments)


# ---------- Image Prompt Generation ----------
def _unload_ollama() -> None:    
    if IMAGE_PROMPT_PROVIDER == "ollama":
        url = 'http://localhost:11434/api/generate'
        data = {'model': OLLAMA_MODEL, 'keep_alive': 0}
        response = requests.post(url, json=data)
        print(response.text)
        time.sleep(3)
        
        
def _reload_ollama() -> None:    
    if IMAGE_PROMPT_PROVIDER == "ollama":
        url = 'http://localhost:11434/api/generate'
        data = {'model': OLLAMA_MODEL, 'keep_alive': 1}
        response = requests.post(url, json=data)
        print(response.text)
        time.sleep(3)


def _find_characters(fragment: str) -> str:
    """Return comma-separated character descriptions if any character is mentioned."""
    for name, desc in CHAR_DESC.items():
        if re.search(rf"\b{name}\b", fragment, flags=re.IGNORECASE):
            return f"[[[ {desc} ]]], "
    return ""


@lru_cache(maxsize=1)
def _get_kw_model() -> KeyBERT:
    return KeyBERT("all-mpnet-base-v2")


def _keywords_fallback(fragment: str) -> str:
    """Use KeyBERT when LLM refuses prompt generation."""
    kw_model = _get_kw_model()
    ngram_range = (1, 8)
    keywords = kw_model.extract_keywords(
        fragment,
        keyphrase_ngram_range=ngram_range, 
        stop_words='english', 
        highlight=False,
        top_n=1
    )
    keywords_list = list(dict(keywords).keys())
    del kw_model
    del keywords
    # Force a garbage collection
    gc.collect()
    image_prompt = ', '.join(keywords_list)
    return image_prompt
    

def build_image_prompt(fragment: str) -> str:
    """Generate an SD prompt for the given fragment."""
    prompt_instruction = (
        "You are an expert prompt writer for Stable-Diffusion-XL. "
        f"Style context: {POSITIVE_SUFFIX}. "
        "Describe the scene in a single sentence, max 20 words. "
        "Do NOT include any explanations or quotes."
    )

    if IMAGE_PROMPT_PROVIDER == "chatgpt":
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"{prompt_instruction}\n{fragment}",
                max_tokens=40,
                temperature=0.9,
            )
            prompt = response.choices[0].text.strip()
                
        except Exception as e:
            _log(f"ChatGPT failed: {e}. Using KeyBERT fallback.")
            prompt = _keywords_fallback(fragment)

    elif IMAGE_PROMPT_PROVIDER == "ollama":
        try:
            resp: ChatResponse = chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": f"{prompt_instruction}\n{fragment}"}],
            )
            prompt = resp["message"]["content"].strip()
            _log(prompt)
                
        except Exception as e:
            _log(f"Ollama failed: {e}. Using KeyBERT fallback.")
            prompt = _keywords_fallback(fragment)

    else:
        prompt = _keywords_fallback(fragment)

    if any(x in prompt.lower() for x in ("i cannot", "?")):
        prompt = _keywords_fallback(fragment)

    if CHAR_DESC:
        prompt = _find_characters(fragment) + prompt

    return prompt


# ---------- Image Generation ---------- 
def _unload_sd() -> None:
    if USE_SD_API == "yes":
        response = requests.post(url=f"{SD_URL}/sdapi/v1/unload-checkpoint", json={})
        print(response.text)
        time.sleep(3)
    

def _reload_sd() -> None:
    if USE_SD_API == "yes":
        response = requests.post(url=f"{SD_URL}/sdapi/v1/reload-checkpoint", json={})
        print(response.text)
        time.sleep(3)


def _sd_api_payload(prompt: str) -> dict:
    """Return A1111 API payload for txt2img."""
    return {
        "prompt": f"{prompt}",
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": 20,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "seed": SEED,
        "guidance_scale": 4.0,
        "sampler_index": "Euler a",
    }


def generate_image(idx: int, project_dir: pathlib.Path) -> None:
    """Generate image for fragment *idx*."""
    prompt_path = project_dir / f"text/image_prompts/image_prompt{idx}.txt"
    image_path = project_dir / f"images/image{idx}.jpg"
    if image_path.exists():
        return

    prompt = _read_text(prompt_path)
    prompt = f"{POSITIVE_PREFIX} {prompt} {POSITIVE_SUFFIX}"
    
    _log(f"{idx} Loaded Prompt: {prompt}")
    do_it = True
    wait_time = 10
    
    while(do_it):
        try:
            if USE_SD_API == "yes":
                url = SD_URL
                payload = _sd_api_payload(prompt)
                option_payload = {
                    "sd_model_checkpoint": "aamXLAnimeMix_v10.safetensors",
                    "sd_vae": "sdxl_vae.safetensors",
                }
                requests.post(f"{url}/sdapi/v1/options", json=option_payload)
                r = requests.post(f"{url}/sdapi/v1/txt2img", json=payload).json()

                for b64 in r["images"]:
                    img = Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[0])))
                    info = PngImagePlugin.PngInfo()
                    info.add_text("parameters", r.get("info", ""))
                    img.save(image_path, pnginfo=info)

            elif USE_SD_API == "pollinations":
                ua = UserAgent()
                url = (
                    f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
                    f"?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}&nologo=true&model=flux&enhance=false"
                    f"&seed={time.time()}&negative=nsfw"
                )
                response = requests.get(url, headers={"User-Agent": ua.random}, timeout=60)
                _log(f"{idx}: {response.text[:100]}")
                if response.status_code == 200:
                    image = io.BytesIO(response.content)
                    img = Image.open(image)
                    img.save(image_path)
                else:
                    raise requests.exceptions.HTTPError(f'Failed to download the image. Status code: {response.status_code}')    
                
            else:
                pass
                
            do_it = False
            
        except Exception as e:   
            _log(f"Exception!!! {idx} \n{e} \nWaiting for {wait_time} seconds and trying again...")
            time.sleep(wait_time)



# ---------- TTS ----------
async def tts_edge(text: str, out: pathlib.Path) -> None:
    """Edge-TTS voice-over."""
    out.parent.mkdir(parents=True, exist_ok=True)
    com = edge_tts.Communicate(text, VOICE)
    await com.save(str(out))


def tts_elevenlabs(text: str, out: pathlib.Path) -> None:
    """ElevenLabs voice-over."""
    url = "https://api.elevenlabs.io/v1/user/subscription"
    
    headers = {
          "Accept": "audio/mpeg",
          "Content-Type": "application/json",
          "xi-api-key": ELEVENLABS_API_KEY
        }
        
    usage = requests.get(url, headers=headers).json()
    if usage["character_limit"] - usage["character_count"] < len(text)+1:
        raise RuntimeError(f"ElevenLabs character limit almost exceeded! Characters left: {usage['character_count']}")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    resp = requests.post(tts_url, json=payload, headers=headers)
    resp.raise_for_status()
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024):
            f.write(chunk)


def tts_kokoro(text: str, out: pathlib.Path) -> None:
    """Kokoro voice-over."""
    out.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.post(
        KOKORO_URL,
        json={
            "model": "kokoro",
            "input": text.lower(),
            "voice": KOKORO_VOICE_ID,
            "speed": VOICE_SPEED,
            "response_format": "wav",
            "stream": True,
        },
        stream=True,
    )
    resp.raise_for_status()
    with open(out, "wb") as f:
        shutil.copyfileobj(resp.raw, f)


# ---------- Video Assembly ----------
def create_video_clip(idx: int, project_dir: pathlib.Path) -> None:
    """Combine image/audio into a 2-second padded clip."""
    frag_path = project_dir / f"text/story_fragments/story_fragment{idx}.txt"
    img_path = project_dir / f"images/image{idx}.jpg"
    audio_wav = project_dir / f"audio/voiceover{idx}.wav"
    audio_mp3 = project_dir / f"audio/voiceover{idx}.mp3"

    audio_clip = AudioFileClip(str(audio_mp3 if audio_mp3.exists() else audio_wav))
    audio_clip = audio_clip.subclip(0, audio_clip.duration - 0.1)  # fix glitch
    audio_clip = audio_clip.audio_fadein(0.05).audio_fadeout(0.05)
    audio_clip = concatenate_audioclips(
        [
            AudioClip(lambda t: 0, duration=0.5),
            audio_clip,
            AudioClip(lambda t: 0, duration=0.5),
        ]
    )

    image_clip = ImageClip(str(img_path)).set_duration(audio_clip.duration)

    # # Text overlay
    # txt_clip = TextClip(
    #     _read_text(frag_path),
    #     fontsize=int(0.06 * IMAGE_HEIGHT),
    #     font="Impact",
    #     color="black",
    #     stroke_color="white",
    #     stroke_width=round(0.0026 * IMAGE_HEIGHT, 1),
    #     size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    #     method="caption",
    #     align="South",
    # ).set_duration(audio_clip.duration)

    txt_clip = TextClip(
        _read_text(frag_path),
        fontsize=int(0.05 * IMAGE_HEIGHT), # Iets kleiner zodat het mooi in de balk past
        font="Arial",                      # Beter leesbaar lettertype
        color="white",                     # Witte letters
        bg_color="rgba(0, 0, 0, 0.6)",     # Donkere transparante achtergrondbalk
        size=(IMAGE_WIDTH, int(0.15 * IMAGE_HEIGHT)), # Balk beslaat 15% van de hoogte
        method="caption",
        align="Center",
    ).set_duration(audio_clip.duration).set_position(("center", "bottom")) 


    video = CompositeVideoClip([image_clip.set_audio(audio_clip), txt_clip])
    out = project_dir / f"videos/video{idx}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(str(out), fps=FPS, codec="libx264", logger=None)
    _log(f"Video {idx} done.")


def concat_clips(project_dir: pathlib.Path) -> List[VideoFileClip]:
    """Return sorted list of VideoFileClip objects."""
    files = sorted(project_dir.glob("videos/video*.mp4"), key=lambda p: int(p.stem[5:]))
    print(files)
    return [VideoFileClip(str(f)) for f in files]


def make_final_video(project_name: str, project_dir: pathlib.Path) -> None:
    """Concatenate all clips, add background music, write final MP4."""
    clips = concat_clips(project_dir)
    clips = [c.crossfadein(1.0).crossfadeout(1.0) for c in clips]
    final = concatenate_videoclips(clips, padding=-1, method="compose")

    if BG_MUSIC:
        bg = AudioFileClip(str(BG_MUSIC_PATH)).audio_loop(duration=final.duration)
        bg = volumex(bg, MUSIC_VOLUME)
        final = final.set_audio(CompositeAudioClip([final.audio, bg]))

    out = project_dir / f"{project_name}.mp4"
    final.write_videofile(str(out), fps=FPS, codec="libx264")
    _log("Final video created successfully.")


# ---------- Main Orchestrator ----------
def run_project(project_dir: pathlib.Path) -> None:
    """Process a single project directory end-to-end."""
    story_file = project_dir / "story.txt"
    if not story_file.exists():
        _log(f"No story.txt in {project_dir}")
        return

    # Ensure folder layout
    for sub in ("text", "audio", "images", "videos"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    # Split text
    sent_dir = project_dir / "text/story_sentences"
    frag_dir = project_dir / "text/story_fragments"
    if not any(frag_dir.glob("*")):
        num_sentences = load_and_split_to_sentences(story_file)
        num_frags = sentences_to_fragments(num_sentences, project_dir)
    else:
        num_frags = len(list(frag_dir.glob("story_fragment*.txt")))

    # Generate prompts
    if not AUDIO_ONLY:
        _unload_sd()
        _reload_ollama()
        prompt_dir = project_dir / "text/image_prompts"
        for idx in range(num_frags):
            prompt_file = prompt_dir / f"image_prompt{idx}.txt"
            if not prompt_file.exists():
                prompt = build_image_prompt(_read_text(frag_dir / f"story_fragment{idx}.txt"))
                _write_text(prompt_file, prompt)
                _log(f"Done: image_prompt{idx}")

    # Generate audio
    for idx in range(num_frags):
        wav = project_dir / f"audio/voiceover{idx}.wav"
        mp3 = project_dir / f"audio/voiceover{idx}.mp3"
        if not (wav.exists() or mp3.exists()):
            frag = _read_text(frag_dir / f"story_fragment{idx}.txt")
            if TTS_PROVIDER == "elevenlabs":
                tts_elevenlabs(frag, mp3)
            elif TTS_PROVIDER == "kokoro":
                tts_kokoro(frag, wav)
            else:
                asyncio.run(tts_edge(frag, wav))
            _log(f"Done: voiceover{idx}")


    # Generate images
    if not AUDIO_ONLY:
        _unload_ollama()
        _reload_sd()
        for idx in range(num_frags):
            img = project_dir / f"images/image{idx}.jpg"
            if not img.exists():
                generate_image(idx, project_dir)

        # Generate clips (multi-process for speed)
        MAX_CORES = min(multiprocessing.cpu_count(), 8)          # cap at 8 
        
        def _ready(idx: int) -> bool:
            return (project_dir / f"videos/video{idx}.mp4").exists()

        tasks = [idx for idx in range(num_frags) if not _ready(idx)]
        if not tasks:
            _log("All clips already exist – skipping.")
        else:
            _log(f"Building {len(tasks)} clips using ≤ {MAX_CORES} processes …")

            with ProcessPoolExecutor(max_workers=MAX_CORES) as pool:
                for idx in tasks:
                    pool.submit(create_video_clip, idx, project_dir)
                    time.sleep(1)

        # Final render
        final_video = project_dir / f"{project_dir.name}.mp4"
        if not final_video.exists():
            make_final_video(project_dir.name, project_dir)


    
    if AUDIO_ONLY:
        final_audio_path = project_dir / f"{project_dir.name}.mp3"
        
        if not final_audio_path.exists():
            _log("AUDIO_ONLY mode: Concatenating fragments into final MP3...")
            audio_clips_list = []
            
            for idx in range(num_frags):
                # Try finding the wav first, fallback to mp3 if ElevenLabs was used
                wav = project_dir / f"audio/voiceover{idx}.wav"
                mp3 = project_dir / f"audio/voiceover{idx}.mp3"
                
                chosen_path = wav if wav.exists() else mp3
                
                if chosen_path.exists():
                    audio_clips_list.append(AudioFileClip(str(chosen_path)))
                else:
                    _log(f"Warning: Audio file missing for fragment {idx}")

            if audio_clips_list:
                # Merge the voice tracks sequentially
                final_audio = concatenate_audioclips(audio_clips_list)
                
                # Mix in background music if enabled in config
                if BG_MUSIC and BG_MUSIC_PATH.exists():
                    _log("Mixing background music into the track...")
                    bg_clip = AudioFileClip(str(BG_MUSIC_PATH)).fx(volumex, MUSIC_VOLUME)
                    
                    if bg_clip.duration < final_audio.duration:
                        bg_clip = bg_clip.loop(duration=final_audio.duration)
                    else:
                        bg_clip = bg_clip.set_duration(final_audio.duration)
                        
                    final_audio = CompositeAudioClip([final_audio, bg_clip])
                
                # Render to a single combined MP3 file
                _log(f"Writing final MP3 file: {final_audio_path}")
                final_audio.write_audiofile(
                    str(final_audio_path), 
                    fps=44100, 
                    bitrate="192k", 
                    codec="libmp3lame"
                )
                _log(f"🎉 Success! Audio book generated at: {final_audio_path}")
                
                # Free memory
                final_audio.close()
                for clip in audio_clips_list:
                    clip.close()
            else:
                _log("Error: No audio fragment files were found to merge.")
        else:
            _log("Final MP3 already exists – skipping render.")

def compile_all_projects_combined(main_projects_dir: pathlib.Path) -> None:
    """Combines all completed project outputs into one master file.
    Outputs a native .m4b with baked-in chapters if AUDIO_ONLY is True.
    """
    if not COMBINE_ALL_PROJECTS:
        return

    _log("--- Starting Global Master Compilation ---")
    
    # Sorteer mappen alfabetisch/numeriek voor de juiste verhaalvolgorde
    project_folders = sorted([d for d in main_projects_dir.iterdir() if d.is_dir()])
    
    if not project_folders:
        _log("No project folders found to combine.")
        return

    if AUDIO_ONLY:
        master_m4b_path = main_projects_dir / "Master_Audiobook.m4b"
        temp_audio_path = main_projects_dir / "temp_combined_audio.mp3"
        metadata_file = main_projects_dir / "ffmetadata.txt"
        
        master_clips = []
        current_time_ms = 0
        metadata_lines = [";FFMETADATA1\n"]  # FFmpeg metadata header

        _log("Gathering project audio files...")
        for folder in project_folders:
            project_mp3 = folder / f"{folder.name}.mp3"
            if project_mp3.exists():
                clip = AudioFileClip(str(project_mp3))
                master_clips.append(clip)
                
                # Bereken duur in milliseconden voor FFmpeg hoofdstukken
                duration_ms = int(clip.duration * 1000)
                end_time_ms = current_time_ms + duration_ms
                
                # Schrijf hoofdstuk-blok voor FFmpeg metadata
                metadata_lines.append("[CHAPTER]\n")
                metadata_lines.append("TIMEBASE=1/1000\n")
                metadata_lines.append(f"START={current_time_ms}\n")
                metadata_lines.append(f"END={end_time_ms}\n")
                metadata_lines.append(f"title={folder.name}\n\n")
                
                current_time_ms = end_time_ms
            else:
                _log(f"Skipping {folder.name}: Final MP3 not found.")

        # (Inside the AUDIO_ONLY branch of compile_all_projects_combined)
        if master_clips:
            _log("Merging tracks into a temporary file...")
            final_audio = concatenate_audioclips(master_clips)
            final_audio.write_audiofile(
                str(temp_audio_path),
                fps=44100,
                bitrate="192k",
                codec="libmp3lame"
            )
            
            final_audio.close()
            for c in master_clips:
                c.close()

            # Write out metadata configuration file
            _write_text(metadata_file, "".join(metadata_lines))
            
            # --- Check for local cover art asset ---
            cover_art_path = main_projects_dir / "cover.jpg"
            
            # Base FFmpeg command mappings
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_audio_path),
                "-i", str(metadata_file),
            ]
            
            # Append cover image mapping parameters if found
            if cover_art_path.exists():
                _log("🎨 cover.jpg asset detected! Mapping image stream into M4B...")
                ffmpeg_cmd.extend(["-i", str(cover_art_path)])
                
                # Setup Stream mappings (0=Audio, 1=Metadata, 2=Cover Art Image)
                mapping_args = [
                    "-map", "0:a",
                    "-map", "2:v",  # Map image input directly as a video/visual stream
                    "-c:v", "mjpeg", # Standard thumbnail stream format
                    "-disposition:v", "attached_pic" # Flag stream explicitly as album art
                ]
            else:
                _log("No cover.jpg found in project root. Compiling audio-only...")
                mapping_args = ["-map", "0:a"]

            # Append trailing parameters and codecs
            ffmpeg_cmd.extend(mapping_args + [
                "-map_metadata", "1",
                "-map_chapters", "1",
                "-c:a", "aac",
                "-b:a", "128k",
                str(master_m4b_path)
            ])
            
            # Execute background process pipeline
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Clean up working staging assets
            if temp_audio_path.exists(): os.remove(temp_audio_path)
            if metadata_file.exists(): os.remove(metadata_file)

            if result.returncode == 0:
                _log(f"🎉 Success! Native Audiobook with baked chapters & cover art created: {master_m4b_path}")
            else:
                _log(f"FFmpeg Error (Code {result.returncode}): {result.stderr}")
        else:
            _log("No project MP3 files were found to combine.")

    else:
        # VIDEO COMBINATION PATH (Onveranderd)
        master_video_path = main_projects_dir / "Master_Movie.mp4"
        master_videos = []

        _log("Gathering project video files...")
        for folder in project_folders:
            project_mp4 = folder / f"{folder.name}.mp4"
            if project_mp4.exists():
                master_videos.append(VideoFileClip(str(project_mp4)))
            else:
                _log(f"Skipping {folder.name}: Final MP4 not found.")

        if master_videos:
            _log("Merging videos into final Master Movie...")
            final_master_video = concatenate_videoclips(master_videos, method="compose")
            final_master_video.write_videofile(
                str(master_video_path),
                fps=FPS,
                codec="libx264",
                audio_codec="aac"
            )
            _log(f"🎉 Master Movie generated: {master_video_path}")
            
            final_master_video.close()
            for v in master_videos:
                v.close()
        else:
            _log("No project MP4 files were found to combine.")


if __name__ == "__main__":
    base = pathlib.Path.cwd() / "projects"
    if not base.exists():
        base.mkdir()
        
    # 1. Verwerk alle individuele projecten
    for proj in sorted(base.iterdir()):
        if proj.is_dir():
            _log(f"=== Running project {proj.name} ===")
            run_project(proj)
            
    # 2. Voeg alle projecten samen als de feature flag aan staat
    compile_all_projects_combined(base)
