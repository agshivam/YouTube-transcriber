from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from transcriber import (
    MODEL_OPTIONS,
    TranscriptionError,
    _is_blocked,
    download_youtube_audio,
    fetch_supadata_transcript,
    fetch_youtube_captions,
    load_whisper_model,
    safe_stem,
    transcribe_file_groq,
    transcribe_file_local,
)


def _secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.environ.get(name, "")


# Optional: YouTube cookies (YT_COOKIES = contents of a cookies.txt file).
_cookies = _secret("YT_COOKIES")
if _cookies and not os.environ.get("YT_COOKIES_FILE"):
    _path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
    with open(_path, "w", encoding="utf-8") as fh:
        fh.write(_cookies)
    os.environ["YT_COOKIES_FILE"] = _path

GROQ_KEY = _secret("GROQ_API_KEY")
SUPADATA_KEY = _secret("SUPADATA_API_KEY")

st.set_page_config(page_title="YouTube Hindi/English Transcriber", page_icon="🎙️", layout="centered")
st.title("🎙️ YouTube Hindi, English & Hinglish Transcriber")
st.caption(
    "Paste a YouTube link or upload an audio/video file. YouTube's own Hindi/English captions are used when "
    "they exist; otherwise the audio is transcribed with Whisper AI."
)

source_mode = st.radio("Source", ["YouTube link", "Upload audio/video file"], horizontal=True)
url = ""
upload = None
if source_mode == "YouTube link":
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
else:
    upload = st.file_uploader(
        "Audio or video file (mp3, m4a, mp4, wav, ogg, webm… up to 500 MB)",
        type=["mp3", "m4a", "mp4", "wav", "ogg", "opus", "webm", "aac", "flac", "mov", "mkv", "mpeg", "mpga"],
    )

language_label = st.selectbox("Language", ["Auto: Hindi / English / Hinglish", "Hindi", "English"])
language = {"Auto: Hindi / English / Hinglish": None, "Hindi": "hi", "English": "en"}[language_label]

if GROQ_KEY:
    st.caption("Speech-to-text engine: **Groq Whisper large-v3** (free, about 8 hours of audio per day).")
else:
    st.warning(
        "No GROQ_API_KEY set, so audio is transcribed slowly on this server. Add a free Groq key "
        "(README → 'Groq') for fast, accurate Hindi transcription of long audio."
    )

with st.expander("Advanced options"):
    force_whisper = st.checkbox(
        "Skip YouTube captions and always transcribe the audio",
        help="Useful when YouTube's captions are missing in your language or poor quality.",
    )
    use_local = st.checkbox(
        "Use the on-server Whisper model instead of Groq", value=not GROQ_KEY, disabled=not GROQ_KEY,
        help="Slow on the free cloud. Mainly for running the app on your own computer.",
    )
    model_name = st.selectbox("On-server Whisper model", MODEL_OPTIONS, index=0)
    local_model_path = st.text_input("Optional local faster-whisper model folder")
    beam_size = st.slider("Beam size (on-server model)", 1, 10, 5)


@st.cache_resource(show_spinner=False)
def cached_model(selected_model: str, local_path: str):
    return load_whisper_model(selected_model, local_path or None)


def transcribe_audio_file(path: Path, title: str):
    """Transcribe a local audio/video file with Groq (default) or the on-server model."""
    if GROQ_KEY and not use_local:
        bar = st.progress(0.0, text="Preparing audio…")

        def progress(done: int, total: int, msg: str) -> None:
            bar.progress(min(done / max(total, 1), 1.0), text=msg)

        return transcribe_file_groq(path, GROQ_KEY, language, title=title, progress=progress)
    st.write("Loading speech model (first run downloads it)…")
    model, runtime = cached_model(model_name, local_model_path.strip())
    st.write(f"Transcribing on {runtime} (this can take a while)…")
    return transcribe_file_local(path, model, language, beam_size, title=title)


def run_link(notes: list[str]):
    result = None
    if not force_whisper:
        st.write("Checking YouTube for Hindi/English captions…")
        result = fetch_youtube_captions(url, language, notes)
        for note in notes:
            st.write(note)
        if result is not None and (language is None or result.detected_language == language):
            return result

    st.write("Downloading the audio from YouTube…")
    with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp:
        try:
            audio_path, title = download_youtube_audio(url, tmp)
        except TranscriptionError as exc:
            blocked = _is_blocked(exc)
            if blocked and SUPADATA_KEY:
                st.write("YouTube blocked the download; asking Supadata instead…")
                title = result.title if result is not None else "YouTube transcript"
                mode = "generate" if (force_whisper or language) else "auto"
                return fetch_supadata_transcript(url, SUPADATA_KEY, language, mode, notes, title=title)
            if result is not None:
                notes.append("YouTube blocked the audio download, so only the captions found are shown.")
                return result
            if blocked:
                raise TranscriptionError(
                    "YouTube is blocking audio downloads from this server. Switch Source to "
                    "'Upload audio/video file' and upload the audio instead (README → 'Getting the audio on iPad')."
                ) from exc
            raise
        st.write("Audio downloaded. Transcribing…")
        return transcribe_audio_file(audio_path, title)


if st.button("Generate transcript", type="primary", use_container_width=True):
    notes: list[str] = []
    st.session_state["result"] = None
    try:
        with st.status("Preparing transcript…", expanded=True) as status:
            if source_mode == "YouTube link":
                result = run_link(notes)
            else:
                if upload is None:
                    raise TranscriptionError("Choose an audio or video file first.")
                with tempfile.TemporaryDirectory(prefix="upload_") as tmp:
                    path = Path(tmp) / f"input{Path(upload.name).suffix.lower() or '.bin'}"
                    with open(path, "wb") as fh:
                        for block in iter(lambda: upload.read(8 * 1024 * 1024), b""):
                            fh.write(block)
                    result = transcribe_audio_file(path, safe_stem(Path(upload.name).stem) or "Transcript")
            status.update(label="Transcript ready", state="complete", expanded=False)
        st.session_state["result"] = result
        st.session_state["notes"] = [n for n in dict.fromkeys(notes) if n.startswith(("No ", "Supadata returned", "YouTube blocked"))]
    except TranscriptionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

result = st.session_state.get("result")
if result is not None:
    for note in st.session_state.get("notes", []):
        st.info(note)
    st.success(
        f"**{result.title}** — source: **{result.source}**. Language: **{result.detected_language or 'unknown'}**."
    )
    clean_tab, timed_tab, srt_tab = st.tabs(["Clean transcript", "Timestamps", "SRT subtitles"])
    with clean_tab:
        st.text_area("Transcript", result.clean_text, height=360)
        st.download_button("Download TXT", result.clean_text.encode("utf-8"), result.txt_filename, "text/plain")
    with timed_tab:
        st.text_area("Timestamped transcript", result.timestamped_text, height=360)
        st.download_button(
            "Download timestamped TXT", result.timestamped_text.encode("utf-8"), result.timestamped_filename, "text/plain"
        )
    with srt_tab:
        st.text_area("SRT", result.srt_text, height=360)
        st.download_button("Download SRT", result.srt_text.encode("utf-8"), result.srt_filename, "application/x-subrip")
