from __future__ import annotations

import os
import tempfile

import streamlit as st

from transcriber import (
    MODEL_OPTIONS,
    TranscriptionError,
    fetch_supadata_transcript,
    fetch_youtube_captions,
    load_whisper_model,
    transcribe_youtube,
)

# Optional: YouTube cookies stored as a Streamlit secret (YT_COOKIES = contents of a cookies.txt file).
try:
    _cookies = st.secrets.get("YT_COOKIES", "")
except Exception:
    _cookies = ""
if _cookies and not os.environ.get("YT_COOKIES_FILE"):
    _path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
    with open(_path, "w", encoding="utf-8") as fh:
        fh.write(_cookies)
    os.environ["YT_COOKIES_FILE"] = _path

try:
    SUPADATA_KEY = st.secrets.get("SUPADATA_API_KEY", "") or os.environ.get("SUPADATA_API_KEY", "")
except Exception:
    SUPADATA_KEY = os.environ.get("SUPADATA_API_KEY", "")

st.set_page_config(page_title="YouTube Hindi/English Transcriber", page_icon="🎙️", layout="centered")
st.title("🎙️ YouTube Hindi, English & Hinglish Transcriber")
st.caption("Paste a YouTube link. Uses YouTube's Hindi/English captions instantly when they exist; if not, it listens to the audio with Whisper and writes the transcript itself.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
left, right = st.columns(2)
with left:
    model_name = st.selectbox(
        "Whisper model",
        MODEL_OPTIONS,
        index=0,
        help="Used only when the video has no Hindi/English captions. small runs on the free cloud; large-v3-turbo is most accurate on a strong computer.",
    )
with right:
    language_label = st.selectbox(
        "Language mode",
        ["Auto: Hindi / English / Hinglish", "Hindi", "English"],
    )

language_map = {
    "Auto: Hindi / English / Hinglish": None,
    "Hindi": "hi",
    "English": "en",
}

with st.expander("Advanced options"):
    local_model_path = st.text_input(
        "Optional local faster-whisper model folder",
        help="Leave blank to download the selected model from Hugging Face on first use.",
    )
    beam_size = st.slider("Beam size", 1, 10, 5)
    force_whisper = st.checkbox(
        "Skip YouTube captions and always use Whisper",
        help="Useful when YouTube's auto-captions are poor quality.",
    )

@st.cache_resource(show_spinner=False)
def cached_model(selected_model: str, local_path: str):
    return load_whisper_model(selected_model, local_path or None)

if st.button("Generate transcript", type="primary", use_container_width=True):
    try:
        language = language_map[language_label]
        with st.status("Preparing transcript…", expanded=True) as status:
            result = None
            notes: list[str] = []
            if not force_whisper:
                st.write("Checking YouTube for Hindi/English captions…")
                result = fetch_youtube_captions(url, language, notes)
                for note in notes:
                    st.write(note)
            exact = result is not None and (language is None or result.detected_language == language)
            if not exact and SUPADATA_KEY:
                mode = "generate" if (force_whisper or language) else "auto"
                st.write(
                    "Creating an AI transcript from the audio with Supadata (2 free credits per video minute)…"
                    if mode == "generate" else "Fetching the transcript with Supadata…"
                )
                try:
                    title = result.title if result is not None else "YouTube transcript"
                    result = fetch_supadata_transcript(url, SUPADATA_KEY, language, mode, notes, title=title)
                except TranscriptionError as exc:
                    if result is None:
                        raise
                    st.write(f"Supadata failed ({exc}); showing the captions found instead.")
            if result is None:
                st.write("Loading speech model (first run downloads it)…")
                model, runtime = cached_model(model_name, local_model_path.strip())
                st.write(f"Downloading audio and transcribing on {runtime}…")
                result = transcribe_youtube(url=url, model=model, language=language, beam_size=beam_size)
            status.update(label="Transcript ready", state="complete", expanded=False)
        shown = set()
        for note in notes:
            if note.startswith(("No ", "Supadata returned")) and note not in shown:
                st.info(note)
                shown.add(note)

        st.success(
            f"**{result.title}** — source: **{result.source}**. "
            f"Language: **{result.detected_language or 'unknown'}**."
        )
        clean_tab, timed_tab, srt_tab = st.tabs(["Clean transcript", "Timestamps", "SRT subtitles"])
        with clean_tab:
            st.text_area("Transcript", result.clean_text, height=360)
            st.download_button("Download TXT", result.clean_text.encode("utf-8"), result.txt_filename, "text/plain")
        with timed_tab:
            st.text_area("Timestamped transcript", result.timestamped_text, height=360)
            st.download_button(
                "Download timestamped TXT",
                result.timestamped_text.encode("utf-8"),
                result.timestamped_filename,
                "text/plain",
            )
        with srt_tab:
            st.text_area("SRT", result.srt_text, height=360)
            st.download_button("Download SRT", result.srt_text.encode("utf-8"), result.srt_filename, "application/x-subrip")
    except TranscriptionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
