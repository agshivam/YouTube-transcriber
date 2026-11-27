from __future__ import annotations

import os
import tempfile
from pathlib import Path

import json

import streamlit as st
import streamlit.components.v1 as components

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

COLAB_URL = "https://colab.research.google.com/github/agshivam/YouTube-transcriber/blob/main/Colab_Transcriber.ipynb"
GROQ_KEY = _secret("GROQ_API_KEY")
SUPADATA_KEY = _secret("SUPADATA_API_KEY")

st.set_page_config(page_title="YouTube Hindi/English Transcriber", page_icon="🎙️", layout="centered")
st.title("🎙️ YouTube Hindi, English & Hinglish Transcriber")
st.caption(
    "Paste a YouTube link or upload an audio/video file. YouTube's own Hindi/English captions are used when "
    "they exist; otherwise the audio is transcribed with Whisper AI."
)

st.markdown(f"**Long videos without captions?** Use the [free Google Colab version]({COLAB_URL}): paste a link, it downloads and transcribes on its own.")

source_mode = st.radio("Source", ["YouTube link", "Upload audio/video file"], horizontal=True)
url = ""
uploads = []
if source_mode == "YouTube link":
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploads = st.file_uploader(
        "Audio or video files: select one or many (mp3, m4a, mp4, wav, ogg, webm… up to 500 MB each)",
        type=["mp3", "m4a", "mp4", "wav", "ogg", "opus", "webm", "aac", "flac", "mov", "mkv", "mpeg", "mpga"],
        accept_multiple_files=True,
    ) or []
    if len(uploads) > 1:
        st.caption(f"{len(uploads)} files selected. They are transcribed one after another; keep this tab open.")

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
                    "YouTube is blocking audio downloads from this server. Use the free Google Colab version "
                    f"(paste the link there and it downloads the audio itself): {COLAB_URL} — or switch Source to "
                    "'Upload audio/video file'."
                ) from exc
            raise
        st.write("Audio downloaded. Transcribing…")
        return transcribe_audio_file(audio_path, title)


def save_upload(upload, folder: str) -> Path:
    path = Path(folder) / f"input{Path(upload.name).suffix.lower() or '.bin'}"
    upload.seek(0)
    with open(path, "wb") as fh:
        for block in iter(lambda: upload.read(8 * 1024 * 1024), b""):
            fh.write(block)
    return path


if st.button("Generate transcript", type="primary", use_container_width=True):
    notes: list[str] = []
    st.session_state["results"] = []
    st.session_state["failed"] = []
    try:
        if source_mode == "YouTube link":
            with st.status("Preparing transcript…", expanded=True) as status:
                st.session_state["results"] = [run_link(notes)]
                status.update(label="Transcript ready", state="complete", expanded=False)
        else:
            if not uploads:
                raise TranscriptionError("Choose one or more audio or video files first.")
            if not GROQ_KEY and len(uploads) > 1:
                st.warning("Without a GROQ_API_KEY each file is transcribed slowly on this server. Add the free key for batches.")
            overall = st.progress(0.0, text=f"0 of {len(uploads)} files done")
            for i, upload in enumerate(uploads, 1):
                with st.status(f"[{i}/{len(uploads)}] {upload.name}", expanded=True) as status:
                    try:
                        with tempfile.TemporaryDirectory(prefix="upload_") as tmp:
                            r = transcribe_audio_file(save_upload(upload, tmp), safe_stem(Path(upload.name).stem) or "Transcript")
                        st.session_state["results"].append(r)
                        status.update(label=f"✅ {upload.name}", state="complete", expanded=False)
                    except Exception as exc:  # keep going with the other files
                        st.session_state["failed"].append(f"{upload.name}: {exc}")
                        status.update(label=f"❌ {upload.name}: {exc}", state="error", expanded=False)
                overall.progress(i / len(uploads), text=f"{i} of {len(uploads)} files done")
        st.session_state["notes"] = [n for n in dict.fromkeys(notes) if n.startswith(("No ", "Supadata returned", "YouTube blocked"))]
    except TranscriptionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def zip_results(results) -> bytes:
    import io
    import zipfile

    buffer, used = io.BytesIO(), set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for r in results:
            stem, n = Path(r.txt_filename).stem, 2
            while stem in used:
                stem, n = f"{Path(r.txt_filename).stem} ({n})", n + 1
            used.add(stem)
            z.writestr(f"{stem}.txt", r.clean_text)
            z.writestr(f"{stem}_timestamps.txt", r.timestamped_text)
            z.writestr(f"{stem}.srt", r.srt_text)
    return buffer.getvalue()


def copy_button(text: str, label: str = "📋 Copy transcript", key: str = "copy") -> None:
    """A tap-friendly copy-to-clipboard button (works on iPad/iPhone Safari too)."""
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <button id="b" style="width:100%;padding:10px 14px;font-size:16px;border-radius:8px;cursor:pointer;
          border:1px solid #E5484D;background:#fff;color:#E5484D;font-family:sans-serif;font-weight:600">{label}</button>
        <script>
        const text = {payload};
        const btn = document.getElementById("b");
        function done(ok) {{
          btn.textContent = ok ? "✅ Copied!" : "Copy failed: select the text and copy manually";
          setTimeout(() => btn.textContent = {json.dumps(label)}, 2500);
        }}
        function fallback() {{
          const ta = document.createElement("textarea");
          ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
          document.body.appendChild(ta); ta.focus(); ta.select(); ta.setSelectionRange(0, text.length);
          let ok = false; try {{ ok = document.execCommand("copy"); }} catch (e) {{}}
          document.body.removeChild(ta); done(ok);
        }}
        btn.addEventListener("click", () => {{
          if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(() => done(true), fallback);
          }} else {{ fallback(); }}
        }});
        </script>
        """,
        height=52,
    )


def show_result(result, k: str) -> None:
    st.success(
        f"**{result.title}** — source: **{result.source}**. Language: **{result.detected_language or 'unknown'}**."
    )
    clean_tab, timed_tab, srt_tab = st.tabs(["Clean transcript", "Timestamps", "SRT subtitles"])
    with clean_tab:
        st.text_area("Transcript", result.clean_text, height=360, key=f"t_clean_{k}")
        copy_button(result.clean_text, key=f"copy_clean_{k}")
        st.download_button("Download TXT", result.clean_text.encode("utf-8"), result.txt_filename, "text/plain",
                           key=f"d_txt_{k}", on_click="ignore")
    with timed_tab:
        st.text_area("Timestamped transcript", result.timestamped_text, height=360, key=f"t_timed_{k}")
        copy_button(result.timestamped_text, "📋 Copy with timestamps", key=f"copy_timed_{k}")
        st.download_button("Download timestamped TXT", result.timestamped_text.encode("utf-8"),
                           result.timestamped_filename, "text/plain", key=f"d_ts_{k}", on_click="ignore")
    with srt_tab:
        st.text_area("SRT", result.srt_text, height=360, key=f"t_srt_{k}")
        copy_button(result.srt_text, "📋 Copy SRT", key=f"copy_srt_{k}")
        st.download_button("Download SRT", result.srt_text.encode("utf-8"), result.srt_filename,
                           "application/x-subrip", key=f"d_srt_{k}", on_click="ignore")


results = st.session_state.get("results") or []
for note in st.session_state.get("notes", []) if results else []:
    st.info(note)
for failure in st.session_state.get("failed", []):
    st.error(failure)
if len(results) == 1:
    show_result(results[0], "0")
elif len(results) > 1:
    st.subheader(f"{len(results)} transcripts ready")
    st.download_button("⬇️ Download all as ZIP (TXT + timestamps + SRT)", zip_results(results), "transcripts.zip",
                       "application/zip", type="primary", use_container_width=True,
                       on_click="ignore")
    all_text = "\n\n".join(f"===== {r.title} =====\n{r.clean_text}" for r in results)
    copy_button(all_text, "📋 Copy all transcripts", key="copy_all")
    names = [f"{i}. {r.title}" for i, r in enumerate(results, 1)]
    pick = st.selectbox("View one transcript", names)
    idx = names.index(pick)
    show_result(results[idx], str(idx))
