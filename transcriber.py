from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse
import html
import os
import shutil
import json
import re

# "small" is the default: it fits the free Streamlit Cloud machine. Pick large-v3-turbo locally for best accuracy.
MODEL_OPTIONS = ["small", "base", "medium", "large-v3-turbo"]
_ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


class TranscriptionError(RuntimeError):
    """A user-facing transcription failure."""


@dataclass(frozen=True)
class TranscriptResult:
    title: str
    clean_text: str
    timestamped_text: str
    srt_text: str
    detected_language: str | None
    txt_filename: str
    timestamped_filename: str
    srt_filename: str
    source: str = "Whisper speech-to-text"


def validate_youtube_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        raise TranscriptionError("Paste a YouTube URL first.")
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise TranscriptionError("The YouTube URL is invalid.") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _ALLOWED_HOSTS:
        raise TranscriptionError("Enter a valid youtube.com or youtu.be URL.")
    if host == "youtu.be":
        valid_video = bool(parsed.path.strip("/"))
    elif parsed.path == "/watch":
        valid_video = bool(parse_qs(parsed.query).get("v", [""])[0])
    else:
        valid_video = parsed.path.startswith(("/shorts/", "/live/", "/embed/"))
    if not valid_video:
        raise TranscriptionError("The URL does not contain a YouTube video ID.")
    return value


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_short_timestamp(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def safe_stem(title: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "_", title or "youtube_transcript", flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip(" ._")
    return (stem[:120] or "youtube_transcript")


def build_outputs(
    title: str,
    segments: Iterable[Any],
    detected_language: str | None = None,
    source: str = "Whisper speech-to-text",
) -> TranscriptResult:
    clean, timed, srt = [], [], []
    number = 1
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        start = float(getattr(segment, "start", 0.0))
        end = max(start, float(getattr(segment, "end", start)))
        clean.append(text)
        timed.append(f"[{format_short_timestamp(start)}] {text}")
        srt.append(f"{number}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}")
        number += 1
    if not clean:
        raise TranscriptionError("No speech was detected in this video.")
    stem = safe_stem(title)
    return TranscriptResult(
        title=title or "YouTube transcript",
        clean_text="\n".join(clean),
        timestamped_text="\n".join(timed),
        srt_text="\n\n".join(srt) + "\n",
        detected_language=detected_language,
        txt_filename=f"{stem}.txt",
        timestamped_filename=f"{stem}_timestamps.txt",
        srt_filename=f"{stem}.srt",
        source=source,
    )


def load_whisper_model(model_name: str, local_model_path: str | None = None):
    source = local_model_path.strip() if local_model_path and local_model_path.strip() else model_name
    if local_model_path and not Path(source).is_dir():
        raise TranscriptionError(f"Local model folder does not exist: {source}")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc
    try:
        return WhisperModel(source, device="cuda", compute_type="float16"), "GPU (CUDA/float16)"
    except Exception:
        try:
            return WhisperModel(source, device="cpu", compute_type="int8"), "CPU (int8)"
        except Exception as exc:
            raise TranscriptionError(
                "Whisper model could not be loaded. Check the internet connection for the first model download, "
                "or provide a valid local model folder."
            ) from exc


# YouTube player clients to try in order. If YouTube blocks one (HTTP 403 / bot check), the next is tried.
_CLIENT_FALLBACKS: list[list[str] | None] = [None, ["tv", "web_safari"], ["mweb", "android_vr"]]


def _ydl_options(player_client: list[str] | None = None, **extra: Any) -> dict:
    """Common yt-dlp options: JS runtime (needed by YouTube), optional cookies, client choice."""
    options: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30}
    try:  # pip package "deno" ships the binary; YouTube needs a JS runtime to unlock streams
        import deno

        options["js_runtimes"] = {"deno": {"path": deno.find_deno_bin()}}
    except Exception:
        pass  # yt-dlp will look for deno/node on PATH
    options["remote_components"] = ["ejs:github"]
    cookies = os.environ.get("YT_COOKIES_FILE")
    if cookies and Path(cookies).is_file():
        options["cookiefile"] = cookies
    if player_client:
        options["extractor_args"] = {"youtube": {"player_client": player_client}}
    options.update(extra)
    return options


def _is_blocked(exc: Exception) -> bool:
    text = str(exc).lower()
    keys = ("403", "forbidden", "not a bot", "sign in to confirm", "po token", "unavailable",
            "needs to be reloaded", "reload", "429", "too many requests", "precondition")
    return any(k in text for k in keys)


def _download_audio(url: str, directory: str) -> tuple[Path, str]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise TranscriptionError("Required packages are missing. Run: pip install -r requirements.txt") from exc

    if not shutil.which("ffmpeg"):  # Streamlit Cloud installs ffmpeg via packages.txt
        try:
            import static_ffmpeg

            static_ffmpeg.add_paths()
        except Exception as exc:
            raise TranscriptionError("FFmpeg is not available. Install it or run: pip install static-ffmpeg") from exc
    output_template = str(Path(directory) / "audio.%(ext)s")
    info, last_exc = None, None
    for client in _CLIENT_FALLBACKS:
        options = _ydl_options(
            client,
            format="bestaudio/best",
            outtmpl=output_template,
            retries=3,
            fragment_retries=3,
            postprocessors=[{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            postprocessor_args=["-ar", "16000", "-ac", "1"],
        )
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as exc:
            last_exc = exc
            for leftover in Path(directory).glob("audio.*"):
                leftover.unlink(missing_ok=True)
            if not _is_blocked(exc):
                break
    if info is None:
        hint = (
            " YouTube is blocking this server. Add a free Supadata API key (README → 'Supadata') "
            "or YouTube cookies, or run the app on your own computer."
            if last_exc is not None and _is_blocked(last_exc)
            else ""
        )
        raise TranscriptionError(f"YouTube audio download failed: {last_exc}.{hint}")
    audio_path = Path(directory) / "audio.wav"
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise TranscriptionError("YouTube audio conversion failed. FFmpeg did not create a valid WAV file.")
    return audio_path, str((info or {}).get("title") or "YouTube transcript")


def transcribe_youtube(url: str, model: Any, language: str | None = None, beam_size: int = 5) -> TranscriptResult:
    checked_url = validate_youtube_url(url)
    if language not in {None, "hi", "en"}:
        raise TranscriptionError("Language must be auto, Hindi, or English.")
    if not 1 <= int(beam_size) <= 10:
        raise TranscriptionError("Beam size must be between 1 and 10.")
    with TemporaryDirectory(prefix="youtube_transcriber_") as temp_dir:
        audio_path, title = _download_audio(checked_url, temp_dir)
        try:
            segments, info = model.transcribe(
                load_audio_array(audio_path),
                language=language,
                beam_size=int(beam_size),
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=True,
            )
            detected = getattr(info, "language", language)
            return build_outputs(title, segments, detected)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Speech transcription failed: {exc}") from exc


# ---------------------------------------------------------------------------
# YouTube captions (fast path: no audio download, no Whisper model needed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaptionSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CaptionTrack:
    language: str  # base code: "hi" or "en"
    kind: str  # human-readable label shown in the UI
    url: str
    ext: str


_LANG_NAMES = {"hi": "Hindi", "en": "English"}


def _base_lang(code: str) -> str:
    return (code or "").split("-")[0].lower()


def _is_translated(track_url: str) -> bool:
    return "tlang" in parse_qs(urlparse(track_url).query)


def _best_format(formats: list[dict]) -> tuple[str, str] | None:
    by_ext = {f.get("ext"): f.get("url") for f in formats or [] if f.get("url")}
    for ext in ("json3", "vtt"):
        if by_ext.get(ext):
            return ext, by_ext[ext]
    return None


def pick_caption_track(info: dict, language: str | None = None) -> CaptionTrack | None:
    """Choose the best Hindi/English caption track.

    Order: human-uploaded captions -> YouTube auto-generated captions (original
    language) -> YouTube auto-translated captions (only when a specific language
    was requested). In Auto mode the video's own language is tried first.
    """
    if language in {"hi", "en"}:
        wanted = [language]
    else:
        original = _base_lang(info.get("language") or "")
        wanted = [original] + [x for x in ("hi", "en") if x != original] if original in {"hi", "en"} else ["hi", "en"]

    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    def candidates(tracks: dict, lang: str, translated: bool):
        # "en-orig" / "en" / "en-US" ... ; prefer "-orig" and exact matches first.
        keys = [k for k in tracks if _base_lang(k) == lang and k != "live_chat"]
        keys.sort(key=lambda k: (not k.endswith("-orig"), k != lang, k))
        for key in keys:
            fmt = _best_format(tracks[key])
            if fmt and _is_translated(fmt[1]) == translated:
                yield fmt

    for lang in wanted:
        for fmt in candidates(manual, lang, False):
            return CaptionTrack(lang, f"YouTube captions ({_LANG_NAMES[lang]}, uploaded by creator)", fmt[1], fmt[0])
    for lang in wanted:
        for fmt in candidates(automatic, lang, False):
            return CaptionTrack(lang, f"YouTube auto-generated captions ({_LANG_NAMES[lang]})", fmt[1], fmt[0])
    if language in {"hi", "en"}:
        for fmt in candidates(automatic, language, True):
            return CaptionTrack(language, f"YouTube auto-translated captions ({_LANG_NAMES[language]})", fmt[1], fmt[0])
        # Requested language missing: use the other language rather than nothing.
        other = "en" if language == "hi" else "hi"
        for tracks, label in ((manual, "uploaded by creator"), (automatic, "auto-generated")):
            for fmt in candidates(tracks, other, False):
                return CaptionTrack(other, f"YouTube captions ({_LANG_NAMES[other]}, {label})", fmt[1], fmt[0])
    return None


def parse_json3(raw: str) -> list[CaptionSegment]:
    data = json.loads(raw)
    segments = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs)).strip()
        if not text:
            continue
        start = event.get("tStartMs", 0) / 1000
        end = start + event.get("dDurationMs", 0) / 1000
        segments.append(CaptionSegment(start, end, text))
    return segments


_VTT_TIME = re.compile(r"(\d+:)?(\d{2}):(\d{2})\.(\d{3})")


def _vtt_seconds(stamp: str) -> float:
    m = _VTT_TIME.match(stamp.strip())
    if not m:
        return 0.0
    hours = int(m.group(1)[:-1]) if m.group(1) else 0
    return hours * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000


def parse_vtt(raw: str) -> list[CaptionSegment]:
    segments: list[CaptionSegment] = []
    seen_last = ""
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        lines = [l for l in block.split("\n") if l.strip()]
        timing = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if timing is None:
            continue
        start_s, end_s = lines[timing].split("-->")[:2]
        for line in lines[timing + 1:]:
            text = html.unescape(re.sub(r"<[^>]+>", "", line)).strip()
            # Auto-generated VTT repeats the previous line in each cue; skip repeats.
            if text and text != seen_last:
                segments.append(CaptionSegment(_vtt_seconds(start_s), _vtt_seconds(end_s.split()[0]), text))
                seen_last = text
    return segments


def _extract_info(url: str) -> dict:
    try:
        import yt_dlp
    except ImportError as exc:
        raise TranscriptionError("Required packages are missing. Run: pip install -r requirements.txt") from exc
    last_exc: Exception | None = None
    for client in _CLIENT_FALLBACKS:
        try:
            with yt_dlp.YoutubeDL(_ydl_options(client, skip_download=True)) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except Exception as exc:
            last_exc = exc
            if not _is_blocked(exc):
                break
    raise TranscriptionError(f"Could not read the YouTube video: {last_exc}")


def _download_text(track_url: str) -> str:
    import yt_dlp

    with yt_dlp.YoutubeDL(_ydl_options()) as ydl:
        return ydl.urlopen(track_url).read().decode("utf-8", errors="replace")


def fetch_youtube_captions(
    url: str, language: str | None = None, notes: list[str] | None = None
) -> TranscriptResult | None:
    """Return a transcript built from YouTube's own Hindi/English captions, or None if none exist.

    Reasons for returning None are appended to ``notes`` so the UI can show them.
    """
    notes = notes if notes is not None else []
    checked_url = validate_youtube_url(url)
    if language not in {None, "hi", "en"}:
        raise TranscriptionError("Language must be auto, Hindi, or English.")
    info = _extract_info(checked_url)
    track = pick_caption_track(info, language)
    if track is None:
        available = sorted(set((info.get("subtitles") or {})) | {
            k for k, v in (info.get("automatic_captions") or {}).items()
            if v and not _is_translated((v[0] or {}).get("url", ""))
        })
        notes.append("This video has no Hindi/English captions on YouTube"
                     + (f" (found: {', '.join(available[:6])})." if available else "."))
        return None
    try:
        raw = _download_text(track.url)
        segments = parse_json3(raw) if track.ext == "json3" else parse_vtt(raw)
    except Exception as exc:  # caption download/parse failed -> caller falls back to Whisper
        notes.append(f"{track.kind} exist, but YouTube refused to send them ({str(exc)[:120]}).")
        return None
    if not segments:
        notes.append(f"{track.kind} exist, but YouTube sent them empty (it often does this to cloud servers).")
        return None
    if language in {"hi", "en"} and track.language != language:
        notes.append(
            f"No {_LANG_NAMES[language]} captions on this video, so the {_LANG_NAMES[track.language]} captions are "
            f"shown. For a {_LANG_NAMES[language]} transcript from the audio, tick Advanced options → "
            "'Skip YouTube captions and always use Whisper'."
        )
    title = str(info.get("title") or "YouTube transcript")
    return build_outputs(title, segments, track.language, source=track.kind)


# ---------------------------------------------------------------------------
# Supadata (optional): fetches/generates transcripts on Supadata's servers, so it
# keeps working when YouTube blocks the hosting server. Needs SUPADATA_API_KEY.
# Free plan: 100 credits/month. Native captions = 1 credit; AI transcript = 2 credits per video minute.
# ---------------------------------------------------------------------------

SUPADATA_BASE = "https://api.supadata.ai/v1/transcript"


def _http_get_json(url: str, api_key: str) -> tuple[int, dict]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"x-api-key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8") or "{}"
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:
            return exc.code, {}


def fetch_supadata_transcript(
    url: str,
    api_key: str,
    language: str | None = None,
    mode: str = "auto",
    notes: list[str] | None = None,
    http_get=_http_get_json,
    sleep=None,
    max_wait_seconds: int = 900,
    title: str = "YouTube transcript",
) -> TranscriptResult:
    """Get a transcript from Supadata. mode: native (captions only), generate (AI from audio), auto (both)."""
    import time
    from urllib.parse import urlencode

    notes = notes if notes is not None else []
    sleep = sleep or time.sleep
    checked_url = validate_youtube_url(url)
    if mode not in {"native", "generate", "auto"}:
        raise TranscriptionError("Supadata mode must be native, generate or auto.")
    params = {"url": checked_url, "mode": mode}
    if language in {"hi", "en"}:
        params["lang"] = language
    status, data = http_get(f"{SUPADATA_BASE}?{urlencode(params)}", api_key)

    if status == 202 and data.get("jobId"):
        job_url = f"{SUPADATA_BASE}/{data['jobId']}"
        waited = 0
        while True:
            status, data = http_get(job_url, api_key)
            state = data.get("status")
            if state == "completed" or (status == 200 and "content" in data and state is None):
                status = 200
                break
            if state == "failed":
                raise TranscriptionError(f"Supadata could not transcribe this video: {data.get('error')}")
            if waited >= max_wait_seconds:
                raise TranscriptionError("Supadata is taking too long. Try again in a few minutes.")
            sleep(2)
            waited += 2

    if status == 206:
        raise TranscriptionError("Supadata found no captions for this video (try with Whisper / AI mode).")
    if status in {401, 403} and "api" in json.dumps(data).lower() and "key" in json.dumps(data).lower():
        raise TranscriptionError("Supadata API key is invalid. Check SUPADATA_API_KEY in the app's Secrets.")
    if status == 429 or (status == 402):
        raise TranscriptionError("Supadata free credits are used up for this month.")
    if status == 404:
        raise TranscriptionError("Supadata says this video does not exist or is private.")
    if status != 200:
        detail = data.get("message") or data.get("error") or data.get("details") or status
        raise TranscriptionError(f"Supadata request failed: {detail}")

    chunks = data.get("content") or []
    if isinstance(chunks, str):
        chunks = [{"text": chunks, "offset": 0, "duration": 0}]
    segments = [
        CaptionSegment(
            (c.get("offset") or 0) / 1000,
            ((c.get("offset") or 0) + (c.get("duration") or 0)) / 1000,
            re.sub(r"\s+", " ", html.unescape(c.get("text") or "")).strip(),
        )
        for c in chunks
    ]
    segments = [s for s in segments if s.text]
    if not segments:
        raise TranscriptionError("No speech was detected in this video.")
    lang = _base_lang(data.get("lang") or "") or language
    if language in {"hi", "en"} and lang and lang != language:
        notes.append(f"Supadata returned {_LANG_NAMES.get(lang, lang)} instead of {_LANG_NAMES[language]}.")
    label = "Supadata AI transcript" if mode == "generate" else "Supadata (YouTube captions or AI)"
    return build_outputs(title, segments, lang, source=label)


# ---------------------------------------------------------------------------
# Groq Whisper (optional, free): whisper-large-v3 on Groq's servers. Free plan allows
# about 8 hours of audio per day (2 hours per hour), 25 MB per request, so long audio
# is cut into 10-minute MP3 chunks first. Needs GROQ_API_KEY.
# ---------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
CHUNK_SECONDS = 600


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception as exc:
        raise TranscriptionError("FFmpeg is not available.") from exc
    path = shutil.which("ffmpeg")
    if not path:
        raise TranscriptionError("FFmpeg is not available.")
    return path


def media_duration(path: str | Path) -> float:
    """Duration in seconds, read from ffmpeg's banner (ffprobe may be missing)."""
    import subprocess

    out = subprocess.run([_ensure_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out.stderr)
    if not m:
        raise TranscriptionError("Could not read this file. Upload an audio or video file (mp3, m4a, mp4, wav…).")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def split_audio(path: str | Path, out_dir: str | Path, chunk_seconds: int = CHUNK_SECONDS) -> list[tuple[Path, float]]:
    """Convert any audio/video to 16 kHz mono MP3 chunks. Returns [(chunk_path, start_offset_seconds)]."""
    import subprocess

    duration = media_duration(path)
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    index = 0
    while start < duration - 0.5:
        target = Path(out_dir) / f"chunk_{index:04d}.mp3"
        cmd = [_ensure_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
               "-t", str(chunk_seconds), "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(target)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not target.exists():
            raise TranscriptionError(f"Audio conversion failed: {result.stderr.strip()[:200]}")
        chunks.append((target, start))
        start += chunk_seconds
        index += 1
    if not chunks:
        raise TranscriptionError("The audio is empty.")
    return chunks


def _groq_post(chunk: Path, api_key: str, language: str | None) -> tuple[int, dict, dict]:
    import requests

    data = {"model": GROQ_MODEL, "response_format": "verbose_json", "temperature": "0"}
    if language in {"hi", "en"}:
        data["language"] = language
    with open(chunk, "rb") as fh:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"file": (chunk.name, fh, "audio/mpeg")},
            timeout=300,
        )
    try:
        body = response.json()
    except Exception:
        body = {"error": {"message": response.text[:200]}}
    return response.status_code, body, dict(response.headers)


def _retry_seconds(headers: dict, body: dict) -> float:
    for key in ("retry-after", "Retry-After"):
        if headers.get(key):
            try:
                return float(headers[key])
            except ValueError:
                pass
    message = json.dumps(body)
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", message)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    return 30.0


def transcribe_file_groq(
    path: str | Path,
    api_key: str,
    language: str | None = None,
    title: str = "Transcript",
    progress=None,
    post=_groq_post,
    sleep=None,
    max_wait_per_chunk: float = 3900,
) -> TranscriptResult:
    """Transcribe a local audio/video file of any length with Groq's free Whisper API."""
    import time

    sleep = sleep or time.sleep
    if not api_key:
        raise TranscriptionError("GROQ_API_KEY is missing. Add it in the app's Secrets.")
    progress = progress or (lambda done, total, msg: None)
    with TemporaryDirectory(prefix="groq_chunks_") as tmp:
        chunks = split_audio(path, tmp)
        segments: list[CaptionSegment] = []
        detected = None
        for i, (chunk, offset) in enumerate(chunks):
            progress(i, len(chunks), f"Transcribing part {i + 1} of {len(chunks)}…")
            waited = 0.0
            while True:
                try:
                    status, body, headers = post(chunk, api_key, language)
                except Exception as exc:  # network hiccup: retry for up to 2 minutes
                    if waited >= 120:
                        raise TranscriptionError(f"Could not reach Groq: {exc}") from exc
                    sleep(10)
                    waited += 10
                    continue
                if status == 200:
                    break
                if status == 429:
                    pause = min(max(_retry_seconds(headers, body), 5.0), 600.0)
                    if waited + pause > max_wait_per_chunk:
                        raise TranscriptionError(
                            "Groq's free daily limit (about 8 hours of audio) is reached. Try again tomorrow; "
                            f"{i} of {len(chunks)} parts were done."
                        )
                    progress(i, len(chunks), f"Groq free limit pause: waiting {int(pause)} s, then continuing…")
                    sleep(pause)
                    waited += pause
                    continue
                if status == 401:
                    raise TranscriptionError("Groq API key is invalid. Check GROQ_API_KEY in the app's Secrets.")
                if status >= 500 and waited < 120:
                    sleep(10)
                    waited += 10
                    continue
                message = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body
                raise TranscriptionError(f"Groq transcription failed: {message}")
            detected = detected or _GROQ_LANGS.get(str(body.get("language", "")).lower(), body.get("language"))
            parts = body.get("segments") or []
            if not parts and body.get("text"):
                parts = [{"start": 0, "end": min(CHUNK_SECONDS, body.get("duration") or 0), "text": body["text"]}]
            for seg in parts:
                text = re.sub(r"\s+", " ", seg.get("text") or "").strip()
                if text:
                    segments.append(CaptionSegment(offset + float(seg.get("start") or 0), offset + float(seg.get("end") or 0), text))
        progress(len(chunks), len(chunks), "Done")
    if not segments:
        raise TranscriptionError("No speech was detected in this audio.")
    return build_outputs(title, segments, language or detected, source=f"Groq Whisper ({GROQ_MODEL})")


_GROQ_LANGS = {"hindi": "hi", "english": "en", "urdu": "ur", "punjabi": "pa"}


def download_youtube_audio(url: str, directory: str) -> tuple[Path, str]:
    """Download a YouTube video's audio as 16 kHz mono WAV. Returns (path, title)."""
    return _download_audio(validate_youtube_url(url), directory)


def load_audio_array(path: str | Path):
    """Decode any audio/video to 16 kHz mono float32 with FFmpeg (avoids PyAV version breakages)."""
    import subprocess

    import numpy as np

    cmd = [_ensure_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:200]
        raise TranscriptionError(f"Could not read this file. Upload an audio or video file (mp3, m4a, mp4, wav…). {detail}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_file_local(path: str | Path, model: Any, language: str | None = None, beam_size: int = 5,
                          title: str = "Transcript") -> TranscriptResult:
    """Transcribe a local audio/video file with the on-server faster-whisper model."""
    audio = load_audio_array(path)
    try:
        segments, info = model.transcribe(
            audio, language=language, beam_size=int(beam_size), vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=True,
        )
        return build_outputs(title, segments, getattr(info, "language", language))
    except Exception as exc:
        raise TranscriptionError(f"Speech transcription failed: {exc}") from exc
