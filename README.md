# 🎙️ YouTube Hindi / English / Hinglish Transcriber

**Live app:** https://yt-transcriber.streamlit.app  ·  Built with Python, Streamlit, yt-dlp and faster-whisper

Paste any YouTube link and get a clean transcript, a timestamped transcript and SRT subtitles.

## How it works
1. Paste a YouTube link and click **Generate transcript**.
2. The app first checks YouTube for Hindi or English captions (creator-uploaded first, then auto-generated).
   If found, the transcript appears in a few seconds. No audio download and no AI model needed.
3. If the video has **no Hindi/English captions**, the app downloads the audio and writes the transcript
   itself with Whisper (runs on your computer).
4. Download as TXT, timestamped TXT, or SRT subtitles. The result shows which source was used.

Language mode: **Auto** takes whichever of Hindi/English the video has. Picking **Hindi** or **English**
also allows YouTube's auto-translated captions in that language before falling back to Whisper.
Tick *Advanced options → Skip YouTube captions* to force Whisper when YouTube's captions are poor.

## Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository `agshivam/youtube-transcriber`, branch `main`, main file `app.py`.
4. **Advanced settings → Python 3.12**. App URL: `yt-transcriber` (or any free name).
5. Click **Deploy**. `packages.txt` installs FFmpeg; `requirements.txt` installs the rest.

Cloud notes: the free machine has limited memory, so the app defaults to the `small` Whisper model.
YouTube sometimes blocks downloads from cloud servers. If you see a "Sign in to confirm you're not a bot"
error on the live app, run it locally (below) for that video.

## Run locally
### Requirements
- Python 3.10 to 3.12 (64-bit recommended)
- Internet access for YouTube and the first Whisper model download
- About 1 GB free space for `small`, or roughly 2 to 3 GB for `large-v3-turbo`

### Windows setup
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

## Accuracy notes
- Locally, use `large-v3-turbo` for the best practical Hindi/English/Hinglish result.
- Use Auto mode for code-switched Hinglish. Hindi mode favors Devanagari output.
- Whisper transcribes spoken content. It does not guarantee Roman-script Hinglish transliteration.
- Accuracy depends on audio clarity, accents, overlap, music, and names.

## Run tests
```bash
pip install -r requirements-dev.txt
pytest
```

Tests use mocks and do not download a YouTube video or Whisper model.
