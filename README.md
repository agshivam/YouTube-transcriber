# 🎙️ YouTube Hindi / English / Hinglish Transcriber

**Live app:** https://youtube-transcript-live.streamlit.app  ·  Built with Python, Streamlit, yt-dlp and faster-whisper

Paste any YouTube link and get a clean transcript, a timestamped transcript and SRT subtitles.

## ⚡ Easiest: paste a link in Google Colab (free, hours of audio)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agshivam/YouTube-transcriber/blob/main/Colab_Transcriber.ipynb)

Open the link (works in Safari on iPad), tap ▶ on **Step 1**, paste YouTube link(s) in **Step 2**, tap ▶.
It downloads the audio itself and transcribes it with Whisper large-v3 on Google's free GPU,
then saves TXT, timestamps and SRT (ZIP download or Google Drive).
If YouTube blocks the download: **Runtime → Disconnect and delete runtime**, then run both steps again.

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
4. **Advanced settings → Python 3.12**. App URL: any free name (live one: `youtube-transcript-live`).
5. Click **Deploy**. `packages.txt` installs FFmpeg; `requirements.txt` installs the rest.

Cloud notes: the free machine has limited memory, so the app defaults to the `small` Whisper model.
The app installs Deno (via pip), which YouTube now requires for downloads, and retries with other
YouTube player clients if one is blocked.

### Groq: free, fast transcription of hours of audio (recommended)
Groq runs Whisper large-v3 (very good for Hindi and Hinglish) for free: about 8 hours of audio per day,
no card needed. The app cuts long audio into 10-minute parts automatically and waits when Groq's free
per-hour limit is reached, so even multi-hour recordings finish.
1. Sign up at https://console.groq.com, open **API Keys → Create API Key**, and copy it.
2. In Streamlit Cloud: **Manage app → ⋮ → Settings → Secrets**, paste `GROQ_API_KEY = "gsk_..."`, and save.

### Getting the audio on iPad (when YouTube blocks the live app)
YouTube blocks audio downloads from Streamlit's servers, so for videos without captions:
1. Save the video's audio to the **Files** app (any YouTube-to-MP3 website or app that saves to Files works;
   only download videos you have the right to use).
2. In the app choose **Source → Upload audio/video file**, pick the file, select Hindi, and tap Generate.
Files up to 500 MB are accepted (about 10+ hours of MP3 audio).

### Supadata (optional): small free quota, works directly from a link
YouTube blocks audio downloads from Streamlit's servers. Supadata fetches captions, or writes an AI
transcript from the audio, on its own servers.
1. Sign up at https://supadata.ai (free plan: 100 credits/month, no card) and copy your API key.
2. In Streamlit Cloud: **Manage app → ⋮ → Settings → Secrets**, paste `SUPADATA_API_KEY = "your-key"`, and save.

How the app uses it: YouTube captions in your language are fetched directly (free). If they're missing, or
"Skip YouTube captions" is ticked, Supadata makes an AI transcript. Cost: 1 credit for captions,
2 credits per video minute for an AI transcript (about 50 minutes of video per month on the free plan).

### Alternative: YouTube cookies (if YouTube blocks the live app) (HTTP 403 / "confirm you're not a bot")
YouTube blocks many cloud servers. Giving the app YouTube cookies usually fixes it:
1. On a laptop, open YouTube in a **private/incognito window** and sign in with a spare Google account
   (not your main one; heavy use can get an account flagged).
2. Export cookies for youtube.com in Netscape `cookies.txt` format (for example with the
   "Get cookies.txt LOCALLY" browser extension), then close the private window.
3. In Streamlit Cloud: **Manage app → Settings → Secrets**, and paste:
   ```toml
   YT_COOKIES = """
   (paste the whole cookies.txt content here)
   """
   ```
4. Save. The app restarts and uses the cookies. Cookies expire after some weeks; repeat when blocked again.

Running the app on your own computer avoids the block completely.

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
