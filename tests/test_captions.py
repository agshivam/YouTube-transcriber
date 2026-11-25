import json
import pytest
import transcriber as t

def fmt(base, translated=False):
    q = "&tlang=hi" if translated else ""
    return [{"ext": "vtt", "url": f"https://yt/api?v=1{q}&fmt=vtt"},
            {"ext": "json3", "url": f"https://yt/api?v=1{q}&fmt=json3"}]

def test_manual_captions_beat_auto():
    info = {"subtitles": {"en": fmt("en")}, "automatic_captions": {"hi-orig": fmt("hi")}}
    track = t.pick_caption_track(info, None)
    assert track.language == "en" and "uploaded" in track.kind and track.ext == "json3"

def test_auto_mode_prefers_video_original_language():
    info = {"language": "en", "automatic_captions": {"hi": fmt("hi", True), "en-orig": fmt("en"), "en": fmt("en")}}
    track = t.pick_caption_track(info, None)
    assert track.language == "en" and "auto-generated" in track.kind

def test_translated_tracks_ignored_in_auto_mode():
    info = {"automatic_captions": {"hi": fmt("hi", True), "fr-orig": fmt("fr")}}
    assert t.pick_caption_track(info, None) is None

def test_translated_used_only_when_language_requested():
    info = {"automatic_captions": {"hi": fmt("hi", True), "en-orig": fmt("en")}}
    track = t.pick_caption_track(info, "hi")
    assert "auto-translated" in track.kind

def test_no_hi_en_captions_returns_none():
    assert t.pick_caption_track({"subtitles": {"fr": fmt("fr")}}, None) is None
    assert t.pick_caption_track({}, "en") is None

def test_parse_json3_skips_blank_events():
    raw = json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 1500, "segs": [{"utf8": "नमस्ते "}, {"utf8": "दोस्तों"}]},
        {"tStartMs": 1500, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 2000},
        {"tStartMs": 62000, "dDurationMs": 2000, "segs": [{"utf8": "hello world"}]},
    ]})
    segs = t.parse_json3(raw)
    assert [s.text for s in segs] == ["नमस्ते दोस्तों", "hello world"]
    assert segs[1].start == 62 and segs[1].end == 64

def test_parse_vtt_dedupes_rolling_lines_and_tags():
    raw = ("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello<00:00:01.000><c> there</c>\n\n"
           "00:00:02.000 --> 00:00:04.000\nhello there\nhow are you\n\n"
           "01:00:00.500 --> 01:00:01.000 align:start\nbye &amp; thanks\n")
    segs = t.parse_vtt(raw)
    assert [s.text for s in segs] == ["hello there", "how are you", "bye & thanks"]
    assert segs[2].start == 3600.5

def test_fetch_captions_end_to_end(monkeypatch):
    info = {"title": "Demo", "subtitles": {"hi": fmt("hi")}}
    monkeypatch.setattr(t, "_extract_info", lambda url: info)
    monkeypatch.setattr(t, "_download_text", lambda u: json.dumps(
        {"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "राम राम"}]}]}))
    result = t.fetch_youtube_captions("https://youtu.be/abc")
    assert result.clean_text == "राम राम" and result.detected_language == "hi"
    assert "YouTube captions" in result.source and result.srt_filename == "Demo.srt"

def test_fetch_captions_returns_none_for_whisper_fallback(monkeypatch):
    monkeypatch.setattr(t, "_extract_info", lambda url: {"title": "x", "subtitles": {}})
    assert t.fetch_youtube_captions("https://youtu.be/abc") is None
    monkeypatch.setattr(t, "_extract_info", lambda url: {"subtitles": {"en": fmt("en")}})
    def boom(u): raise OSError("network")
    monkeypatch.setattr(t, "_download_text", boom)
    assert t.fetch_youtube_captions("https://youtu.be/abc") is None

def test_fetch_captions_validates_input():
    with pytest.raises(t.TranscriptionError):
        t.fetch_youtube_captions("https://example.com/x")
    with pytest.raises(t.TranscriptionError):
        t.fetch_youtube_captions("https://youtu.be/abc", "fr")

def test_options_include_js_runtime_and_cookies(monkeypatch, tmp_path):
    c = tmp_path / "c.txt"; c.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("YT_COOKIES_FILE", str(c))
    o = t._ydl_options(["tv"], skip_download=True)
    assert o["cookiefile"] == str(c) and o["skip_download"]
    assert o["extractor_args"]["youtube"]["player_client"] == ["tv"]
    assert "remote_components" in o

def test_extract_info_retries_other_clients_when_blocked(monkeypatch):
    import yt_dlp
    calls = []
    class FakeYDL:
        def __init__(self, opts): self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download):
            calls.append(self.opts.get("extractor_args"))
            if len(calls) < 3: raise Exception("HTTP Error 403: Forbidden")
            return {"title": "ok"}
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    assert t._extract_info("https://youtu.be/abc") == {"title": "ok"}
    assert len(calls) == 3

def test_extract_info_stops_on_non_block_error(monkeypatch):
    import yt_dlp
    class FakeYDL:
        def __init__(self, opts): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download): raise Exception("Video is private")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    with pytest.raises(t.TranscriptionError, match="private"):
        t._extract_info("https://youtu.be/abc")

def test_caption_notes_explain_why(monkeypatch):
    monkeypatch.setattr(t, "_extract_info", lambda url: {"subtitles": {"fr": fmt("fr")}})
    notes = []
    assert t.fetch_youtube_captions("https://youtu.be/abc", None, notes) is None
    assert "no Hindi/English captions" in notes[0] and "fr" in notes[0]
    monkeypatch.setattr(t, "_extract_info", lambda url: {"subtitles": {"hi": fmt("hi")}})
    monkeypatch.setattr(t, "_download_text", lambda u: json.dumps({"events": []}))
    notes = []
    assert t.fetch_youtube_captions("https://youtu.be/abc", None, notes) is None
    assert "empty" in notes[0]
