import pytest
import transcriber as t

URL = "https://youtu.be/Jt-cZQjUdaY?si=x"

def fake_http(responses, seen):
    def get(url, key):
        seen.append(url); assert key == "k"
        return responses.pop(0)
    return get

def test_direct_200_builds_transcript():
    seen = []
    http = fake_http([(200, {"lang": "hi", "content": [
        {"text": "नमस्ते", "offset": 0, "duration": 1500}, {"text": "दोस्तों", "offset": 1500, "duration": 1000}]})], seen)
    r = t.fetch_supadata_transcript(URL, "k", "hi", "generate", http_get=http, title="My video")
    assert r.clean_text.split() == ["नमस्ते", "दोस्तों"] and r.detected_language == "hi"
    assert r.srt_filename == "My video.srt" and "AI" in r.source
    assert "mode=generate" in seen[0] and "lang=hi" in seen[0] and "youtu.be" in seen[0]

def test_async_job_is_polled():
    seen = []
    http = fake_http([(202, {"jobId": "j1"}), (200, {"status": "active"}),
                      (200, {"status": "completed", "lang": "hi", "content": [{"text": "राम", "offset": 0, "duration": 900}]})], seen)
    r = t.fetch_supadata_transcript(URL, "k", "hi", "generate", http_get=http, sleep=lambda s: None)
    assert r.clean_text == "राम" and seen[-1].endswith("/transcript/j1")

def test_failed_job_and_errors():
    http = fake_http([(202, {"jobId": "j"}), (200, {"status": "failed", "error": "boom"})], [])
    with pytest.raises(t.TranscriptionError, match="boom"):
        t.fetch_supadata_transcript(URL, "k", "hi", "generate", http_get=http, sleep=lambda s: None)
    for code, msg in [(206, "no captions"), (404, "private"), (429, "credits"), (500, "failed")]:
        with pytest.raises(t.TranscriptionError, match=msg):
            t.fetch_supadata_transcript(URL, "k", None, "auto", http_get=fake_http([(code, {})], []))

def test_text_content_and_language_note():
    notes = []
    r = t.fetch_supadata_transcript(URL, "k", "hi", "auto", notes,
                                    http_get=fake_http([(200, {"lang": "en", "content": "hello there"})], []))
    assert r.clean_text == "hello there" and "instead of Hindi" in notes[0]

def test_empty_and_bad_inputs():
    with pytest.raises(t.TranscriptionError, match="No speech"):
        t.fetch_supadata_transcript(URL, "k", None, "auto", http_get=fake_http([(200, {"content": []})], []))
    with pytest.raises(t.TranscriptionError):
        t.fetch_supadata_transcript("https://example.com", "k")
    with pytest.raises(t.TranscriptionError):
        t.fetch_supadata_transcript(URL, "k", mode="bad")
