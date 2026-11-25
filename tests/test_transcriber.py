from types import SimpleNamespace
from pathlib import Path
import pytest
import transcriber

class Segment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc123",
    "https://youtu.be/abc123",
    "https://youtube.com/shorts/abc123",
    "https://www.youtube.com/live/abc123",
])
def test_valid_urls(url):
    assert transcriber.validate_youtube_url(url) == url

@pytest.mark.parametrize("url", ["", "not a url", "https://example.com/watch?v=x", "https://youtube.com/watch"])
def test_invalid_urls(url):
    with pytest.raises(transcriber.TranscriptionError):
        transcriber.validate_youtube_url(url)

def test_timestamp_rounding_and_long_duration():
    assert transcriber.format_srt_timestamp(0) == "00:00:00,000"
    assert transcriber.format_srt_timestamp(3661.9996) == "01:01:02,000"
    assert transcriber.format_short_timestamp(3661.9) == "01:01:01"

def test_build_outputs_unicode_empty_segments_and_numbering():
    result = transcriber.build_outputs("हिंदी / English!", [
        Segment(0, 1.25, " नमस्ते "), Segment(1.25, 2, " "), Segment(62, 64, "hello भारत"),
    ], "hi")
    assert result.clean_text == "नमस्ते\nhello भारत"
    assert "[01:02] hello भारत" in result.timestamped_text
    assert "2\n00:01:02,000 --> 00:01:04,000\nhello भारत" in result.srt_text
    assert result.srt_text.endswith("\n")
    assert "/" not in result.txt_filename

def test_build_outputs_rejects_silence():
    with pytest.raises(transcriber.TranscriptionError, match="No speech"):
        transcriber.build_outputs("x", [Segment(0, 1, " ")])

def test_safe_stem_has_fallback_and_limit():
    assert transcriber.safe_stem("///") == "youtube_transcript"
    assert len(transcriber.safe_stem("a" * 200)) == 120

def test_end_to_end_orchestration_with_mock_download(monkeypatch, tmp_path):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFFfake")
    monkeypatch.setattr(transcriber, "_download_audio", lambda url, directory: (wav, "Demo Video"))
    class Model:
        def transcribe(self, path, **kwargs):
            assert path == str(wav)
            assert kwargs["language"] is None
            assert kwargs["beam_size"] == 5
            assert kwargs["vad_filter"] is True
            return iter([Segment(0, 2, "Namaste world")]), SimpleNamespace(language="hi")
    result = transcriber.transcribe_youtube("https://youtu.be/abc", Model())
    assert result.title == "Demo Video"
    assert result.clean_text == "Namaste world"
    assert result.detected_language == "hi"

def test_language_and_beam_validation(monkeypatch):
    with pytest.raises(transcriber.TranscriptionError):
        transcriber.transcribe_youtube("https://youtu.be/abc", object(), "fr")
    with pytest.raises(transcriber.TranscriptionError):
        transcriber.transcribe_youtube("https://youtu.be/abc", object(), None, 11)

def test_local_model_path_validation(tmp_path):
    with pytest.raises(transcriber.TranscriptionError, match="does not exist"):
        transcriber.load_whisper_model("small", str(tmp_path / "missing"))
