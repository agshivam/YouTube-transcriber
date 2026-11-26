import shutil
import subprocess
import pytest
import transcriber as t

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def long_audio(tmp_path_factory):
    path = tmp_path_factory.mktemp("a") / "talk.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1500",
                    "-ac", "1", "-ar", "16000", str(path)], check=True)
    return path


def test_split_audio_into_10_minute_chunks_under_25mb(long_audio, tmp_path):
    assert abs(t.media_duration(long_audio) - 1500) < 1
    chunks = t.split_audio(long_audio, tmp_path)
    assert [round(o) for _, o in chunks] == [0, 600, 1200]
    assert all(p.stat().st_size < 25 * 1024 * 1024 for p, _ in chunks)
    assert abs(t.media_duration(chunks[-1][0]) - 300) < 1


def test_groq_merges_chunks_with_offsets_and_retries_429(long_audio):
    calls, sleeps, progress = [], [], []

    def post(chunk, key, lang):
        calls.append((chunk.name, lang))
        if len(calls) == 2:
            return 429, {"error": {"message": "Rate limit reached. Please try again in 1m2.5s."}}, {}
        n = len(calls)
        return 200, {"language": "hindi", "segments": [{"start": 1, "end": 3, "text": f" भाग {n} "}]}, {}

    r = t.transcribe_file_groq(long_audio, "k", "hi", title="Talk", progress=lambda d, tot, m: progress.append(m),
                               post=post, sleep=sleeps.append)
    assert sleeps == [62.5]
    assert r.clean_text.split("\n")[0].strip() == "भाग 1" and len(calls) == 4
    assert "00:10:01,000" in r.srt_text and "00:20:01,000" in r.srt_text
    assert r.detected_language == "hi" and "Groq" in r.source and r.srt_filename == "Talk.srt"
    assert any("free limit" in m for m in progress)


def test_groq_daily_limit_and_bad_key(long_audio):
    always_429 = lambda c, k, l: (429, {}, {"retry-after": "600"})
    with pytest.raises(t.TranscriptionError, match="daily limit"):
        t.transcribe_file_groq(long_audio, "k", post=always_429, sleep=lambda s: None, max_wait_per_chunk=1000)
    with pytest.raises(t.TranscriptionError, match="invalid"):
        t.transcribe_file_groq(long_audio, "k", post=lambda c, k, l: (401, {}, {}), sleep=lambda s: None)
    with pytest.raises(t.TranscriptionError, match="GROQ_API_KEY"):
        t.transcribe_file_groq(long_audio, "")


def test_bad_file_is_rejected(tmp_path):
    bad = tmp_path / "x.mp3"; bad.write_text("not audio")
    with pytest.raises(t.TranscriptionError):
        t.transcribe_file_groq(bad, "k", post=lambda c, k, l: (200, {}, {}))
