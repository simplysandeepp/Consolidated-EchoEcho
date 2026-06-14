from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

import backend.main as main
import backend.auth as auth
import backend.api_generator as api
from backend.api_generator import GenerationInput, KieAIConfigError, KieAIResponseError, submit_to_suno, validate_kieai_config


def test_frontend_pages_load() -> None:
    client = TestClient(main.app)

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "echo" in index_response.text.lower()

    for page in ("index", "login", "dashboard", "generate", "output"):
        response = client.get(f"/{page}.html")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    assert client.get("/no-such-page.html").status_code == 404


def test_auth_login_with_demo_credentials() -> None:
    client = TestClient(main.app)

    response = client.post("/api/auth/login", json={"email": "test@echo.com", "password": "test@123"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["token"]
    assert body["user"] == {"name": "Echo Tester", "email": "test@echo.com"}

    rejected = client.post("/api/auth/login", json={"email": "test@echo.com", "password": "wrong"})
    assert rejected.status_code == 401
    assert rejected.json()["detail"] == "Invalid email or password."

    closed = client.post(
        "/api/auth/signup",
        json={"name": "Anyone", "email": "new@user.com", "password": "abcdef"},
    )
    assert closed.status_code == 403
    assert "closed" in closed.json()["detail"]


def test_firebase_config_comes_from_env(monkeypatch) -> None:
    values = {
        "FIREBASE_API_KEY": "test-api-key",
        "FIREBASE_AUTH_DOMAIN": "test.firebaseapp.com",
        "FIREBASE_PROJECT_ID": "test-project",
        "FIREBASE_STORAGE_BUCKET": "test.firebasestorage.app",
        "FIREBASE_MESSAGING_SENDER_ID": "123456",
        "FIREBASE_APP_ID": "1:123456:web:abcdef",
        "FIREBASE_MEASUREMENT_ID": "G-TEST",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    client = TestClient(main.app)
    response = client.get("/api/firebase-config")

    assert response.status_code == 200
    assert response.json() == {
        "apiKey": "test-api-key",
        "authDomain": "test.firebaseapp.com",
        "projectId": "test-project",
        "storageBucket": "test.firebasestorage.app",
        "messagingSenderId": "123456",
        "appId": "1:123456:web:abcdef",
        "measurementId": "G-TEST",
    }


def test_firebase_config_requires_env(monkeypatch) -> None:
    for env_key in main.FIREBASE_ENV_MAP.values():
        monkeypatch.delenv(env_key, raising=False)

    client = TestClient(main.app)
    response = client.get("/api/firebase-config")

    assert response.status_code == 503
    assert "FIREBASE_API_KEY" in response.json()["detail"]


def test_compose_endpoint_returns_song_spec(monkeypatch) -> None:
    def fake_compose(payload: dict[str, object]) -> dict[str, object]:
        assert payload["mood"] == "Dreamy"
        return {
            "title": "Test Song",
            "key": "A minor",
            "chords": ["Am", "F", "C", "G"],
            "sections": [{"name": "Verse 1", "lines": [{"chord": "Am", "text": "hello"}]}],
            "vocal": {"pitch": 1.0, "rate": 1.0, "style": "soft"},
            "source": "test",
        }

    monkeypatch.setattr(main, "compose_song", fake_compose)
    client = TestClient(main.app)

    response = client.post("/api/compose", json={"mood": "Dreamy", "bpm": 90})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["song"]["title"] == "Test Song"
    assert body["song"]["sections"][0]["lines"][0]["text"] == "hello"


def test_composer_fallback_without_api_key(monkeypatch) -> None:
    from backend import composer

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    song = composer.compose_song({"mood": "Melancholy", "theme": "Rain", "bpm": 90})
    assert song["source"] == "fallback"
    assert song["chords"]
    assert song["sections"]
    assert all(line["text"] for section in song["sections"] for line in section["lines"])


def test_history_and_library_aliases_share_saved_inspirations(tmp_path: Path, monkeypatch) -> None:
    history_file = tmp_path / "song_history.json"
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "GENERATED_DIR", generated_dir)

    main.ensure_files()
    main.write_history(
        [
            {
                "song_id": "ABCD",
                "code": "ABCD",
                "task_id": "task_123",
                "created_at": "2026-06-09T00:00:00+00:00",
                "mood": "Dreamy",
                "theme": "Rain",
                "style": "Lo-fi",
                "instruments": ["Piano"],
                "tempo": 100,
                "energy": 4,
                "prompt": "test prompt",
                "original_audio_filename": "ECHO_ABCD_original.mp3",
                "trimmed_audio_filename": None,
            }
        ]
    )

    client = TestClient(main.app)

    for endpoint in ("/history", "/inspirations", "/library/refresh"):
        response = client.get(endpoint)
        assert response.status_code == 200
        body = response.json()
        assert body["songs"][0]["song_id"] == "ABCD"
        assert body["songs"][0]["code"] == "ABCD"
        assert body["songs"][0]["task_id"] == "task_123"
        assert body["songs"][0]["original_audio_url"] == "/generated/ECHO_ABCD_original.mp3"
        assert body["songs"][0]["trimmed_audio_url"] is None
        assert body["songs"][0]["download_url"] == "/download/ABCD/original"


def test_history_is_scoped_to_authenticated_user(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "USER_DATA_DIR", tmp_path / "users")
    monkeypatch.setattr(auth, "SESSIONS_FILE", tmp_path / "sessions.json")
    main.ensure_files()

    token_a = "token-a"
    token_b = "token-b"
    auth.SESSIONS_FILE.write_text(
        json.dumps({token_a: "one@example.com", token_b: "two@example.com"}),
        encoding="utf-8",
    )

    main.save_user_song(
        main.user_id_from_email("one@example.com"),
        {
            "song_id": "ABCD",
            "code": "ABCD",
            "title": "First User Song",
            "created_at": "2026-06-09T00:00:00+00:00",
            "original_audio_filename": "ABCD.wav",
        },
    )
    main.save_user_song(
        main.user_id_from_email("two@example.com"),
        {
            "song_id": "WXYZ",
            "code": "WXYZ",
            "title": "Second User Song",
            "created_at": "2026-06-09T00:00:00+00:00",
            "original_audio_filename": "WXYZ.wav",
        },
    )

    client = TestClient(main.app)
    first = client.get("/history", headers={"Authorization": f"Bearer {token_a}"}).json()["songs"]
    second = client.get("/history", headers={"Authorization": f"Bearer {token_b}"}).json()["songs"]

    assert [song["song_id"] for song in first] == ["ABCD"]
    assert [song["song_id"] for song in second] == ["WXYZ"]
    assert client.get("/song/WXYZ", headers={"Authorization": f"Bearer {token_a}"}).status_code == 404


def test_generate_endpoint_saves_generated_song(tmp_path: Path, monkeypatch) -> None:
    history_file = tmp_path / "song_history.json"
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "GENERATED_DIR", generated_dir)
    main.ensure_files()

    async def fake_generate_song(
        data: GenerationInput,
        generated_dir: Path,
        existing_ids: set[str],
    ) -> dict[str, object]:
        assert data.mood == "Calm"
        assert "WXYZ" not in existing_ids
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / "ECHO_WXYZ_original.mp3").write_bytes(b"ID3")
        return {
            "code": "WXYZ",
            "song_id": "WXYZ",
            "task_id": "task_fake",
            "created_at": "2026-06-09T00:00:00+00:00",
            "mood": data.mood,
            "theme": data.theme,
            "style": data.style,
            "instruments": data.instruments,
            "tempo": data.tempo,
            "energy": data.energy,
            "prompt": "fake prompt",
            "original_audio_filename": "ECHO_WXYZ_original.mp3",
            "trimmed_audio_filename": None,
        }

    monkeypatch.setattr(main, "generate_api_song", fake_generate_song)
    client = TestClient(main.app)

    response = client.post(
        "/generate",
        json={
            "mode": "api",
            "mood": "Calm",
            "theme": "Hope",
            "style": "Piano",
            "instruments": ["Piano"],
            "tempo": 90,
            "energy": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["success"] is True
    assert body["mode"] == "api"
    assert body["song"]["code"] == "WXYZ"
    assert body["song"]["song_id"] == "WXYZ"
    assert body["song"]["task_id"] == "task_fake"
    assert body["song"]["original_audio_url"] == "/generated/ECHO_WXYZ_original.mp3"
    assert body["song"]["trimmed_audio_url"] is None
    assert body["audio_url"] == "/generated/ECHO_WXYZ_original.mp3"
    assert "lyrics" in body
    assert "copyright" in body
    assert "lyrics" in body["song"]
    assert "copyright" in body["song"]
    assert client.get("/history").json()["songs"][0]["song_id"] == "WXYZ"


def test_copyright_check_endpoint_updates_history(tmp_path: Path, monkeypatch) -> None:
    history_file = tmp_path / "song_history.json"
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "GENERATED_DIR", generated_dir)
    main.ensure_files()
    main.write_history(
        [
            {
                "song_id": "ABCD",
                "code": "ABCD",
                "created_at": "2026-06-09T00:00:00+00:00",
                "prompt": "test prompt",
                "filename": "ABCD.wav",
                "audio_filename": "ABCD.wav",
                "lyrics": {"text": "", "structure": "unavailable"},
            }
        ]
    )

    client = TestClient(main.app)
    response = client.post(
        "/api/copyright/check",
        json={
            "song_id": "ABCD",
            "lyrics": "never gonna give you up never gonna let you down",
            "title": "ABCD",
            "prompt": "test prompt",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "risky"
    assert body["similarity_score"] >= 0.95
    saved_song = client.get("/history").json()["songs"][0]
    assert saved_song["copyright"]["status"] == "risky"


def test_generate_song_saves_original_mp3_without_trimming(tmp_path: Path, monkeypatch) -> None:
    async def fake_submit_to_suno(*_: object, **__: object) -> str:
        return "task_real"

    async def fake_poll_suno(*_: object, **__: object) -> dict[str, object]:
        return {"response": {"sunoData": [{"audioUrl": "https://example.com/full.mp3"}]}}

    async def fake_download_audio(_: object, __: str, destination: Path) -> None:
        destination.write_bytes(b"full mp3")

    def fail_if_trimmed(*_: object, **__: object) -> None:
        raise AssertionError("Generation should not trim automatically")

    monkeypatch.setenv("KIEAI_API_KEY", "test-key")
    monkeypatch.setattr(api, "submit_to_suno", fake_submit_to_suno)
    monkeypatch.setattr(api, "poll_suno", fake_poll_suno)
    monkeypatch.setattr(api, "download_audio", fake_download_audio)
    monkeypatch.setattr(api, "trim_audio", fail_if_trimmed)
    monkeypatch.setattr(api, "generate_song_id", lambda _: "ABCD")

    result = asyncio.run(
        api.generate_song(
            GenerationInput(
                mood="Calm",
                theme="Hope",
                style="Piano",
                instruments=["Piano"],
                tempo=90,
                energy=3,
            ),
            generated_dir=tmp_path,
            existing_ids=set(),
        )
    )

    assert result["code"] == "ABCD"
    assert result["song_id"] == "ABCD"
    assert result["task_id"] == "task_real"
    assert result["original_audio_filename"] == "ECHO_ABCD_original.mp3"
    assert result["trimmed_audio_filename"] is None
    assert (tmp_path / "ECHO_ABCD_original.mp3").read_bytes() == b"full mp3"
    assert not (tmp_path / "ECHO_ABCD_trimmed.mp3").exists()


def test_kieai_callback_endpoint_stores_payload_and_updates_history(tmp_path: Path, monkeypatch) -> None:
    history_file = tmp_path / "song_history.json"
    callback_file = tmp_path / "kiai_callbacks.json"
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "CALLBACK_FILE", callback_file)
    monkeypatch.setattr(main, "GENERATED_DIR", generated_dir)
    main.ensure_files()
    main.write_history(
        [
            {
                "song_id": "ABCD",
                "code": "ABCD",
                "task_id": "task_abc",
                "created_at": "2026-06-09T00:00:00+00:00",
                "mood": "Dreamy",
                "theme": "Rain",
                "style": "Lo-fi",
                "instruments": ["Piano"],
                "tempo": 100,
                "energy": 4,
                "prompt": "test prompt",
                "original_audio_filename": "ECHO_ABCD_original.mp3",
                "trimmed_audio_filename": None,
            }
        ]
    )

    client = TestClient(main.app)
    payload = {
        "code": 200,
        "msg": "All generated successfully.",
        "data": {
            "callbackType": "complete",
            "task_id": "task_abc",
            "data": [{"audio_url": "https://example.com/audio.mp3"}],
        },
    }

    response = client.post("/api/kieai/callback", json=payload)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "received"}
    callbacks = client.get("/api/kieai/callbacks").json()["callbacks"]
    assert callbacks[0]["task_id"] == "task_abc"
    song = client.get("/history").json()["songs"][0]
    assert song["callback_status"] == "complete"
    assert song["callback_payload"] == payload


def test_trim_endpoint_creates_trimmed_mp3_and_downloads(tmp_path: Path, monkeypatch) -> None:
    history_file = tmp_path / "song_history.json"
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "GENERATED_DIR", generated_dir)
    main.ensure_files()
    original = generated_dir / "ECHO_ABCD_original.mp3"
    original.write_bytes(b"original mp3")
    main.write_history(
        [
            {
                "code": "ABCD",
                "song_id": "ABCD",
                "task_id": "task_abc",
                "created_at": "2026-06-09T00:00:00+00:00",
                "mood": "Dreamy",
                "theme": "Rain",
                "style": "Lo-fi",
                "instruments": ["Piano"],
                "tempo": 100,
                "energy": 4,
                "prompt": "test prompt",
                "original_audio_filename": "ECHO_ABCD_original.mp3",
                "trimmed_audio_filename": None,
            }
        ]
    )

    def fake_trim_audio(source: Path, destination: Path, duration_seconds: int | None = None) -> None:
        assert source == original
        assert duration_seconds == 30
        destination.write_bytes(b"trimmed mp3")

    monkeypatch.setattr(main, "trim_audio", fake_trim_audio)
    client = TestClient(main.app)

    response = client.post("/api/library/ABCD/trim", json={"duration_seconds": 30, "force": False})

    assert response.status_code == 200
    body = response.json()
    assert body["song"]["trimmed_audio_filename"] == "ECHO_ABCD_trimmed.mp3"
    assert body["song"]["trimmed_audio_url"] == "/generated/ECHO_ABCD_trimmed.mp3"
    assert (generated_dir / "ECHO_ABCD_original.mp3").read_bytes() == b"original mp3"
    assert client.get("/download/ABCD/original").status_code == 200
    trimmed_download = client.get("/download/ABCD/trimmed")
    assert trimmed_download.status_code == 200
    assert trimmed_download.headers["content-type"].startswith("audio/mpeg")


def test_trim_endpoint_missing_ffmpeg_keeps_original(tmp_path: Path, monkeypatch) -> None:
    history_file = tmp_path / "song_history.json"
    generated_dir = tmp_path / "generated"
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "GENERATED_DIR", generated_dir)
    main.ensure_files()
    original = generated_dir / "ECHO_ABCD_original.mp3"
    original.write_bytes(b"original mp3")
    main.write_history(
        [
            {
                "code": "ABCD",
                "song_id": "ABCD",
                "task_id": "task_abc",
                "created_at": "2026-06-09T00:00:00+00:00",
                "mood": "Dreamy",
                "theme": "Rain",
                "style": "Lo-fi",
                "instruments": ["Piano"],
                "tempo": 100,
                "energy": 4,
                "prompt": "test prompt",
                "original_audio_filename": "ECHO_ABCD_original.mp3",
                "trimmed_audio_filename": None,
            }
        ]
    )

    def missing_ffmpeg(_: Path, __: Path, duration_seconds: int | None = None) -> None:
        raise RuntimeError("FFmpeg is required for trimming. Original audio is still available.")

    monkeypatch.setattr(main, "trim_audio", missing_ffmpeg)
    client = TestClient(main.app)

    response = client.post("/api/library/ABCD/trim", json={"duration_seconds": 30, "force": False})

    assert response.status_code == 500
    assert response.json()["detail"] == "FFmpeg is required for trimming. Original audio is still available."
    assert original.read_bytes() == b"original mp3"
    assert main.read_history()[0]["trimmed_audio_filename"] is None


def test_submit_to_suno_includes_callback_url(monkeypatch) -> None:
    captured_payload = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"code": 200, "data": {"taskId": "task_abc"}})

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://echoecho.ngrok-free.app")
    monkeypatch.setenv("KIEAI_CALLBACK_PATH", "/api/kieai/callback")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run() -> str:
        async with client:
            return await submit_to_suno(
                client,
                "test-key",
                GenerationInput(
                    mood="Calm",
                    theme="Hope",
                    style="Piano",
                    instruments=["Piano"],
                    tempo=90,
                    energy=3,
                ),
                "test prompt",
            )

    assert asyncio.run(run()) == "task_abc"
    assert captured_payload["callBackUrl"] == "https://echoecho.ngrok-free.app/api/kieai/callback"


def test_submit_to_suno_missing_task_id_is_friendly(monkeypatch) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 422, "msg": "Please enter callBackUrl.", "data": None})

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://echoecho.ngrok-free.app")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run() -> None:
        async with client:
            await submit_to_suno(
                client,
                "test-key",
                GenerationInput(
                    mood="Calm",
                    theme="Hope",
                    style="Piano",
                    instruments=["Piano"],
                    tempo=90,
                    energy=3,
                ),
                "test prompt",
            )

    try:
        asyncio.run(run())
    except KieAIResponseError as exc:
        assert "KieAI rejected the generation request" in str(exc)
    else:
        raise AssertionError("Expected KieAIResponseError")


def test_kieai_config_requires_public_base_url_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("KIEAI_API_KEY", "test-key")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)

    try:
        validate_kieai_config()
    except KieAIConfigError as exc:
        assert "PUBLIC_BASE_URL is required" in str(exc)
        assert "ngrok or cloudflared" in str(exc)
    else:
        raise AssertionError("Expected KieAIConfigError")
