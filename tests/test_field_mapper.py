import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from db.schema import migrate
from field_mapper import map_field, resolve_model
from pipeline.finetune_openai import BASE_MODEL, FT_MODEL_ID_FILE

FT_MODEL = "ft:gpt-4o-mini:job-agent:abc123"
RESUME   = '{"name": "Jane Doe"}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "runs.db"
    migrate(str(p))
    return p


@pytest.fixture()
def ft_model_file(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / FT_MODEL_ID_FILE
    p.write_text(FT_MODEL)
    monkeypatch.chdir(tmp_path)
    return p


@pytest.fixture()
def no_model_file(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_client(answer: str = "Jane") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    return client


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    def test_uses_ft_model_when_file_exists(self, ft_model_file):
        model_id, source = resolve_model()
        assert model_id == FT_MODEL
        assert source == "ft"

    def test_uses_base_model_when_no_file(self, no_model_file):
        model_id, source = resolve_model()
        assert model_id == BASE_MODEL
        assert source == "base"

    def test_force_base_overrides_ft_file(self, ft_model_file):
        model_id, source = resolve_model(force_base=True)
        assert model_id == BASE_MODEL
        assert source == "base"

    def test_uses_base_when_file_is_empty(self, tmp_path, monkeypatch):
        p = tmp_path / FT_MODEL_ID_FILE
        p.write_text("")
        monkeypatch.chdir(tmp_path)
        model_id, source = resolve_model()
        assert model_id == BASE_MODEL
        assert source == "base"


# ---------------------------------------------------------------------------
# map_field — return value
# ---------------------------------------------------------------------------

class TestMapFieldReturnValue:
    def test_returns_value(self, ft_model_file, db):
        client = _make_client("Jane Doe")
        with patch("field_mapper._fire_shadow"):
            result = map_field("First Name", RESUME, "greenhouse", str(db), client=client)
        assert result["value"] == "Jane Doe"

    def test_returns_model_id(self, ft_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            result = map_field("First Name", RESUME, "greenhouse", str(db), client=client)
        assert result["model_id"] == FT_MODEL

    def test_returns_model_source_ft(self, ft_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            result = map_field("First Name", RESUME, "greenhouse", str(db), client=client)
        assert result["model_source"] == "ft"

    def test_returns_model_source_base_when_no_file(self, no_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            result = map_field("First Name", RESUME, "greenhouse", str(db), client=client)
        assert result["model_source"] == "base"

    def test_force_base_returns_base_source(self, ft_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            result = map_field("First Name", RESUME, "greenhouse", str(db),
                               force_base=True, client=client)
        assert result["model_source"] == "base"

    def test_strips_whitespace_from_value(self, no_model_file, db):
        client = _make_client("  Jane  ")
        with patch("field_mapper._fire_shadow"):
            result = map_field("First Name", RESUME, "greenhouse", str(db), client=client)
        assert result["value"] == "Jane"


# ---------------------------------------------------------------------------
# map_field — API call
# ---------------------------------------------------------------------------

class TestMapFieldAPICall:
    def test_calls_api_with_resolved_model(self, ft_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            map_field("Email", RESUME, "workday", str(db), client=client)
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["model"] == FT_MODEL

    def test_calls_api_with_temperature_zero(self, no_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            map_field("Email", RESUME, "workday", str(db), client=client)
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0

    def test_user_message_contains_platform(self, no_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            map_field("Email", RESUME, "icims", str(db), client=client)
        msgs = client.chat.completions.create.call_args[1]["messages"]
        assert "icims" in msgs[1]["content"]

    def test_user_message_contains_field_label(self, no_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow"):
            map_field("Salary Expectation", RESUME, "lever", str(db), client=client)
        msgs = client.chat.completions.create.call_args[1]["messages"]
        assert "Salary Expectation" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Shadow fire-and-forget
# ---------------------------------------------------------------------------

class TestShadowFireAndForget:
    def test_shadow_fired_when_ft_model_active(self, ft_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow") as mock_fire:
            map_field("Email", RESUME, "greenhouse", str(db), client=client)
        mock_fire.assert_called_once()

    def test_shadow_not_fired_when_base_model(self, no_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow") as mock_fire:
            map_field("Email", RESUME, "greenhouse", str(db), client=client)
        mock_fire.assert_not_called()

    def test_shadow_not_fired_when_force_base(self, ft_model_file, db):
        client = _make_client()
        with patch("field_mapper._fire_shadow") as mock_fire:
            map_field("Email", RESUME, "greenhouse", str(db),
                      force_base=True, client=client)
        mock_fire.assert_not_called()
