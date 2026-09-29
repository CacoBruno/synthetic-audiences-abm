from pathlib import Path

from synthetic_audiences.population import generate_profiles


def test_generate_profiles_shape_and_id():
    df = generate_profiles(n=25, seed=123)
    assert len(df) == 25
    assert "id_persona" in df.columns
    assert df["id_persona"].is_unique
    assert "segmento_e_bolhas" in df.columns
    assert "Persona" in df.columns


def test_generate_profiles_csv(tmp_path: Path):
    out = tmp_path / "agents.csv"
    df = generate_profiles(n=10, output_csv=out, seed=1)
    assert out.exists()
    assert len(df) == 10
