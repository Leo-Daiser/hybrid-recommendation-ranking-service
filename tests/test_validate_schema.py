import pytest
import pandas as pd
from src.data.validate_schema import (
    validate_required_columns, validate_non_empty, validate_unique_key,
    validate_foreign_key_relationship, validate_raw_tables
)

def test_validate_required_columns_missing_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        validate_required_columns("t1", df, ["a", "b"])

def test_validate_non_empty_raises_on_empty_table():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_non_empty("t1", df)

def test_validate_unique_key_raises_on_duplicates():
    df = pd.DataFrame({"id": [1, 1]})
    with pytest.raises(ValueError):
        validate_unique_key("t1", df, ["id"])

def test_validate_foreign_key_relationship_strict_raises_on_orphans():
    child = pd.DataFrame({"fk": [1, 2]})
    parent = pd.DataFrame({"pk": [1]})
    with pytest.raises(ValueError):
        validate_foreign_key_relationship(child, "fk", parent, "pk", "rel", strict=True)

def test_validate_foreign_key_relationship_non_strict_returns_report():
    child = pd.DataFrame({"fk": [1, 2]})
    parent = pd.DataFrame({"pk": [1]})
    report = validate_foreign_key_relationship(child, "fk", parent, "pk", "rel", strict=False)
    assert report["orphan_count"] == 1
    assert report["sample_orphans"] == [2]

def test_validate_raw_tables_success():
    config = {
        "tables": {
            "movies": {
                "required_columns": ["movieId", "title"],
                "unique_key": ["movieId"]
            },
            "ratings": {
                "required_columns": ["userId", "movieId", "rating"]
            }
        }
    }
    tables = {
        "movies": pd.DataFrame({"movieId": [1], "title": ["A"]}),
        "ratings": pd.DataFrame({"userId": [1], "movieId": [1], "rating": [5.0], "timestamp": [123]})
    }
    report = validate_raw_tables(tables, config, strict_foreign_keys=True)
    assert "movies" in report["tables"]
    assert "ratings" in report["tables"]
    assert not report["warnings"]

def test_validate_raw_tables_missing_movie_id_in_movies():
    config = {"tables": {"movies": {"required_columns": ["movieId"]}}}
    tables = {"movies": pd.DataFrame({"title": ["A"]})}
    with pytest.raises(ValueError):
        validate_raw_tables(tables, config)

def test_validate_raw_tables_duplicate_movie_key():
    config = {"tables": {"movies": {"unique_key": ["movieId"]}}}
    tables = {"movies": pd.DataFrame({"movieId": [1, 1]})}
    with pytest.raises(ValueError):
        validate_raw_tables(tables, config)

def test_validate_raw_tables_rating_movie_fk_warning():
    config = {
        "tables": {
            "movies": {"required_columns": ["movieId"]},
            "ratings": {"required_columns": ["movieId"]}
        }
    }
    tables = {
        "movies": pd.DataFrame({"movieId": [1]}),
        "ratings": pd.DataFrame({"movieId": [1, 2], "rating": [4.0, 4.0], "timestamp": [123, 123]})
    }
    report = validate_raw_tables(tables, config, strict_foreign_keys=False)
    assert report["foreign_keys"]["ratings.movieId -> movies.movieId"]["orphan_count"] == 1
