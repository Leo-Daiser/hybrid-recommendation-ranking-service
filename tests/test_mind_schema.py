import pandas as pd
from src.data.mind_schema import (
    get_mind_behavior_columns,
    get_mind_news_columns,
    parse_behaviors_tsv,
    parse_news_tsv
)

def test_parse_behaviors_tsv_success(tmp_path):
    f = tmp_path / "behaviors.tsv"
    f.write_text("1\tU1\t11/11/2019 9:05:58 AM\tN1\tN2-1\n2\tU2\t11/12/2019\t\t\n")
    
    df = parse_behaviors_tsv(f)
    assert len(df) == 2
    assert list(df.columns) == get_mind_behavior_columns()
    assert df.iloc[0]["impression_id"] == 1
    assert df.iloc[0]["user_id"] == "U1"
    assert df.iloc[1]["impression_id"] == 2
    assert pd.isna(df.iloc[1]["history"])

def test_parse_news_tsv_success(tmp_path):
    f = tmp_path / "news.tsv"
    # item_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    f.write_text("N1\tsports\tfootball\tTitle 1\tAbstract 1\turl1\t[]\t[]\nN2\tnews\tworld\tTitle 2\t\turl2\t\t\n")
    
    df = parse_news_tsv(f)
    assert len(df) == 2
    assert list(df.columns) == get_mind_news_columns()
    assert df.iloc[0]["item_id"] == "N1"
    assert df.iloc[1]["item_id"] == "N2"
    assert df.iloc[1]["abstract"] == "" # NaN filled with ""
    assert df.iloc[1]["title_entities"] == "" # NaN filled with ""
