import pandas as pd

from model.kenpom_day import grade, record


def test_grade_sides_and_push():
    pred = pd.Series([5.0, -3.0, 4.0, 2.0])
    line = pd.Series([3.0, 1.0, 4.0, 2.0])
    actual = pd.Series([10.0, 4.0, 1.0, 2.0])
    g = grade(pred, line, actual)
    # home side wins; away side loses (home covered 1); pred == line: no bet
    assert list(g["result"][:2]) == ["W", "L"]
    assert g["result"][2:].isna().all()
    g2 = grade(pd.Series([6.0]), pd.Series([3.0]), pd.Series([3.0]))
    assert g2["result"][0] == "P"
    assert record(pd.DataFrame({"result": ["W", "W", "L", "P"]})).startswith("2-1-1")
