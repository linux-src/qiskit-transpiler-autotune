import csv

from qta.runner import CsvLog


def test_csv_log_resumes(tmp_path):
    path = tmp_path / "log.csv"
    log = CsvLog(path, ["a", "b", "value"], key=["a", "b"])
    log.write({"a": "x", "b": 1, "value": 0.5})
    assert log.has(a="x", b=1)

    reopened = CsvLog(path, ["a", "b", "value"], key=["a", "b"])
    assert reopened.has(a="x", b=1)
    assert not reopened.has(a="x", b=2)
    reopened.write({"a": "x", "b": 2, "value": 1.5, "extra": "ignored"})

    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["b"] for r in rows] == ["1", "2"]
