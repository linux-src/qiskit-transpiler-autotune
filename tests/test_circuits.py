from qta.circuits import load_suite


def test_suite_cache_roundtrip(tmp_path):
    suite = [("qft", 4), ("graphstate", 6)]
    first = load_suite(tmp_path, suite)
    second = load_suite(tmp_path, suite)
    assert list(first) == ["qft_4", "graphstate_6"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["graphstate_6.qpy", "qft_4.qpy"]
    for cid in first:
        assert first[cid] == second[cid]
        assert first[cid].num_qubits == int(cid.rsplit("_", 1)[1])
