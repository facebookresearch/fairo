import mrp


def test_import():
    assert mrp.defined_processes == {}

    mrp.import_msetup("./test_data/")
    assert list(mrp.defined_processes.keys()) == ["proc1", "proc2"]

    mrp.defined_processes.clear()
    mrp.import_msetup("./test_data/msetup.py")
    assert list(mrp.defined_processes.keys()) == ["proc1", "proc2"]

    mrp.defined_processes.clear()
    mrp.import_msetup("./test_data/", processes=["proc1"])
    assert list(mrp.defined_processes.keys()) == ["proc1"]

    mrp.defined_processes.clear()
    mrp.import_msetup("./test_data/", processes=["proc2"])
    assert list(mrp.defined_processes.keys()) == ["proc2"]

    mrp.defined_processes.clear()
    mrp.import_msetup("./test_data/", processes=["proc1", "proc2"])
    assert list(mrp.defined_processes.keys()) == ["proc1", "proc2"]
