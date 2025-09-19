def test_export(export_fixture):
    assert export_fixture.exists()


def test_compile(compile_fixture):
    assert export_fixture.exists()


def test_validate_vmfb(validate_vmfb_fixture):
    assert validate_vmfb_fixture.exists()


def test_benchmark(benchmark_fixture):
    assert benchmark_fixture.exists()


def test_online_serving(online_serving_fixture):
    assert online_serving_fixture.exists()
