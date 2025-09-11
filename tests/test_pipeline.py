def test_export(export_fixture):
    assert export_fixture.exists()


def test_compile(compile_fixture):
    assert compile_fixture.exists()


def test_validate_vmfb(validate_vmfb_fixture):
    assert validate_vmfb_fixture.exists()


def test_benchmark(benchmark_fixture):
    assert benchmark_fixture.exists()


def test_serving(serving_fixture):
    assert serving_fixture.exists()
