from gazegesturekit.filters.one_euro import OneEuro
def test_one_euro_basic():
    f=OneEuro(); x=0.0
    for i in range(10): x=f(i*0.1)
    assert isinstance(x, float)
