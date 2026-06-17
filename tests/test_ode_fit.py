from kefir_models import ode_fit


def test_parse_args_help(capsys):
    """Test that the parser can be instantiated and help can be requested."""
    try:
        ode_fit.parse_args(["--help"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "Fit water kefir trial data" in captured.out
