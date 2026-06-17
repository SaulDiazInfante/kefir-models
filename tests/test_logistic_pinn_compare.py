from kefir_models import logistic_pinn_compare


def test_parse_args_help(capsys):
    """Test that the logistic PINN parser exposes help text."""

    try:
        logistic_pinn_compare.parse_args(["--help"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "deterministic and stochastic logistic PINN" in captured.out
