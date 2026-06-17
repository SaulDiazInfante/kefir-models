import pandas as pd

from kefir_models import plot_all_models, plot_style


def test_parse_args_help(capsys):
    """The all-model plotting CLI exposes help text."""

    try:
        plot_all_models.parse_args(["--help"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "step-by-step comparison across all kefir models" in captured.out


def test_infer_trial_columns():
    """Trial names are inferred from observed columns."""

    frame = pd.DataFrame(
        {
            "time": [0.0],
            "trial_1_observed": [1.0],
            "trial_2_observed": [2.0],
            "observed_mean": [1.5],
        },
    )

    assert plot_all_models.infer_trial_columns(frame) == ["trial_1", "trial_2"]


def test_save_fixed_layout_png_uses_one_tight_crop_box(tmp_path):
    """Sequence frames use one fixed tight crop instead of recropping per legend."""

    class FakeFigure:
        def __init__(self):
            self.size = None
            self.savefig_args = None
            self.savefig_kwargs = None

        def set_size_inches(self, *args, **kwargs):
            self.size = (args, kwargs)

        def savefig(self, *args, **kwargs):
            self.savefig_args = args
            self.savefig_kwargs = kwargs

    figure = FakeFigure()
    path = tmp_path / "frame.png"

    plot_style.save_fixed_layout_png(figure, path)

    assert figure.size == ((11.5, 5.9), {"forward": True})
    assert figure.savefig_args == (path,)
    assert figure.savefig_kwargs["dpi"] == 300
    assert figure.savefig_kwargs["bbox_inches"] != "tight"
    assert tuple(figure.savefig_kwargs["bbox_inches"].extents) == (
        *plot_style.FIXED_TIGHT_BBOX_INCHES,
    )
