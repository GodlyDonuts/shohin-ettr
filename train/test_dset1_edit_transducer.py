import pytest
import torch

from dset1_edit_transducer import (
    DSET1Error,
    KEEP,
    REPLACE_LAST,
    execute_script,
    normalized_script_loss,
    parse_script,
    render_script,
)


def test_keep_round_trip_copies_exactly() -> None:
    text = render_script(KEEP)
    assert text == "<KEEP>\n"
    assert execute_script("draft\n", parse_script(text)) == "draft\n"


def test_replace_last_changes_only_last_surface() -> None:
    text = render_script(REPLACE_LAST, "3", "4")
    assert execute_script("3 then answer 3", parse_script(text)) == "3 then answer 4"


@pytest.mark.parametrize(
    "text",
    ["", "<KEEP>\nextra", "<REPLACE_LAST>\nold", "<OTHER>\na\nb"],
)
def test_parser_fails_closed(text: str) -> None:
    with pytest.raises(DSET1Error):
        parse_script(text)


def test_executor_rejects_missing_old_surface() -> None:
    with pytest.raises(DSET1Error):
        execute_script("draft", parse_script("<REPLACE_LAST>\nold\nnew"))


def test_normalized_loss_weights_rows_equally() -> None:
    logits = torch.zeros(2, 5, 3, requires_grad=True)
    labels = torch.tensor([[-100, 0, 1, -100, -100], [-100, 2, 1, 0, 2]])
    loss = normalized_script_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad).item() > 0
