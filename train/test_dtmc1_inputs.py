import torch

from dtmc1_inputs import DraftExample, render_draft_source, tokenize_draft_sources
from typed_microcode_graph import compile_typed_graph
from natural_microcode_program import parse_program


class TinyTokenizer:
    chat_template = None

    def __call__(self, rendered, **_kwargs):
        widths = [len(text) for text in rendered]
        maximum = max(widths)
        ids = []
        masks = []
        offsets = []
        for text in rendered:
            pad = maximum - len(text)
            ids.append([0] * pad + [ord(char) % 251 + 1 for char in text])
            masks.append([0] * pad + [1] * len(text))
            offsets.append([(0, 0)] * pad + [(i, i + 1) for i in range(len(text))])
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
            "offset_mapping": torch.tensor(offsets),
        }


def _example() -> DraftExample:
    graph = compile_typed_graph(
        "A has 12 items and gets 3 more.",
        parse_program("<MICROCODE_V1>\nR0 P:12 P:3 A\nC:0\n</MICROCODE_V1>"),
    )
    return DraftExample("a" * 64, graph, "12 + 3 = 15. #### 15", True, False)


def test_source_pointer_mask_excludes_draft_numbers() -> None:
    tokenizer = TinyTokenizer()
    example = _example()
    encoded, mask, receipt = tokenize_draft_sources(
        tokenizer, [example], torch.device("cpu"), 1024
    )
    rendered, source_start = render_draft_source(
        tokenizer, example.graph.source, example.draft
    )
    source_end = source_start + len(example.graph.source)
    selected = mask[0].any(0).nonzero().flatten().tolist()
    nonpadding = encoded["attention_mask"][0].nonzero().flatten()[0].item()
    selected_offsets = [index - nonpadding for index in selected]
    assert selected_offsets
    assert all(source_start <= offset < source_end for offset in selected_offsets)
    assert receipt["maximum_tokens"] == len(rendered)


def test_source_boundary_survives_chat_template_trailing_trim() -> None:
    class TrimmingTokenizer(TinyTokenizer):
        chat_template = "present"

        def apply_chat_template(self, messages, **_kwargs):
            return "HEADER\n" + messages[-1]["content"].rstrip() + "\nASSISTANT"

    example = _example()
    rendered, source_start = render_draft_source(
        TrimmingTokenizer(), example.graph.source, example.draft + " "
    )
    assert rendered[source_start : source_start + len(example.graph.source)] == (
        example.graph.source
    )
