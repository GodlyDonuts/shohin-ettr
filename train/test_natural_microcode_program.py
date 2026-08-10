from fractions import Fraction

from learned_arithmetic_microcode import LearnedDigitMicrocode
from natural_microcode_program import (
    compile_gsm8k_answer,
    execute_fraction,
    execute_learned,
    parse_program,
    render_program,
    result_fields_absent,
)
from train_lam1_microcode import candidate_fraction


def perfect_microcode() -> LearnedDigitMicrocode:
    model = LearnedDigitMicrocode()
    for parameter in model.parameters():
        parameter.data.zero_()
    for _ in range(32):
        optimizer = __import__("torch").optim.SGD(model.parameters(), lr=1.0)
        optimizer.zero_grad()
        model.transition_loss().backward()
        optimizer.step()
    model.freeze_discrete()
    return model


def test_compiles_result_free_register_program() -> None:
    answer = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips. "
        "Altogether 48+24 = <<48+24=72>>72.\n#### 72"
    )
    program, final = compile_gsm8k_answer(answer)
    text = render_program(program)
    assert "L:0" in text
    assert "=24" not in text and "=72" not in text
    assert result_fields_absent(text, "72")
    assert parse_program(text) == program
    assert execute_fraction(program) == final == Fraction(72)


def test_learned_execution_matches_fraction() -> None:
    answer = "First 16-3-4 = <<16-3-4=9>>9. Then 9*2 = <<9*2=18>>18.\n#### 18"
    program, final = compile_gsm8k_answer(answer)
    predicted = candidate_fraction(execute_learned(perfect_microcode(), program))
    assert predicted == final
