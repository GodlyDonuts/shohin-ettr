"""Target-free document masking for fixed-width token-native transports.

ETTR transports place one complete syntax tree first and deterministic cover
codewords afterward.  The ordinary attention mask deliberately covers the
whole transport, so semantic compilers otherwise treat cover as source.  This
router recovers only the tree boundary from the public token-native grammar.
It does not decode ontology symbols, inspect targets, or execute a program.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import torch
import torch.nn as nn


MAX_NATIVE_ARITY = 32
MAX_HEAD = 15
CALL_STRIDE = MAX_NATIVE_ARITY + 1
CALL_END = (MAX_HEAD + 1) * CALL_STRIDE
FRAME_A = CALL_END
FRAME_B = FRAME_A + 1
FRAME_END = FRAME_B + 1
FRAME_FILL = FRAME_END + 1
REIFY_BASE = FRAME_FILL + 1
REIFY_END = REIFY_BASE + MAX_NATIVE_ARITY + 1
INTEGER_BASE = REIFY_END
ROOT_CODES = (
    14 * CALL_STRIDE + 3,
    15 * CALL_STRIDE + 2,
)


class TokenNativeSyntaxRouterError(ValueError):
    """A token sequence cannot be routed under the public grammar."""


class TokenNativeDocumentMask(nn.Module):
    """Recover the exact leading AST span and delete transport cover."""

    def __init__(
        self,
        codebook_token_ids: Sequence[int],
        *,
        vocab_size: int,
    ) -> None:
        super().__init__()
        ids = tuple(int(value) for value in codebook_token_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not isinstance(vocab_size, int)
            or vocab_size < 1
            or min(ids) < 0
            or max(ids) >= vocab_size
            or len(ids) <= REIFY_END
        ):
            raise TokenNativeSyntaxRouterError("token-native codebook geometry differs")
        inverse = torch.full((vocab_size,), -1, dtype=torch.long)
        inverse[torch.tensor(ids, dtype=torch.long)] = torch.arange(
            len(ids),
            dtype=torch.long,
        )
        self.register_buffer("inverse_codebook", inverse)

    def forward(
        self,
        tokens: torch.Tensor,
        transport_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tokens.ndim != 2
            or tokens.dtype != torch.long
            or transport_mask.shape != tokens.shape
            or transport_mask.dtype != torch.bool
            or tokens.shape[1] < 3
            or tokens.device != transport_mask.device
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native transport geometry differs"
            )
        codes = self.inverse_codebook[tokens]
        torch._assert_async(
            codes.ge(0).all(),
            "token-native transport leaves the bound codebook",
        )
        preamble = (codes[:, 0].eq(FRAME_A) | codes[:, 0].eq(FRAME_B)) & (
            codes[:, 1].eq(FRAME_A) | codes[:, 1].eq(FRAME_B)
        )
        torch._assert_async(
            preamble.all(),
            "token-native transport preamble differs",
        )

        body = codes[:, 2:]
        call = body.lt(CALL_END)
        arity = body.remainder(CALL_STRIDE)
        reified = body.ge(REIFY_BASE) & body.lt(REIFY_END)
        reified_arity = body - REIFY_BASE

        prefix_effect = torch.full_like(body, -1)
        prefix_effect = torch.where(call, arity - 1, prefix_effect)
        prefix_effect = torch.where(reified, reified_arity, prefix_effect)
        remaining = 1 + prefix_effect.cumsum(dim=1)
        prefix_complete = remaining.eq(0)

        postfix_effect = torch.ones_like(body)
        postfix_effect = torch.where(call, 1 - arity, postfix_effect)
        postfix_effect = torch.where(reified, -reified_arity, postfix_effect)
        stack = postfix_effect.cumsum(dim=1)
        root = torch.zeros_like(body, dtype=torch.bool)
        for root_code in ROOT_CODES:
            root |= body.eq(root_code)
        postfix_complete = stack.eq(1) & root

        is_prefix = codes[:, 0].eq(FRAME_A)
        completion = torch.where(
            is_prefix[:, None],
            prefix_complete,
            postfix_complete,
        )
        has_completion = completion.any(dim=1)
        torch._assert_async(
            has_completion.all(),
            "token-native syntax tree does not terminate",
        )
        terminal_index = completion.to(torch.int64).argmax(dim=1) + 2
        document_mask = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            <= terminal_index[:, None]
        )
        routed = transport_mask & document_mask
        torch._assert_async(
            routed.sum(dim=1).eq(terminal_index + 1).all(),
            "token-native source mask truncates the syntax tree",
        )
        return routed


def _cover_indices(
    document_payload: bytes,
    *,
    width: int,
    count: int,
    codebook_size: int,
) -> tuple[int, ...]:
    seed = hashlib.sha256(
        b"R12-ETTR-IL-v2\0transport-cover\0"
        + document_payload
        + width.to_bytes(4, "big")
    ).digest()
    result: list[int] = []
    counter = 0
    while len(result) < count:
        block = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
        for offset in range(0, len(block), 2):
            result.append(
                int.from_bytes(block[offset : offset + 2], "big")
                % codebook_size
            )
            if len(result) == count:
                break
    return tuple(result)


def _cover_verified_document_end(
    codes: Sequence[int],
    codebook_atoms: Sequence[str],
) -> int:
    values = tuple(int(value) for value in codes)
    atoms = tuple(str(value) for value in codebook_atoms)
    if (
        len(values) < 3
        or values[0] not in {FRAME_A, FRAME_B}
        or values[1] not in {FRAME_A, FRAME_B}
        or not atoms
        or min(values) < 0
        or max(values) >= len(atoms)
    ):
        raise TokenNativeSyntaxRouterError(
            "cover-verified token-native transport differs"
        )
    prefix = values[0] == FRAME_A
    state = 1 if prefix else 0
    candidates: list[int] = []
    for body_index, code in enumerate(values[2:]):
        if 0 <= code < CALL_END:
            arity = code % CALL_STRIDE
        elif REIFY_BASE <= code < REIFY_END:
            arity = code - REIFY_BASE + 1
        else:
            arity = 0
        state += arity - 1 if prefix else 1 - arity
        if (prefix and state == 0 and values[2] in ROOT_CODES) or (
            not prefix and state == 1 and code in ROOT_CODES
        ):
            candidates.append(body_index + 3)
    matches = []
    for end in candidates:
        try:
            payload = "".join(atoms[code] for code in values[:end]).encode("ascii")
        except UnicodeEncodeError as exc:
            raise TokenNativeSyntaxRouterError(
                "token-native codebook atoms must be ASCII"
            ) from exc
        expected = _cover_indices(
            payload,
            width=len(values),
            count=len(values) - end,
            codebook_size=len(atoms),
        )
        if values[end:] == expected:
            matches.append(end)
    if len(matches) != 1:
        raise TokenNativeSyntaxRouterError(
            "token-native transport lacks one cover-verified document"
        )
    return matches[0]


class CoverVerifiedTokenNativeDocumentMask(TokenNativeDocumentMask):
    """Recover the exact AST boundary by verifying the public cover hash.

    The mask performs no ontology decoding. It synchronizes the small integer
    token matrix to CPU because SHA-256 is intentionally outside the learned
    graph, then returns a device-local boolean mask for the neural compiler.
    """

    def __init__(
        self,
        codebook_token_ids: Sequence[int],
        codebook_atoms: Sequence[str],
        *,
        vocab_size: int,
    ) -> None:
        super().__init__(codebook_token_ids, vocab_size=vocab_size)
        atoms = tuple(codebook_atoms)
        if (
            len(atoms) != len(tuple(codebook_token_ids))
            or len(set(atoms)) != len(atoms)
            or any(
                not isinstance(atom, str)
                or not atom
                or not atom.startswith(" ")
                or not atom.isascii()
                for atom in atoms
            )
        ):
            raise TokenNativeSyntaxRouterError(
                "cover-verified token-native codebook differs"
            )
        self.codebook_atoms = atoms

    def forward(
        self,
        tokens: torch.Tensor,
        transport_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tokens.ndim != 2
            or tokens.dtype != torch.long
            or transport_mask.shape != tokens.shape
            or transport_mask.dtype != torch.bool
            or tokens.shape[1] < 3
            or tokens.device != transport_mask.device
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native transport geometry differs"
            )
        codes = self.inverse_codebook[tokens]
        torch._assert_async(
            codes.ge(0).all(),
            "token-native transport leaves the bound codebook",
        )
        cpu_codes = codes.detach().to(device="cpu", non_blocking=False).tolist()
        ends = tuple(
            _cover_verified_document_end(row, self.codebook_atoms)
            for row in cpu_codes
        )
        terminal = torch.tensor(ends, dtype=torch.long, device=tokens.device)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        routed = transport_mask & positions[None, :].lt(terminal[:, None])
        torch._assert_async(
            routed.sum(dim=1).eq(terminal).all(),
            "cover-verified source mask truncates the syntax tree",
        )
        return routed


class TokenNativeOccurrenceEncoder(nn.Module):
    """Expose grammar roles and bind repeated local identifiers explicitly.

    Token-native symbol names are deliberately opaque and vary between
    documents.  Their usable invariant is equality between occurrences, not
    the accidental tokenizer embedding assigned to an ordinal.  This encoder
    factorizes public call heads/arities and integer atoms, then broadcasts a
    shared representation across equal identifier occurrences before a
    second contextual pass.  It performs no semantic execution.
    """

    _CATEGORY_COUNT = 5
    _HEAD_COUNT = MAX_HEAD + 2

    def __init__(
        self,
        codebook_token_ids: Sequence[int],
        *,
        vocab_size: int,
        width: int,
        num_heads: int,
        maximum_positions: int = 96,
        maximum_identifier_codes: int = 96,
    ) -> None:
        super().__init__()
        ids = tuple(int(value) for value in codebook_token_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not isinstance(vocab_size, int)
            or vocab_size < 1
            or min(ids) < 0
            or max(ids) >= vocab_size
            or len(ids) <= INTEGER_BASE + maximum_identifier_codes
            or not isinstance(width, int)
            or width < 64
            or not isinstance(num_heads, int)
            or num_heads < 1
            or width % num_heads
            or not isinstance(maximum_positions, int)
            or maximum_positions < 3
            or not isinstance(maximum_identifier_codes, int)
            or not 1 <= maximum_identifier_codes < maximum_positions + 1
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native occurrence geometry differs"
            )
        inverse = torch.full((vocab_size,), -1, dtype=torch.long)
        inverse[torch.tensor(ids, dtype=torch.long)] = torch.arange(
            len(ids),
            dtype=torch.long,
        )
        self.register_buffer("inverse_codebook", inverse)
        self.codebook_size = len(ids)
        self.width = width
        self.maximum_positions = maximum_positions
        self.maximum_identifier_codes = maximum_identifier_codes

        self.category_embedding = nn.Embedding(self._CATEGORY_COUNT, width)
        self.head_embedding = nn.Embedding(self._HEAD_COUNT, width)
        self.arity_embedding = nn.Embedding(MAX_NATIVE_ARITY + 2, width)
        self.integer_embedding = nn.Embedding(
            self.codebook_size - INTEGER_BASE,
            width,
        )
        self.renderer_embedding = nn.Embedding(4, width)
        self.position_embedding = nn.Embedding(maximum_positions, width)
        self.input_norm = nn.LayerNorm(width)
        self.pre_binding = self._layer(width, num_heads)
        self.binding_projection = nn.Linear(width, width, bias=False)
        self.post_binding = self._layer(width, num_heads)
        self.output_norm = nn.LayerNorm(width)

    @staticmethod
    def _layer(width: int, num_heads: int) -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )

    def forward(
        self,
        memory: torch.Tensor,
        tokens: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            memory.ndim != 3
            or memory.shape[:2] != tokens.shape
            or memory.shape[-1] != self.width
            or tokens.ndim != 2
            or tokens.dtype != torch.long
            or document_mask.shape != tokens.shape
            or document_mask.dtype != torch.bool
            or tokens.shape[1] > self.maximum_positions
            or memory.device != tokens.device
            or tokens.device != document_mask.device
        ):
            raise TokenNativeSyntaxRouterError("token-native occurrence input differs")
        codes = self.inverse_codebook[tokens]
        torch._assert_async(
            codes.ge(0).all(),
            "token-native occurrence leaves the bound codebook",
        )
        is_call = codes.lt(CALL_END)
        is_reified = codes.ge(REIFY_BASE) & codes.lt(REIFY_END)
        identifier_floor = self.codebook_size - self.maximum_identifier_codes
        is_identifier = codes.ge(identifier_floor) & document_mask
        is_integer = codes.ge(INTEGER_BASE) & ~is_identifier & document_mask
        is_frame = document_mask & ~(is_call | is_reified | is_integer | is_identifier)
        category = torch.zeros_like(codes)
        category = torch.where(is_call, torch.ones_like(category), category)
        category = torch.where(
            is_reified,
            torch.full_like(category, 2),
            category,
        )
        category = torch.where(
            is_integer,
            torch.full_like(category, 3),
            category,
        )
        category = torch.where(
            is_identifier,
            torch.full_like(category, 4),
            category,
        )
        torch._assert_async(
            (is_frame | is_call | is_reified | is_integer | is_identifier)[
                document_mask
            ].all(),
            "token-native occurrence category is incomplete",
        )

        head = torch.zeros_like(codes)
        head = torch.where(is_call, codes.div(CALL_STRIDE, rounding_mode="floor"), head)
        head = torch.where(
            is_reified,
            torch.full_like(head, MAX_HEAD + 1),
            head,
        )
        arity = torch.zeros_like(codes)
        arity = torch.where(is_call, codes.remainder(CALL_STRIDE), arity)
        arity = torch.where(is_reified, codes - REIFY_BASE + 1, arity)
        integer = (codes - INTEGER_BASE).clamp(
            min=0,
            max=self.codebook_size - INTEGER_BASE - 1,
        )
        renderer = codes[:, 0].eq(FRAME_B).to(torch.long) * 2 + codes[:, 1].eq(
            FRAME_B
        ).to(torch.long)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        structural = (
            self.category_embedding(category)
            + self.head_embedding(head)
            + self.arity_embedding(arity)
            + self.renderer_embedding(renderer)[:, None, :]
            + self.position_embedding(positions)[None, :, :]
        )
        structural = structural + (
            self.integer_embedding(integer) * is_integer.unsqueeze(-1)
        )
        padding = ~document_mask
        hidden = self.pre_binding(
            self.input_norm(memory + structural.to(memory.dtype)),
            src_key_padding_mask=padding,
        )

        equality = (
            codes[:, :, None].eq(codes[:, None, :])
            & is_identifier[:, :, None]
            & is_identifier[:, None, :]
        )
        counts = equality.sum(dim=-1, keepdim=True).clamp_min(1)
        shared = torch.bmm(
            equality.to(hidden.dtype),
            hidden,
        ) / counts.to(hidden.dtype)
        hidden = hidden + self.binding_projection(shared) * is_identifier.unsqueeze(-1)
        hidden = self.post_binding(hidden, src_key_padding_mask=padding)
        return self.output_norm(hidden)


class _SyntaxGraphLayer(nn.Module):
    """Exchange messages only across public syntax and equality edges."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.parent_projection = nn.Linear(width, width, bias=False)
        self.children_projection = nn.Linear(width, width, bias=False)
        self.occurrence_projection = nn.Linear(width, width, bias=False)
        self.message_gate = nn.Linear(3 * width, width)
        self.message_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        parent_index: torch.Tensor,
        has_parent: torch.Tensor,
        identifier_equality: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, width = hidden.shape
        gather_index = (
            parent_index.clamp_min(0)
            .unsqueeze(-1)
            .expand(
                batch,
                length,
                width,
            )
        )
        parent = hidden.gather(1, gather_index) * has_parent.unsqueeze(-1)

        children = torch.zeros_like(hidden)
        child_count = torch.zeros(
            batch,
            length,
            1,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        children.scatter_add_(1, gather_index, hidden * has_parent.unsqueeze(-1))
        child_count.scatter_add_(
            1,
            parent_index.clamp_min(0).unsqueeze(-1),
            has_parent.unsqueeze(-1).to(hidden.dtype),
        )
        children = children / child_count.clamp_min(1.0)

        occurrence_count = identifier_equality.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        occurrence = torch.bmm(
            identifier_equality.to(hidden.dtype),
            hidden,
        ) / occurrence_count.to(hidden.dtype)
        projected = torch.cat(
            (
                self.parent_projection(parent),
                self.children_projection(children),
                self.occurrence_projection(occurrence),
            ),
            dim=-1,
        )
        message = torch.sigmoid(self.message_gate(projected)) * (
            projected[..., :width]
            + projected[..., width : 2 * width]
            + projected[..., 2 * width :]
        )
        hidden = self.message_norm(hidden + message)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        return hidden * document_mask.unsqueeze(-1)


class TokenNativeSyntaxGraphEncoder(nn.Module):
    """Route contextual states over the exact public token-native AST.

    The graph is reconstructed from prefix/postfix arities.  It contains only
    parent/child edges, semantic child ranks, depth, and equality edges between
    opaque local identifiers.  It never decodes identifier names or executes
    the represented command.
    """

    _CATEGORY_COUNT = 5
    _HEAD_COUNT = MAX_HEAD + 2

    def __init__(
        self,
        codebook_token_ids: Sequence[int],
        *,
        vocab_size: int,
        width: int,
        layers: int = 3,
        maximum_positions: int = 96,
        maximum_identifier_codes: int = 96,
    ) -> None:
        super().__init__()
        ids = tuple(int(value) for value in codebook_token_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not isinstance(vocab_size, int)
            or vocab_size < 1
            or min(ids) < 0
            or max(ids) >= vocab_size
            or len(ids) <= INTEGER_BASE + maximum_identifier_codes
            or not isinstance(width, int)
            or width < 64
            or not isinstance(layers, int)
            or not 1 <= layers <= 8
            or not isinstance(maximum_positions, int)
            or maximum_positions < 3
            or not isinstance(maximum_identifier_codes, int)
            or not 1 <= maximum_identifier_codes < maximum_positions + 1
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native syntax graph geometry differs"
            )
        inverse = torch.full((vocab_size,), -1, dtype=torch.long)
        inverse[torch.tensor(ids, dtype=torch.long)] = torch.arange(
            len(ids),
            dtype=torch.long,
        )
        self.register_buffer("inverse_codebook", inverse)
        self.codebook_size = len(ids)
        self.width = width
        self.maximum_positions = maximum_positions
        self.maximum_identifier_codes = maximum_identifier_codes

        self.category_embedding = nn.Embedding(self._CATEGORY_COUNT, width)
        self.head_embedding = nn.Embedding(self._HEAD_COUNT, width)
        self.arity_embedding = nn.Embedding(MAX_NATIVE_ARITY + 2, width)
        self.integer_embedding = nn.Embedding(
            self.codebook_size - INTEGER_BASE,
            width,
        )
        self.renderer_embedding = nn.Embedding(4, width)
        self.position_embedding = nn.Embedding(maximum_positions, width)
        self.depth_embedding = nn.Embedding(maximum_positions, width)
        self.child_rank_embedding = nn.Embedding(MAX_NATIVE_ARITY + 1, width)
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(_SyntaxGraphLayer(width) for _ in range(layers))
        self.output_norm = nn.LayerNorm(width)

    @staticmethod
    def _syntax_links(
        codes: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact parent, semantic child rank, and tree depth."""

        batch, length = codes.shape
        positions = torch.arange(length, device=codes.device)
        nodes = document_mask & positions[None, :].ge(2)
        is_call = codes.lt(CALL_END)
        is_reified = codes.ge(REIFY_BASE) & codes.lt(REIFY_END)
        child_count = torch.zeros_like(codes)
        child_count = torch.where(
            is_call,
            codes.remainder(CALL_STRIDE),
            child_count,
        )
        child_count = torch.where(
            is_reified,
            codes - REIFY_BASE + 1,
            child_count,
        )
        child_count = child_count * nodes
        prefix = codes[:, 0].eq(FRAME_A)
        reverse_children = codes[:, 1].eq(FRAME_B)
        before = positions[:, None].lt(positions[None, :])
        span = ~before.T

        def segment_sums(effects: torch.Tensor) -> torch.Tensor:
            cumulative = effects.cumsum(dim=1)
            previous = torch.cat(
                (
                    torch.zeros(
                        batch,
                        1,
                        dtype=effects.dtype,
                        device=effects.device,
                    ),
                    cumulative[:, :-1],
                ),
                dim=1,
            )
            return cumulative[:, None, :] - previous[:, :, None]

        prefix_segments = segment_sums((child_count - 1) * nodes)
        prefix_completion = (
            nodes[:, :, None]
            & nodes[:, None, :]
            & span[None, :, :]
            & prefix_segments.eq(-1)
        )
        subtree_end = torch.where(
            prefix_completion,
            positions[None, None, :],
            torch.full(
                (1, 1, length),
                length,
                dtype=torch.long,
                device=codes.device,
            ),
        ).amin(dim=-1)
        torch._assert_async(
            (~nodes | ~prefix[:, None] | subtree_end.lt(length)).all(),
            "token-native prefix subtree does not terminate",
        )
        prefix_ancestors = (
            child_count[:, None, :].gt(0)
            & before.T[None, :, :]
            & subtree_end[:, None, :].ge(positions[None, :, None])
            & nodes[:, :, None]
            & nodes[:, None, :]
        )
        prefix_parent = torch.where(
            prefix_ancestors,
            positions[None, None, :],
            torch.full(
                (1, 1, length),
                -1,
                dtype=torch.long,
                device=codes.device,
            ),
        ).amax(dim=-1)

        postfix_segments = segment_sums((1 - child_count) * nodes)
        postfix_completion = (
            nodes[:, :, None]
            & nodes[:, None, :]
            & span[None, :, :]
            & postfix_segments.eq(1)
        )
        subtree_start = torch.where(
            postfix_completion,
            positions[None, :, None],
            torch.full(
                (1, length, 1),
                -1,
                dtype=torch.long,
                device=codes.device,
            ),
        ).amax(dim=1)
        torch._assert_async(
            (~nodes | prefix[:, None] | subtree_start.ge(2)).all(),
            "token-native postfix subtree does not start",
        )
        postfix_ancestors = (
            child_count[:, None, :].gt(0)
            & before[None, :, :]
            & subtree_start[:, None, :].le(positions[None, :, None])
            & nodes[:, :, None]
            & nodes[:, None, :]
        )
        postfix_parent = torch.where(
            postfix_ancestors,
            positions[None, None, :],
            torch.full(
                (1, 1, length),
                length,
                dtype=torch.long,
                device=codes.device,
            ),
        ).amin(dim=-1)
        postfix_parent = torch.where(
            postfix_parent.eq(length),
            torch.full_like(postfix_parent, -1),
            postfix_parent,
        )

        parent = torch.where(prefix[:, None], prefix_parent, postfix_parent)
        ancestor_count = torch.where(
            prefix[:, None, None],
            prefix_ancestors,
            postfix_ancestors,
        ).sum(dim=-1)

        root_count = (nodes & parent.lt(0)).sum(dim=1)
        torch._assert_async(
            root_count.eq(1).all(),
            "token-native syntax graph does not contain one root",
        )
        has_parent = nodes & parent.ge(0)
        same_parent = (
            parent[:, :, None].eq(parent[:, None, :])
            & has_parent[:, :, None]
            & has_parent[:, None, :]
        )
        observed_rank = (same_parent & before.T[None, :, :]).sum(dim=-1)
        total_children = child_count.gather(1, parent.clamp_min(0))
        child_rank = torch.where(
            reverse_children[:, None],
            total_children - 1 - observed_rank,
            observed_rank,
        ).clamp_min(0)
        child_rank = child_rank * has_parent
        depth = ancestor_count
        torch._assert_async(
            depth.lt(length).all(),
            "token-native syntax graph depth differs",
        )
        return parent, child_rank, depth

    def forward(
        self,
        memory: torch.Tensor,
        tokens: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            memory.ndim != 3
            or memory.shape[:2] != tokens.shape
            or memory.shape[-1] != self.width
            or tokens.ndim != 2
            or tokens.dtype != torch.long
            or document_mask.shape != tokens.shape
            or document_mask.dtype != torch.bool
            or tokens.shape[1] > self.maximum_positions
            or memory.device != tokens.device
            or tokens.device != document_mask.device
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native syntax graph input differs"
            )
        codes = self.inverse_codebook[tokens]
        torch._assert_async(
            codes.ge(0).all(),
            "token-native syntax graph leaves the bound codebook",
        )
        is_call = codes.lt(CALL_END)
        is_reified = codes.ge(REIFY_BASE) & codes.lt(REIFY_END)
        identifier_floor = self.codebook_size - self.maximum_identifier_codes
        is_identifier = codes.ge(identifier_floor) & document_mask
        is_integer = codes.ge(INTEGER_BASE) & ~is_identifier & document_mask
        is_frame = document_mask & ~(is_call | is_reified | is_integer | is_identifier)
        category = torch.zeros_like(codes)
        category = torch.where(is_call, torch.ones_like(category), category)
        category = torch.where(is_reified, torch.full_like(category, 2), category)
        category = torch.where(is_integer, torch.full_like(category, 3), category)
        category = torch.where(
            is_identifier,
            torch.full_like(category, 4),
            category,
        )
        torch._assert_async(
            (is_frame | is_call | is_reified | is_integer | is_identifier)[
                document_mask
            ].all(),
            "token-native syntax graph category is incomplete",
        )
        head = torch.zeros_like(codes)
        head = torch.where(
            is_call,
            codes.div(CALL_STRIDE, rounding_mode="floor"),
            head,
        )
        head = torch.where(
            is_reified,
            torch.full_like(head, MAX_HEAD + 1),
            head,
        )
        arity = torch.zeros_like(codes)
        arity = torch.where(is_call, codes.remainder(CALL_STRIDE), arity)
        arity = torch.where(is_reified, codes - REIFY_BASE + 1, arity)
        integer = (codes - INTEGER_BASE).clamp(
            min=0,
            max=self.codebook_size - INTEGER_BASE - 1,
        )
        renderer = codes[:, 0].eq(FRAME_B).to(torch.long) * 2 + codes[:, 1].eq(
            FRAME_B
        ).to(torch.long)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        parent, child_rank, depth = self._syntax_links(codes, document_mask)
        structural = (
            self.category_embedding(category)
            + self.head_embedding(head)
            + self.arity_embedding(arity)
            + self.renderer_embedding(renderer)[:, None, :]
            + self.position_embedding(positions)[None, :, :]
            + self.depth_embedding(depth)
            + self.child_rank_embedding(child_rank)
        )
        structural = structural + (
            self.integer_embedding(integer) * is_integer.unsqueeze(-1)
        )
        hidden = self.input_norm(memory + structural.to(memory.dtype))
        equality = (
            codes[:, :, None].eq(codes[:, None, :])
            & is_identifier[:, :, None]
            & is_identifier[:, None, :]
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                parent_index=parent,
                has_parent=document_mask & parent.ge(0),
                identifier_equality=equality,
                document_mask=document_mask,
            )
        return self.output_norm(hidden)


__all__ = [
    "CoverVerifiedTokenNativeDocumentMask",
    "TokenNativeDocumentMask",
    "TokenNativeOccurrenceEncoder",
    "TokenNativeSyntaxGraphEncoder",
    "TokenNativeSyntaxRouterError",
]
