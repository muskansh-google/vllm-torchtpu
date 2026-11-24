# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``repetition_penalty`` for the TPU sampler.

The TPU runner samples on device without running vLLM's GPU ``Sampler``
penalties pass. vLLM's ``InputBatch`` already keeps the penalty of each row
(``repetition_penalties_cpu``), the requests that have one
(``repetition_penalties_reqs``) and their output ids
(``req_output_token_ids``), and moves them with the rows when the batch is
condensed or reordered. This module reads them from there and keeps only a
seen-token bitmask per request id (``RepetitionPenaltyTracker``).

For each penalized request, ``SeenTokenBitmask`` packs presence bits over
vocabulary ids ``[0, vocab_size)`` from ``prompt_token_ids`` and the output ids
into ``cdiv(vocab_size, 32)`` ``int32`` words (out-of-range ids such as
multimodal placeholders are ignored). On device
(``layers/adapter/sample/repetition_penalty.py``), the bitmask is unpacked and
seen positions are scaled in ``float32`` by ``1 / penalty`` where
``logits > 0`` and ``penalty`` otherwise before casting back to
``logits.dtype``.

Difference from vLLM: vLLM casts the logits to ``float32`` before the penalty
and picks the token from ``float32``. Here the scaled logits go back to
``logits.dtype`` (usually ``bfloat16``), the dtype the TPU sampler and its
precompiled graphs take, so they are rounded a second time. Two logits closer
than one ``bfloat16`` step (0.125 near 20) can then tie or swap, and the
greedy token can differ from vLLM's.

A speculative verify chunk has one logits row for each draft position of a
request and one bonus row. As in vLLM (``_combine_outputs_with_spec_tokens`` of
``RejectionSampler`` and ``Sampler``), the row at draft position ``p`` sees the
request's tokens plus ``drafts[:p]``, and the bonus row sees all the drafts.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.layers.adapter.sample import repetition_penalty as penalty_op
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import min_tokens

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)

_BIT_MASKS: np.ndarray = (np.uint32(1) << np.arange(32, dtype=np.uint32)).view(np.int32)


class SeenTokenBitmask:
    """Packed presence bits of one request's prompt and output ids, updated
    incrementally as the output grows."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self._bitmask = np.zeros((cdiv(vocab_size, 32),), dtype=np.int32)
        self._prompt_ids: Sequence[int] | None = None
        self._output_ids: Sequence[int] | None = None
        self._committed_len = 0
        self._built = False

    def update(
        self, prompt_ids: Sequence[int] | None, output_ids: Sequence[int]
    ) -> np.ndarray:
        """Return the ``int32`` ``[cdiv(vocab_size, 32)]`` presence bitmask over
        ``prompt_ids`` and ``output_ids``.

        The bitmask is built again when the request's prompt or output list is
        a different object (a streaming-input continuation, or a finished id
        sent again) or the output became shorter.
        """
        vocab_size = self._vocab_size
        cur_len = len(output_ids)
        if (
            not self._built
            or prompt_ids is not self._prompt_ids
            or output_ids is not self._output_ids
            or cur_len < self._committed_len
        ):
            self._bitmask.fill(0)
            if prompt_ids is not None and len(prompt_ids) > 0:
                prompt = np.asarray(prompt_ids, dtype=np.int64)
                valid = prompt[(prompt >= 0) & (prompt < vocab_size)]
                np.bitwise_or.at(self._bitmask, valid >> 5, _BIT_MASKS[valid & 31])
            self._built = True
            self._prompt_ids = prompt_ids
            self._output_ids = output_ids
            self._committed_len = 0

        if cur_len > self._committed_len:
            bitmask = self._bitmask
            for tok in output_ids[self._committed_len : cur_len]:
                if 0 <= tok < vocab_size:
                    bitmask[tok >> 5] |= _BIT_MASKS[tok & 31]
            self._committed_len = cur_len

        return self._bitmask


class RepetitionPenaltyTracker:
    """Seen-token bitmasks, by request id, of the requests that
    ``InputBatch.repetition_penalties_reqs`` lists, and the host staging
    buffers for one logits tensor. The buffers start at ``max_num_reqs`` rows
    and grow to the largest verify bucket when ``warmup`` stages it."""

    def __init__(
        self,
        vocab_size: int,
        max_num_reqs: int,
        *,
        async_scheduling: bool,
    ) -> None:
        self.vocab_size = vocab_size
        # TODO(#2861): apply under async_scheduling. There the previous step's
        # sampled token is still on the device when the bitmask is built.
        self.supported = not async_scheduling
        self._seen: dict[str, SeenTokenBitmask] = {}
        self._bitmask_buf = np.zeros(
            (max_num_reqs, cdiv(vocab_size, 32)), dtype=np.int32
        )
        self._penalties_buf = np.ones((max_num_reqs, 1), dtype=np.float32)
        self._dirty_rows: list[int] = []

    def active(self, input_batch: InputBatch) -> bool:
        """Whether a row of ``input_batch`` gets the penalty in this mode."""
        return self.supported and bool(input_batch.repetition_penalties_reqs)

    def begin_step(self, input_batch: InputBatch) -> bool:
        """``active(input_batch)``, warning once when a penalized request runs
        in an unsupported mode, and dropping the bitmasks of requests that are
        no longer penalized."""
        penalized = input_batch.repetition_penalties_reqs
        if not penalized:
            self._seen.clear()
            return False
        if not self.supported:
            logger.warning_once(
                "repetition_penalty is not yet applied on TPU with "
                "async_scheduling; requests run without it. Run with "
                "async_scheduling=False to apply it."
            )
            return False
        self._prune(penalized)
        return True

    def apply(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Scale the plain 1-row/req logits chunk of batch rows ``[start, end)``."""
        inputs = self.build_plain_inputs(
            input_batch, requests, start, end, logits.shape[0]
        )
        if inputs is None:
            return logits
        return apply_penalties_table(logits, *inputs)

    def apply_verify_logits(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        start: int,
        drafts: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Scale a verify chunk's logits before the target / bonus gathers.

        Request ``start + k`` has drafts ``drafts[k]`` and owns
        ``len(drafts[k]) + 1`` consecutive rows: its draft positions, then its
        bonus.
        """
        rows = min_tokens.verify_logits_rows(
            self._penalized_indices(input_batch), start, [len(d) for d in drafts]
        )
        return self._apply_rows(logits, input_batch, requests, rows, start, drafts)

    def apply_verify_split(
        self,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        start: int,
        drafts: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale a verify chunk's target and bonus logits after the gathers.

        Request ``start + k`` owns ``len(drafts[k])`` consecutive target rows
        and bonus row ``k``.
        """
        target_rows, bonus_rows = min_tokens.verify_rows(
            self._penalized_indices(input_batch), start, [len(d) for d in drafts]
        )
        # The staging buffers are free again when a call returns: the upload
        # in apply_penalties_table is blocking.
        target_logits = self._apply_rows(
            target_logits, input_batch, requests, target_rows, start, drafts
        )
        bonus_logits = self._apply_rows(
            bonus_logits, input_batch, requests, bonus_rows, start, drafts
        )
        return target_logits, bonus_logits

    def warmup(self, logits: torch.Tensor) -> torch.Tensor:
        """Run the kernel on ``logits`` with the staging buffers, as ``apply``
        and the verify calls do for a tensor of ``logits.shape[0]`` rows. The
        runner's precompile calls this, so warmup and serving stage their
        inputs in one way."""
        return apply_penalties_table(logits, *self._buffers(logits.shape[0]))

    def _prune(self, req_ids: Collection[str]) -> None:
        """Drop the bitmasks of requests that are not in ``req_ids``."""
        if len(self._seen) > len(req_ids):
            for req_id in [r for r in self._seen if r not in req_ids]:
                del self._seen[req_id]

    def _penalized_indices(self, input_batch: InputBatch) -> list[int]:
        """Batch indices of the requests that have a penalty."""
        req_id_to_index = input_batch.req_id_to_index
        return [
            req_id_to_index[req_id] for req_id in input_batch.repetition_penalties_reqs
        ]

    def _buffers(self, num_rows: int) -> tuple[np.ndarray, np.ndarray]:
        """The first ``num_rows`` rows of the staging buffers."""
        if num_rows > self._bitmask_buf.shape[0]:
            self._bitmask_buf = np.zeros(
                (num_rows, self._bitmask_buf.shape[1]), dtype=np.int32
            )
            self._penalties_buf = np.ones((num_rows, 1), dtype=np.float32)
            self._dirty_rows.clear()
        return self._bitmask_buf[:num_rows], self._penalties_buf[:num_rows]

    def build_plain_inputs(
        self,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        start: int,
        end: int,
        num_rows: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stage ``(seen_bitmask, penalties)`` for the plain 1-row/req logits
        chunk of batch rows ``[start, end)`` into reusable host buffers, or
        return ``None`` when no row in the chunk has a penalty."""
        rows = min_tokens.plain_rows(self._penalized_indices(input_batch), start, end)
        return self.build_inputs(input_batch, requests, rows, num_rows)

    def build_inputs(
        self,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        rows: min_tokens.StateRows,
        num_rows: int,
        start: int = 0,
        drafts: Sequence[Sequence[int]] = (),
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stage ``(seen_bitmask, penalties)`` for a logits tensor of
        ``num_rows`` rows into reusable host buffers, or return ``None`` when
        ``rows`` is empty.

        ``rows`` maps the batch index of a penalized request to ``(first row,
        drafts before that row, number of rows)``, as ``min_tokens.StateRows``.
        Row ``first + j`` sees the request's prompt and output ids plus the
        first ``ahead + j`` ids of ``drafts[batch index - start]``.
        """
        if not rows:
            return None
        bitmask_buf, penalties_buf = self._buffers(num_rows)
        for prev_row in self._dirty_rows:
            self._bitmask_buf[prev_row].fill(0)
            self._penalties_buf[prev_row, 0] = 1.0
        self._dirty_rows.clear()

        vocab_size = self.vocab_size
        for batch_idx, (first, ahead, count) in rows.items():
            req_id = input_batch.req_ids[batch_idx]
            output_ids = input_batch.req_output_token_ids[batch_idx]
            assert req_id is not None and output_ids is not None
            seen = self._seen.get(req_id)
            if seen is None:
                seen = self._seen[req_id] = SeenTokenBitmask(vocab_size)
            base = seen.update(requests[req_id].prompt_token_ids, output_ids)
            penalty = input_batch.repetition_penalties_cpu[batch_idx]
            for j in range(count):
                row = first + j
                bitmask = bitmask_buf[row]
                bitmask[:] = base
                for tok in drafts[batch_idx - start][: ahead + j] if drafts else ():
                    if 0 <= tok < vocab_size:
                        bitmask[tok >> 5] |= _BIT_MASKS[tok & 31]
                penalties_buf[row, 0] = penalty
                self._dirty_rows.append(row)
        return bitmask_buf, penalties_buf

    def _apply_rows(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        rows: min_tokens.StateRows,
        start: int,
        drafts: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        inputs = self.build_inputs(
            input_batch, requests, rows, logits.shape[0], start, drafts
        )
        if inputs is None:
            return logits
        return apply_penalties_table(logits, *inputs)


def apply_penalties_table(
    logits: torch.Tensor,
    seen_bitmask_np: np.ndarray,
    penalties_np: np.ndarray,
) -> torch.Tensor:
    """Stage host packed bitmask and penalty tables to ``logits.device`` and apply
    scaling."""
    device = logits.device
    seen_bitmask = torch.from_numpy(seen_bitmask_np).to(device)
    penalties = torch.from_numpy(penalties_np).to(device)
    return penalty_op.apply_repetition_penalty(logits, seen_bitmask, penalties)
