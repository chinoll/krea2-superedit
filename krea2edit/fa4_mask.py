"""CuTe DSL mask for FA4; imported only by the supported CUDA backend."""

import cutlass
import cutlass.cute as cute


@cute.jit
def prefix_mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    # FA4 supplies sample-local positions. Aux data uses packed global offsets.
    # Clamp tile-tail queries before reading metadata; FA4 masks padded lanes.
    query = cute.make_rmem_tensor(1, cutlass.Int32)
    query.store(q_idx)
    key_start = cute.make_rmem_tensor(1, cutlass.Int32)
    key_end = cute.make_rmem_tensor(1, cutlass.Int32)
    packed_query = seqlen_info.offset_q + cutlass.min(
        query[0], seqlen_info.seqlen_q - 1
    )
    key_start[0] = aux_tensors[0][packed_query]
    key_end[0] = aux_tensors[1][packed_query]
    return (kv_idx >= key_start.load()) & (kv_idx < key_end.load())
