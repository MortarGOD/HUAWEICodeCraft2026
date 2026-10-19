# Legal Tensor Reconstruction V2

## Information-flow invariant

- Linear `activation_state` contains only a channel balance computed from the
  calibration activations. The helper that creates it does not accept `W`.
- W consumes the already-computed A-only balance but cannot alter the state.
- Linear A/W use reciprocal A-only scales and the same code-fixed orthogonal
  transform.
- Attention Q/K states contain only calibration Q/K statistics. Q/K use
  reciprocal scales and the same code-fixed orthogonal transform; V is
  quantized independently.
- Every tensor's HiF4 parameters are selected only from that tensor's own
  reconstruction error inside the interface that receives it.
- Runtime audit: negating every W and multiplying every W source scale by
  `1.125` left `activation_state` bit-identical across all 82 Linear groups.

Local scores recorded below are surrogate results, not online scores.
