# src/uni20/linalg/backends/cusolver

This directory implements the blocking exact SVD operation-tag adapter in
`svd.hpp`, using cuSOLVER `gesvd` on CUDA-buffer descriptors. The current slice
supports real `float` and `double`, nonempty matrices with rows >= columns,
column-major provider-compatible storage, and separate factor outputs.
Unsupported shapes, layouts, or overwrite modes are declined before execution.

The same file implements values-only `singular_values_op` for real and complex
`float` and `double`, through `gesvd` with `jobu = jobvt = 'N'` and null U/VT.
It accepts tall column-major workspace or wide row-major workspace read as its
tall transpose, and returns immediately for empty matrices.

Both operations select the operand device around the provider calls, since
cuSOLVER launches into the calling thread's current device.

Execution acquires buffer access on the cuSOLVER stream and synchronizes before
returning, including checking the device convergence status. This is a blocking
adapter, not a coroutine-native provider task. See the
[cuSOLVER architecture](../../../../../docs/backends/cuda/cusolver.md) for the
execution and ownership model.

## Notes

- Gate operations through `kernel_accepts_types` and `try_kernel` before
  committing to a cuSOLVER path.
- Keep CUDA library ABI details in the
  [cuSOLVER provider layer](../../../backend/cusolver/).

## Related Documentation

- [Linalg backend source map](../)
- [CUDA kernel dispatch and provider scheduling](../../../../../docs/backends/cuda/kernel_dispatch.md)
- [cuSOLVER architecture](../../../../../docs/backends/cuda/cusolver.md)
- [CUDA runtime resolution](../../../../../docs/backends/cuda/runtime_resolution.md)
