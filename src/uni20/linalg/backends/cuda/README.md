# CUDA Reference Linalg Backend

This directory contains generic CUDA runtime implementations of linalg
operation tags. `CudaReferenceBackend` follows provider backends such as cuBLAS
in `CudaStorage`'s ordered selector and handles operations that do not require a
provider library.

The `copy_op` implementation keeps runtime-copy fast paths for compatible raw
contiguous mappings. Other same-device positive-strided mappings, including
padding and differing physical order, use the reference backend's elementwise
CUDA kernel. An identity copy moves each element as unsigned words of its
alignment, so any trivially copyable scalar of up to four words is supported;
conjugating copies use typed `uni20::cfloat` and `uni20::cdouble` kernels.

The reference copy declines host-device copies whose operands do not share one
contiguous element order, and strided copies between devices. Two staging
backends in `copy_staging.hpp` follow it and complete those copies by
dispatching each step through a selector they hold:

- `CudaHostStagingBackend` moves one contiguous range: the host operand's order
  when it is contiguous, otherwise the device operand's, otherwise column-major.
  The `HostStorage` copy selector packs or unpacks the host side and the
  `CudaStorage` copy selector changes the device side's layout.
  `cuda_host_transfer_selector()` places it after `CudaReferenceBackend` for
  every copy between host and CUDA storage.
- `CudaPeerStagingBackend` peer-copies the source's storage range into a
  staging buffer on the destination device, then copies from that buffer there.
  `backend_selector_default<copy_op, CudaStorage>` appends it to the storage
  selector, which it retains for its steps.

`staging_backends.hpp` declares both backend values without the copy
implementation, so `cuda_storage.hpp` can name the peer stager in its copy
default.

`elementwise_plan.hpp` lowers the same backend-neutral affine plan used by the
CPU executor to a device POD, preferring 32-bit logical indices and offsets when
every reachable offset fits and using a 64-bit payload otherwise. One launch
covers at most eight compact dimensions; outer dimensions beyond eight become
one launch per outer index. The compiled executor is arity-generic and
separates overwrite operations, which do not read the output, from update
operations, which do.

Copy is the first one-input overwrite instantiation. `conjugate_inplace_op` is
the first zero-input update instantiation: it reads and writes each output
element through one exclusive CUDA access. Real and integer conjugation succeeds
without acquiring a stream or launching because it is semantically the identity.
Complex conjugation supports persistent `uni20::cfloat` and `uni20::cdouble`
storage through CUDA execution proxies.

The CUDA reference backend explicitly registers stateless unary `negate`,
`square`, and `reciprocal` transforms; stateful `scale<Factor>`; and stateless
binary `add`, `subtract`, `multiply`, and `divide` transforms. Binary operations
carry two independently strided input operands. Supported real and complex
combinations are compiled into the library. Other callable types are not
assumed to be available inside the precompiled library and therefore remain
ineligible for this backend.

The kernels publish stream completion through the same buffer ledgers as the
runtime copy path. Nonpositive strides on active axes cleanly decline for
nontrivial elementwise work; supporting a future negative-stride Uni20 layout
requires descriptor rebasing and signed traversal rather than raw runtime copying.
Distinct-offset views into one CUDA buffer use a single exclusive access and
rely on the C++ copy precondition that the operands do not destructively
overlap. Nontrivial same-offset transformations decline.

## Related Documentation

- [Linalg backend source map](../)
- [CUDA backend guide](../../../../../docs/backends/cuda/)
- [Tensor scalar and transfer design](../../../../../docs/tensor/scalar_tensors_and_transfer.md)
