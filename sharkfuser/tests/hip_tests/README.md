# `hip_tests`

Fusilli will soon be integrated into `hipDNN` and potentially other projects
whose interfaces require `fusilli`-generated kernels to execute using
`hipMalloc`'ed buffers.

From iree-runtime's perspective, these buffers are "externally managed." The
runtime normally expects `iree_hal_buffer_view` types allocated with its own
APIs. However, iree-runtime provides an `import/export_buffer` API that can
create `iree_hal_buffer_view` types from externally managed buffers.

Since `fusilli` only works with `iree_hal_buffer_view`s and the `hipDNN`
integration doesn't exist yet, this directory serves as a temporary test bed for
the `hipMalloc`'ed `void*` -> `iree_hal_buffer_view` (via `import_buffer`) use
case.
