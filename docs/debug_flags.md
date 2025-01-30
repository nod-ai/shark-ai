# Debug flags

Various debug flags are managed via environment variables.

## Modeling flags

Model level flags (used for running or compiling a model) are defined via the
`TURBINE_LLM_DEBUG` environment variable. This is a comma delimeted string of
`name[=value]` pairs. For boolean options the `=value` is not included. Instead,
presence of the name sets the flag to true. Prefixing it with `-` sets it to
false.

### Flags:

* `enable_tensor_trace`: Boolean flag that enables the trace_tensor facility
  to emit information. When disabled, it is a no-op.
* `enable_nan_checks`: Enables certain expensive nan checks that may be
  included in the model.
* `trace_path`: When set to a path, any tensor traced via
  `trace_tensor()` will be added to a safetensors file and output
  in a deterministic way to the path.
* `use_custom_int_conv_kernel`: Uses custom kernels for integer convolution
  arithmetic. This produces the most optimal compiled results but can impede
  debugging and interactive use. Defaults to True.
* `use_custom_int_mm_kernel`: Uses custom kernels for integer matmul
  arithmetic. This produces the most optimal compiled results but can impede
  debugging and interactive use. Defaults to True.
