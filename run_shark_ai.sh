# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env bash

set -o pipefail
shopt -s extglob

# The script accepts a number of command line arguments from the user.
#   1. -h -- prints the help string [same as no flag given at all].
#   2. -o -- specifies the export directory, where the MLIR and the VMFB are placed.
#   3. -w -- specifies the location of the weight file.
#   4. -a -- specifies where to place the artifacts (such as sharded weights).
#            if this flag is unset, the artifact dir defaults to <export_dir>/artifacts/.
#   5. -b -- specifies the batch sizes for the export.
#   6. -i -- path to the IREE build dir (for dev purposes).
#   7. -v -- enable verbose logging.
#   8. -s -- Whether to shard the weights or not. Defaults to tp1 if not set, else tp8.
#   9. -x -- If set, compilation is skipped. Useful when only sharding or export is required.
#  10. -e -- If set, export is called. This allows sharding to be run separately from export.


print_help_string() {
  printf "%s\n" "USAGE: run_shark_ai [OPTIONS] ... [EXPORT_DIR]
Exports the model using shark-ai and compiles the IR to produce the VMFB module.

Some options must be madatorily specified.
    -h            prints the help string. The same output is emitted
                    when no flags are specified.
    -v            enables verbose logging. When not specified, only errors are logged.
    -w            the location of the GGUF/IRPA file(s) that contain the parameters. (required)
    -a            the location of the artifacts (sharded weights). (defaults to EXPORT_DIR/artifacts)
    -b            batch sizes for the export. Multiple batch sizes should be separated by ','.
                    defaults to (1,4).
    -i            location of the IREE build directory, to use local build for compilation.
                    defaults to the PATH value if not set.
    -s            use to enable sharding. shards to tp8 if set, otherwise tp1.
    -x            whether to skip compilation and only export the model.
    -e            calls export first if set. This allows the export and sharding commands to be
                    run independently.

EXPORT_DIR is a positional argument, to be placed at the end. Specifies where the exported model will be placed."
}

VERBOSE_LOGGING=false
WEIGHT_FILE_LOC=0
ARTIFACT_LOC=0
EXPORT_BATCH_SIZES=1,4
EXPORT=0
ROCR_VISIBLE_DEVICES=1
MODEL_CONFIG_LOC=0
IREE_BUILD_DIR=0
SHARD_WEIGHTS=0
SKIP_COMPILE=0
EXPORT_DIR=0

err() { printf "%s\n" "$*" >&2 && exit 1; }
log() { $VERBOSE_LOGGING && printf "%s\n" "$*" >&2; }

unset h
unset i
unset s
unset v
unset x
unset w
unset t
unset a
unset b
unset e

parse_and_handle_args() {
  while getopts ":ehvsxw:t:a:b:i:" flag; do
    case "$flag" in
      h)
        print_help_string && exit 0
        ;;
      v)
        VERBOSE_LOGGING=true
        ;;
      w)
        WEIGHT_FILE_LOC=$OPTARG
        ;;
      a)
        ARTIFACT_LOC=$OPTARG
        if [[ ! -d $ARTIFACT_LOC ]]; then
            log "[WARNING] $ARTIFACT_LOC does not exist, creating..."
            mkdir -p $ARTIFACT_LOC
        fi
        ;;
      b)
        EXPORT_BATCH_SIZES=$OPTARG
        ;;
      i)
        IREE_BUILD_DIR=$OPTARG
        ;;
      s)
        SHARD_WEIGHTS=1
        log "[INFO] Weights will be sharded with tp 8"
        ;;
      x)
        SKIP_COMPILE=1
        log "[INFO] Compilation will be skipped"
        ;;
      e)
        EXPORT=1
        log "[INFO] Export is enabled"
	;;
      :)
        err "[ERROR] option -$OPTARG expected an argument"
        ;;
      \?)   print_help_string && err "[ERROR] INVALID OPTION -$OPTARG"
        ;;
    esac
  done
  shift $(( $OPTIND-1 ))

  [[ $# -eq 1 ]] || err "[ERROR] expected EXPORT_DIR as exactly one positional argument"
  EXPORT_DIR=$1
}

check_valid() {
  if [[ ! -d $EXPORT_DIR ]]; then
    log "[WARNING] $EXPORT_DIR does not exist, creating..."
    mkdir -p $EXPORT_DIR
  fi

  MODEL_CONFIG_LOC=$EXPORT_DIR/config.json

  [[ -f $WEIGHT_FILE_LOC ]] || (err "[ERROR] Missing argument -w or invalid path to weight file" && exit 1)

  # When we do not want to skip compile.
  if [[ $IREE_BUILD_DIR -eq 0 ]] && [[ $SKIP_COMPILE -eq 0 ]]; then
    compiler_loc=$(which iree-compile)
    [[ ! -z $compiler_loc ]] || err "[ERROR] iree-compile not in PATH, and IREE_BUILD_DIR was not provided through '-i'"
    IREE_BUILD_DIR=$(dirname $compiler_loc)
    log "[WARNING] Invalid path to IREE build dir. Defaulting to installed python package"
    log "[INFO] Using iree-compile from $IREE_BUILD_DIR/tools/"
  fi

  log "[INFO] Using weight file from $WEIGHT_FILE_LOC"
}

sharktank_export() {
  log "[INFO] Starting Export..."
  MLIR_PATH=$EXPORT_DIR/model.mlir
  log "[INFO] Model MLIR will be saved at $MLIR_PATH"
  log "[INFO] Generated config.json will be placed in $MODEL_CONFIG_LOC"

  python3 -m sharktank.examples.export_paged_llm_v1 --gguf-file=$WEIGHT_FILE_LOC --output-mlir=$MLIR_PATH --output-config=$MODEL_CONFIG_LOC --bs=$EXPORT_BATCH_SIZES 2>&1
  if [[ $? -ne 0 ]]; then
    err "[ERROR] Failed to run export"
  fi

  log "[INFO] Successfully exported model to $EXPORT_DIR"

}

shard() {
    log "[INFO] Sharding enabled. Sharding weights and saving to $ARTIFACT_LOC"

    if [[ $ARTIFACT_LOC -eq 0 ]]; then
      ARTIFACT_LOC=$EXPORT_DIR/artifacts
      mkdir -p $ARTIFACT_LOC
      log "[WARNING] Invalid or missing path to artifacts dir; Defaulting to $EXPORT_DIR/artifacts"
    fi

    log "[INFO] Setting artifacts dir to $ARTIFACT_LOC"

    weight_file_stem=$(echo $WEIGHT_FILE_LOC | rev | cut -d / -f 1 | cut --complement -d . -f 1 | rev)
    shard_file_suffix="_tp8.irpa"
    SHARD_OUTPUT_FILE="$ARTIFACT_LOC/$weight_file_stem$shard_file_suffix"
    python3 -m sharktank.examples.sharding.shard_llm_dataset \
      --irpa-file $WEIGHT_FILE_LOC \
      --output-irpa $SHARD_OUTPUT_FILE \
      --tensor-parallelism-size 8

    if [[ $? -ne 0 ]]; then
      err "[ERROR] Failed to run export"
    fi

    log "[INFO] Succesfully sharded weights; unranked shard file located at $SHARD_OUTPUT_FILE"
}

main() {
  parse_and_handle_args "$@"
  check_valid

  if [[ $EXPORT -eq 0 ]] && [[ $SHARD_WEIGHTS -eq 0 ]]; then
    err "[ERROR] At least one of export (-e) or shard (-s) must be specified"
  fi

  if [[ $EXPORT  -eq 1 ]]; then
    sharktank_export
  fi

  if [[ $SHARD_WEIGHTS -eq 1 ]]; then
    shard
  fi

}

main "$@"
