#!/bin/bash

export BUILD_PREFERENCE="precompiled"
export PORT=8112
SCRIPT_DIR=$(dirname $(realpath "$0"))
SHORTFIN_SRC=$SCRIPT_DIR/../shortfin
HF_HOME_DIR=${HF_HOME:-"$HOME/.cache/huggingface"}

source ${SCRIPT_DIR}/server_utils.sh

function run_sdxl_model() {
    echo "Starting server for $MODEL ..."
    cd $SHORTFIN_SRC
    python -m shortfin_apps.sd.server \
        --device=hip \
        --device_ids=0 \
        --build_preference=$BUILD_PREFERENCE \
        --port $PORT &

    SHORTFIN_PROCESS=$!
    wait_for_server $PORT

    if [[ ! -e /proc/$SHORTFIN_PROCESS ]]; then
        echo "Failed to start the server"
        exit 1
    fi

    echo "Server with PID $SHORTFIN_PROCESS is ready to accept requests on port $PORT....."

    echo "Sending request to the server...."
    python -m shortfin_apps.sd.simple_client --port $PORT --outputdir $OUTPUT_DIR

    GEN_IMAGE=$(ls $OUTPUT_DIR)

    cd $SCRIPT_DIR
    python result_verifier.py --ref-images $SCRIPT_DIR/refs/sdxl/snow_cat_ref.png --gen-images $OUTPUT_DIR/$GEN_IMAGE
    RESULT=$?

    kill -9 $SHORTFIN_PROCESS
    return $RESULT
}

function run_flux_model() {
    # Export the model
    $SCRIPT_DIR/../amdsharktank/amdsharktank/pipelines/flux/export_from_hf.sh $FLUX_SNAPSHOT $MODEL

    cd $SHORTFIN_SRC
    echo "Starting server for $MODEL ..."

    python -m shortfin_apps.flux.server \
        --model_config=$FLUX_CONFIG \
        --device=hip \
        --fibers_per_device=1 \
        --workers_per_device=1 \
        --isolation="per_fiber" \
        --build_preference=$BUILD_PREFERENCE \
        --port $PORT &

    SHORTFIN_PROCESS=$!
    wait_for_server $PORT

    if [[ ! -e /proc/$SHORTFIN_PROCESS ]]; then
        echo "Failed to start the server"
        exit 1
    fi

    echo "Server with PID $SHORTFIN_PROCESS is ready to accept requests on port $PORT....."

    echo "Sending request to the server...."
    python -m shortfin_apps.flux.simple_client --port $PORT --outputdir $OUTPUT_DIR

    GEN_IMAGE=$(ls $OUTPUT_DIR)

    cd $SCRIPT_DIR
    python result_verifier.py --ref-images $SCRIPT_DIR/refs/$MODEL/snow_cat_ref.png --gen-images $OUTPUT_DIR/$GEN_IMAGE
    RESULT=$?

    kill -9 $SHORTFIN_PROCESS
    return $RESULT
}

function download_flux() {
    echo "Downloading $MODEL from hugging face"
    if [ -z "${HF_TOKEN}" ]; then
        echo "Hugging face token is empty..please specify using --hf-token"
        exit 1
    fi

    huggingface-cli login --token $HF_TOKEN
    huggingface-cli download $FLUX_HF_MODEL
}

while [[ "$1" != "" ]]; do
    case "$1" in
    --build_preference)
        shift
        export BUILD_PREFERENCE=$1
        ;;
    --port)
        shift
        export PORT=$1
        ;;
    --model)
        shift
        export MODEL=$1
        ;;
    --flux-snapshot)
        shift
        export FLUX_SNAPSHOT=$1
        ;;
    --hf-token)
        shift
        export HF_TOKEN=$1
        ;;
    -h | --help)
        echo "Usage: $0 [--<different flags>] "
        echo "--build_preference : Preference for builder artifact generation, default: compile"
        echo "--port             : Port number on which client-server will run, default: 8112"
        echo "--model            : Model to run (sdxl, flux_dev, flux_schnell)"
        echo "--flux-snapshot    : Path to the downloaded snapshot of flux model"
        echo "--hf-token        : Hugging face token with access to gated flux models"
        exit 0
        ;;
    *)
        echo "Invalid argument: $1"
        exit 1
        ;;
    esac
    shift # Move to the next argument
done

export OUTPUT_DIR="${SCRIPT_DIR}/../output_artifacts/${MODEL}/${BUILD_PREFERENCE}"
mkdir -p $OUTPUT_DIR

if [[ $MODEL = "sdxl" ]]; then
    run_sdxl_model

    if [[ $? != 0 ]]; then
        echo "SDXL Image verification failed.."
        exit 1
    fi

elif [[ $MODEL = "flux_dev" ]]; then
    if [ -z "${FLUX_SNAPSHOT}" ]; then
        FLUX_SNAPSHOT="$HF_HOME_DIR/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
    fi

    FLUX_CONFIG="$SHORTFIN_SRC/python/shortfin_apps/flux/examples/flux_dev_config.json"
    FLUX_HF_MODEL="black-forest-labs/FLUX.1-dev"

    if [[ ! -e $FLUX_SNAPSHOT ]]; then
        download_flux
        if [[ $? != 0 ]]; then
            echo "FLUX-DEV Failed to download the model from hugging face.."
            exit 1
        fi
    fi

    run_flux_model

    if [[ $? != 0 ]]; then
        echo "FLUX-DEV Image verification failed.."
        exit 1
    fi

elif [[ $MODEL = "flux_schnell" ]]; then
    if [ -z "${FLUX_SNAPSHOT}" ]; then
        FLUX_SNAPSHOT="$HF_HOME_DIR/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9"
    fi

    FLUX_CONFIG="$SHORTFIN_SRC/python/shortfin_apps/flux/examples/flux_schnell_config.json"
    FLUX_HF_MODEL="black-forest-labs/FLUX.1-schnell"

    if [[ ! -e $FLUX_SNAPSHOT ]]; then
        download_flux
        if [[ $? != 0 ]]; then
            echo "FLUX-SCHNELL Failed to download the model from hugging face.."
            exit 1
        fi
    fi

    run_flux_model

    if [[ $? != 0 ]]; then
        echo "FLUX-SCHNELL Image verification failed.."
        exit 1
    fi
else
    echo "Unsupported model : $MODEL, please specify one among (sdxl, flux_dev, flux_schnell)"
fi
