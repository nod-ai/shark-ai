#!/bin/bash

function wait_for_server() {
    SERVER_UP=0
    while [[ $SERVER_UP != 200 && -e /proc/$SHORTFIN_PROCESS ]]; do
        SERVER_UP=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:$1/health)
        sleep 3
    done
}
