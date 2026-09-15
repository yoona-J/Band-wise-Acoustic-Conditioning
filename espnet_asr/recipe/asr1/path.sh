#!/usr/bin/env bash

export ESPNET_ROOT=/home/user/Documents/your_root/dialect/espnet
export RECIPE_ROOT=/home/user/Documents/your_root/dialect/espnet_asr/recipe/asr1

export PATH=${RECIPE_ROOT}/utils:${RECIPE_ROOT}/utils/parallel:${PATH}
export PATH=${ESPNET_ROOT}/tools:${ESPNET_ROOT}/utils:${PATH}
export PYTHONPATH=${ESPNET_ROOT}:${PYTHONPATH:-}

# export LC_ALL=C

# ESPnet tools / utils
# export PATH=${ESPNET_ROOT}/tools:${ESPNET_ROOT}/utils:${PATH}
# export PATH=${ESPNET_ROOT}/tools/venv/bin:${PATH}

# # Python modules
# export PYTHONPATH=${ESPNET_ROOT}:${PYTHONPATH}

# Optional ESPnet extra paths
if [ -f "${ESPNET_ROOT}/tools/extra_path.sh" ]; then
    . "${ESPNET_ROOT}/tools/extra_path.sh"
fi

if [ -f "${ESPNET_ROOT}/tools/activate_python.sh" ]; then
    . "${ESPNET_ROOT}/tools/activate_python.sh"
fi

export LC_ALL=C
