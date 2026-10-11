# Sourced by the sbatch wrappers after they cd into the checkout and load .env.
# BOLDCAST_ENV is the environment prefix: a venv (has bin/activate) or a
# micromamba env.
set +u
: "${BOLDCAST_ENV:?set BOLDCAST_ENV in .env to the environment prefix}"
if [ -f "$BOLDCAST_ENV/bin/activate" ]; then
    . "$BOLDCAST_ENV/bin/activate"
else
    source ~/.bashrc
    micromamba activate "$BOLDCAST_ENV"
fi
set -u
