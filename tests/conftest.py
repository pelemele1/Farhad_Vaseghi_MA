import os

# Loading both torch (via src.models.backbone -> ultralytics) and scipy/
# scikit-image (via the vendored physical_lens_soiling water-droplet code)
# in the same process crashes on Windows with "Fatal Python error: Aborted"
# -- both bring their own bundled Intel OpenMP runtime (libiomp5md.dll), and
# loading it twice aborts the process on first real use. This is the
# standard workaround; it must be set before either library is imported, so
# it lives here in conftest.py, which pytest loads before collecting tests.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
