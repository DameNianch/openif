"""Decode the verified Nasubi artifact without installing legacy global modules."""

from io import BufferedReader, BytesIO

from joblib.compressor import BinaryZlibFile
from joblib.numpy_pickle import NumpyUnpickler


def _identity(tokens):
    return tokens


class _NasubiUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("sparse_tokens", "identity"):
            return _identity
        return super().find_class(module, name)


def _decode_verified(data: bytes) -> dict:
    # joblib 1.5.3's reader for the pinned zlib-compressed NumPy pickle.
    with (
        BinaryZlibFile(BytesIO(data)) as compressed,
        BufferedReader(compressed, buffer_size=1024 * 1024) as stream,
    ):
        return _NasubiUnpickler("", stream, ensure_native_byte_order=True).load()
