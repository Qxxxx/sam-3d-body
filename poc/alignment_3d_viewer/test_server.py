from __future__ import annotations

import unittest

from server import _parse_byte_range


class ByteRangeTests(unittest.TestCase):
    def test_explicit_range(self) -> None:
        self.assertEqual(_parse_byte_range("bytes=100-199", 1000), (100, 199))

    def test_open_ended_range(self) -> None:
        self.assertEqual(_parse_byte_range("bytes=900-", 1000), (900, 999))

    def test_suffix_range(self) -> None:
        self.assertEqual(_parse_byte_range("bytes=-100", 1000), (900, 999))

    def test_rejects_multiple_or_out_of_bounds_ranges(self) -> None:
        for value in ("bytes=1000-", "bytes=20-10", "bytes=0-1,5-6"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_byte_range(value, 1000)


if __name__ == "__main__":
    unittest.main()
