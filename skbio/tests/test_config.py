# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
from unittest.mock import patch

from skbio._config import get_config, set_config, _resolve_engine


class TestOptions(TestCase):
    def tearDown(self):
        # Restore the default engine in case a test changed it.
        set_config("engine", "cython")

    def test_set_config_bad_option(self):
        with self.assertRaisesRegex(KeyError, "Unknown option: 'nonsense'."):
            set_config("nonsense", "asdf")

    def test_set_config_bad_value(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported value 'asdf' for 'table_output'."
        ):
            set_config("table_output", "asdf")

    def test_get_config_bad_option(self):
        with self.assertRaisesRegex(KeyError, "Unknown option: 'frontend'."):
            get_config("frontend")

    def test_engine_default_is_cython(self):
        self.assertEqual(get_config("engine"), "cython")

    def test_set_engine_valid(self):
        set_config("engine", "numba")
        self.assertEqual(get_config("engine"), "numba")

    def test_set_engine_bad_value(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported value 'julia' for 'engine'."
        ):
            set_config("engine", "julia")


class TestResolveEngine(TestCase):
    def setUp(self):
        self._original = get_config("engine")

    def tearDown(self):
        set_config("engine", self._original)

    def test_none_uses_global_default(self):
        set_config("engine", "cython")
        self.assertEqual(_resolve_engine(None, ("cython", "numba")), "cython")

    def test_explicit_cython(self):
        self.assertEqual(_resolve_engine("cython", ("cython", "numba")), "cython")

    def test_unsupported_value_raises(self):
        with self.assertRaisesRegex(ValueError, "engine='julia' is not supported"):
            _resolve_engine("julia", ("cython", "numba"))

    def test_engine_not_in_supported_raises(self):
        with self.assertRaisesRegex(ValueError, "engine='numba' is not supported"):
            _resolve_engine("numba", ("cython",))

    def test_numba_requested_but_absent_raises(self):
        # Simulate Numba not being installed.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "numba":
                raise ImportError("no numba")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "requires the optional numba"):
                _resolve_engine("numba", ("cython", "numba"))

    def test_fast_resolves_to_what_the_caller_names(self):
        self.assertEqual(
            _resolve_engine("fast", ("cython", "numba"), fast="numba"), "numba"
        )
        self.assertEqual(
            _resolve_engine("fast", ("cython", "numba"), fast="cython"), "cython"
        )

    def test_fast_without_a_target_falls_back_to_the_default(self):
        # A function with nothing faster to offer does not pass fast=, and
        # engine="fast" then has to be a no-op rather than an error.
        set_config("engine", "cython")
        self.assertEqual(_resolve_engine("fast", ("cython", "numba")), "cython")

    def test_fast_is_resolved_after_the_global_default(self):
        # Placing the branch after the global lookup means one check covers an
        # explicit engine="fast" and a global default of "fast" alike. The
        # global cannot be set through set_config today, so it is set directly.
        from skbio._config import _SKBIO_OPTIONS

        previous = _SKBIO_OPTIONS["engine"]
        _SKBIO_OPTIONS["engine"] = "fast"
        try:
            self.assertEqual(
                _resolve_engine(None, ("cython", "numba"), fast="numba"), "numba"
            )
        finally:
            _SKBIO_OPTIONS["engine"] = previous

    def test_fast_target_still_checked_against_supported(self):
        with self.assertRaisesRegex(ValueError, "engine='numba' is not supported"):
            _resolve_engine("fast", ("cython",), fast="numba")


if __name__ == "__main__":
    main()
