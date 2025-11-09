# Vendored Dependencies

This directory contains patched versions of upstream dependencies that are required for compatibility with the updated dependency versions.

## ndarray-csv

Patched version of [ndarray-csv](https://github.com/paulkernfeld/ndarray-csv) with updated ndarray version constraint to support ndarray 0.17.

**Change**: Updated `ndarray` dependency from `">= 0.7, < 0.17"` to `">= 0.7, < 0.18"` in Cargo.toml.

This is a minimal change that allows ndarray-csv to work with ndarray 0.17.x. The upstream project should accept this change once ndarray 0.17 becomes more widely adopted.

## rust-numpy

Patched version of [rust-numpy](https://github.com/PyO3/rust-numpy) with updated ndarray version constraint to support ndarray 0.17.

**Change**: Updated `ndarray` dependency from `">= 0.15, < 0.17"` to `">= 0.15, < 0.18"` in Cargo.toml.

This is a minimal change that allows rust-numpy to work with ndarray 0.17.x. The upstream project should accept this change once ndarray 0.17 becomes more widely adopted.

## Notes

These patches are temporary workarounds until the upstream projects update their dependencies. Both changes have been tested and verified to build and pass all tests successfully.
