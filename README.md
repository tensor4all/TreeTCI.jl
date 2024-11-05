# T4ATemplate

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/T4ATemplate.jl/dev)
[![CI](https://github.com/tensor4all/T4ATemplate.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tensor4all/T4ATemplate.jl/actions/workflows/CI.yml)

The [T4ATemplate module](https://github.com/tensor4all/T4ATemplate.jl) provides a tempalte for a Julia package.

1. Git clone this repository to a local machine.
2. Update UUID and the package name in Project.toml.
3. Rename `src/T4ATemplate.jl`.
4. Replace `T4ATempalte` throughtout `src/` and `test/`.


One can use the following command on Mac to generate a new UUID:

```
uuidgen | tr "[:upper:]" "[:lower:]"
```
