# How to build the docs locally

Assuming you are in this (docs/) directory:

## Setup

First, make sure the docs dependencies are installed:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This will use the last *released* KernelFunctions.jl for building the docs. 

If instead you want to reflect the *local* state of the Git repository in your built docs, next change the dependency to the development version:
```bash
julia --project=. -e "using Pkg; Pkg.develop(path=\"/path/to/KernelFunctions.jl/\")"
```
In a bash-like shell on Unix, you can also use the following command from the docs/ directory to get the absolute path:
```bash
julia --project=. -e "using Pkg; Pkg.develop(path=\"$(readlink -f ..)\")"
```

You can undo the pinning to the local path that was created by `Pkg.develop` through running
```bash
julia --project=. -e 'using Pkg; Pkg.free("KernelFunctions")'
```

## Build

To actually build the docs, run
```bash
julia --project=. make.jl
```
The built docs will be underneath build/, and are best viewed in a browser.

If you want to iteratively edit and view the docs, it will be faster to re-run from within the same Julia REPL:
```julia
julia> include("make.jl")
```

# How to contribute to the docs

## To add additional docs dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.add("NewDependency")'
```
and commit the changes to Project.toml


## To add examples

See [../examples](../examples/README.md)
