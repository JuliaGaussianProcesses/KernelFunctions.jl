# How to build the docs locally

If you just want to modify or add an example, you can [build just the example](../examples/README.md) without having to build the full documentation locally.

If you want to build the documentation, navigate to this (docs/) directory and open a Julia REPL.

## Instantiation

First activate the documentation environment:
```julia
julia> ] activate .
```
Alternatively, you can start Julia with `julia --project=.`.

Then install all packages:
```julia
julia> ] instantiate
```
By default, this will use the development version of KernelFunctions in the parent directory.

## Build

To build the documentation, run (after activating the documentation environment)
```julia
julia> include("make.jl")
```
You can speed up the process if you do not execute the examples and comment out the
relevant sections in `docs/make.jl`.

The output is in the `docs/build/` directory and best viewed in a browser.
The documentation uses pretty URLs which can be a hindrance if you browse the documentation locally.
The [Documenter documentation](https://juliadocs.github.io/Documenter.jl/stable/man/guide/#Building-an-Empty-Document) suggests that

> You can run a local web server out of the `docs/build` directory. One way to accomplish
> this is to install the [LiveServer](https://github.com/tlienart/LiveServer.jl) Julia
> package. You can then start the server with `julia -e 'using LiveServer; serve(dir="docs/build")'`.
> Alternatively, if you have Python installed, you can start one with
> `python3 -m http.server --bind localhost` (or `python -m SimpleHTTPServer` with Python 2).

If you make any changes, you can run
```julia
julia> include("make.jl")
```
again to rebuild the documentation.

# How to contribute to the docs

## To add additional docs dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.add("NewDependency")'
```
and commit the changes to Project.toml and Manifest.toml.


## To add examples

See [../examples](../examples/README.md)
