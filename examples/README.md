# Example notebooks

The examples in this directory are stored in [Literate.jl](https://github.com/fredrikekre/Literate.jl) format.

To run them locally, navigate to the directory with the example that you want to run and
start the Julia REPL and activate the project environment of the example:
```julia
julia> ] activate .
```
Alternatively, you can start Julia with `julia --project=.`. Then install all required
packages with
```julia
julia> ] instantiate
```
Afterwards simply run
```julia
julia> include("script.jl")
```

You can convert them to markdown and Jupyter notebook formats, respectively, by executing
```julia
julia> using Literate
julia> Literate.markdown("script.jl", "output_directory")
julia> Literate.notebook("script.jl", "output_directory")
```
or run
```shell
julia docs/literate.jl examples/myexample/script.jl output_directory
```

## Add a new example

Create a new subdirectory in here, and put your code in a file called `script.jl` so that it will get picked up by the automatic docs build.

Every example uses a separate project environment. Therefore you should also create a new
project environment in the directory of the example that contains all packages required by your script.
Note that the dependencies of your example *must* include the `Literate` package.

From a Julia REPL started in your example script's directory, you can run
```julia
julia> ] activate .
julia> ] add Literate
julia> ] add KernelFunctions
julia> # add any other example-specific dependencies
```
to generate the project files.

Make sure to commit both the `Project.toml` and the `Manifest.toml` file when you want to contribute your example in a pull request.
