# Example notebooks

The examples in this directory are stored in [Literate.jl](https://github.com/fredrikekre/Literate.jl) format.
To run them locally, simply `include("script.jl")` in the Julia REPL.

You can convert them to markdown and Jupyter notebook formats, respectively, by executing
```julia
julia> using Literate
julia> Literate.markdown("script.jl", "output_directory")
julia> Literate.notebook("script.jl", "output_directory")
```

## Add a new example

Create a new subdirectory in here, and put your code in a file called `script.jl` so that it will get picked up by the automatic docs build.
You should also add a `Project.toml` file with all your script's dependencies (including KernelFunctions.jl).
