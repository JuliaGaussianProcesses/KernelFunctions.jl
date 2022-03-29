# Contribution guidelines

We follow the [ColPrac guide for collaborative practices](https://colprac.sciml.ai/). New contributors should make sure to read the [first section](https://github.com/SciML/ColPrac#colprac-contributors-guide-on-collaborative-practices-for-community-packages) of that guide, but could also read the [Further Guidance](https://github.com/SciML/ColPrac#colprac-further-guidance) section if interested.


# Workflows


## Creating a new [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)

### Bumping the version number

When contributing a PR, bump the version number (defined by `version = "..."` at the top of the base `Project.toml`) accordingly (as explained by the [guidance on the versioning scheme](https://colprac.sciml.ai/#incrementing-the-package-version) in the ColPrac guide).
If unsure about what the new version should be, please just open the PR anyway -- existing contributors will provide a suggestion.

### Running tests locally

Firstly, run `using Pkg; Pkg.develop("KernelFunctions")` at the Julia REPL, and navigate to `~/.julia/dev/KernelFunctions`.

Running `make test` will now run the entire test suite.
These tests can take a long time to run, so it's often a good idea to simply comment out the blocks of tests not required in `test/runtests.jl`.
Test files are paired 1-1 with source files, so if you're modifying code in `src/foo.jl`, you should only need to run the tests in `test/foo.jl` during development.

### Code formatting

Run `make format` before pushing your changes.


### How to make a new release (for organization members)

We use [JuliaRegistrator](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app):

On Github, go to the commit that you want to register (this should normally be the commit of a squash-merged pull request on the `master` branch that has bumped the version number) and add a comment saying `@JuliaRegistrator register`.

The next release will then be processed automatically. KernelFunctions.jl does use TagBot so you can ignore the message about tagging. Note that it may take around 20 minutes until the actual release appears and you can edit the release notes if needed.
