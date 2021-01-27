# Contribution guidelines

## How to make a new release

1. Bump the version number defined by `version = "..."` at the top of `Project.toml`.
2. Go to the Github URL for the commit to `master` that bumps the version number (normally the commit of a squash-merged pull request).
3. Write a comment on that commit saying `@JuliaRegistrator register`.

The next release will then be processed automatically. KernelFunctions.jl does use TagBot so you can ignore the message about tagging. Note that it may take around 20 minutes until the actual release appears and you can edit the release notes if needed.
