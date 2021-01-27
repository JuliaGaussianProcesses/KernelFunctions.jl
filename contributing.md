# Contribution guidelines

We follow the [ColPrac guide for collaborative practices](https://colprac.sciml.ai/). New contributors should make sure to read that guide.

## PRs and version numbers

When contributing a PR, bump the version number (defined by `version = "..."` at the top of the base `Project.toml`) accordingly: generally this should be the patch version for backwards-compatible changes.

## Workflows

### How to make a new release (for organization members)

We use [JuliaRegistrator](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app):

1. Merge the bumped version number into `master` (see above); normally this will be the commit of a squash-merged pull request.
2. Go to the Github URL for this merge commit and write a comment saying `@JuliaRegistrator register`.

The next release will then be processed automatically. KernelFunctions.jl does use TagBot so you can ignore the message about tagging. Note that it may take around 20 minutes until the actual release appears and you can edit the release notes if needed.
