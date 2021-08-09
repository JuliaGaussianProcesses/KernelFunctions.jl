# Contribution guidelines

We follow the [ColPrac guide for collaborative practices](https://colprac.sciml.ai/). New contributors should make sure to read the [first section](https://github.com/SciML/ColPrac#colprac-contributors-guide-on-collaborative-practices-for-community-packages) of that guide, but could also read the [Further Guidance](https://github.com/SciML/ColPrac#colprac-further-guidance) section if interested.


# Workflows


## Creating a new [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)

### Bumping the version number

When contributing a PR, bump the version number (defined by `version = "..."` at the top of the base `Project.toml`) accordingly (as explained by the [guidance on the versioning scheme](https://colprac.sciml.ai/#incrementing-the-package-version) in the ColPrac guide).
If unsure about what the new version should be, please just open the PR anyway -- existing contrbutors will provide a suggestion.

### Running tests locally

Run `make test`.

### Code formatting

Run `make format` before pushing your changes.


### How to make a new release (for organization members)

We use [JuliaRegistrator](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app):

1. Merge the bumped version number into `master` (see above); normally this will be the commit of a squash-merged pull request.
2. Go to the Github URL for this merge commit and write a comment saying `@JuliaRegistrator register`.

The next release will then be processed automatically. KernelFunctions.jl does use TagBot so you can ignore the message about tagging. Note that it may take around 20 minutes until the actual release appears and you can edit the release notes if needed.
