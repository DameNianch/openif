# Preparing the first PyPI release

OpenIF 0.1.0 is a release candidate maintained by DameNianch. The current GitHub Actions workflow builds, installs and tests the distributions. It has read-only repository permissions.

1. Confirm distribution rights for the package, semantic rules, vocabulary, IDF and head. Review the dependency notices and the terms applicable to the tools used to create the code. Approve the exact release commit, model revision, package contents and repository visibility.
2. Build with `python -m build` and check with `python -m twine check --strict dist/*`. Install both archives into fresh Python 3.12 environments. Exercise the README examples online and from the populated cache offline.
3. Recheck the normalized name `openif` on PyPI. An absent project page leaves availability unconfirmed.
4. After release approval, add a manually triggered GitHub Actions workflow, e.g. `release.yml`, with a protected `pypi` environment. Grant `id-token: write` to its publishing job and pass the reviewed archives to the PyPA publishing action.
5. In DameNianch's PyPI account, go to **Publishing** and register a pending GitHub publisher for project `openif`, owner `DameNianch`, repository `openif`, that workflow filename and the matching environment. A pending publisher takes effect on the first successful upload. The account holder can then verify the release page and install the published wheel in a new environment.

[PyPI Trusted Publishing for a new project](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
