# Releasing OpenIF to PyPI

A maintainer starts **Release to PyPI** from the exact `main` branch. Pushes, pull requests, merges and tags have no publishing trigger. The first planned release is `openif==0.1.0`. Version changes are explicit edits to `pyproject.toml`.

## One-time setup

1. In **DameNianch/openif → Settings → Environments**, create or open `pypi`. Under **Deployment branches and tags**, choose **Selected branches and tags**. Add a rule of type **Branch** with the exact pattern `main`. Ensure this is the only rule, with no tag patterns. Save before setting up the publisher. For a private repository, deployment branch restrictions require a GitHub plan that supports them. [GitHub environment settings](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
2. Sign in to PyPI as **DameNianch**. Under **Your account → Publishing**, add a pending **GitHub** publisher with the values below. The first successful upload creates the project and activates the publisher. A pending publisher does not reserve the name. For a project you already own, use its **Manage → Publishing** settings. [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)

| PyPI setting | Value |
|---|---|
| Project name | `openif` |
| GitHub owner | `DameNianch` |
| Repository | `openif` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

## Publish a version

1. Review distribution rights for the code, rules, fitted model and dependencies. Keep the current license-under-review statement and third-party notices. The maintainer decides whether distribution is authorized.
2. Check the intended version in `pyproject.toml`. Keep `0.1.0` for the first release. For later releases, select an unpublished version and merge the intended changes, including the version update, into `main`.
3. Open **GitHub → Actions → Release to PyPI → Run workflow**. Select the **main** branch, verify its commit, and click **Run workflow**. The workflow must already be present on the default branch to appear in the menu.
4. Monitor **build** and **publish**. Build checks out the dispatched commit, tests the wheel against the pinned Hub model, and tests the sdist offline. Publish downloads the same artifact by ID, verifies its digest and checks the version on PyPI immediately before uploading. Only this job uses the `pypi` environment and requests a short-lived OIDC token. A published version, inconclusive index response or failed verification stops the upload.
5. Open the [PyPI project](https://pypi.org/project/openif/) and verify the version, wheel and sdist. If an upload failed partway, inspect the release's file list before taking further action. The workflow rejects any version already present. Select a fresh version for a subsequent release.
6. After PyPI lists the release, make a fresh Python 3.12 environment and verify installation from PyPI. Replace `0.1.0` in the assertion when verifying a later release.

```bash
python3.12 -m venv .venv-pypi
source .venv-pypi/bin/activate
python -m pip install --index-url https://pypi.org/simple openif
python -I - <<'PY'
from importlib.metadata import version
from openif import Classifier

assert version("openif") == "0.1.0"
model = Classifier.from_pretrained("openif/gpr-1-nasubi-massive")
assert model.predict("明日の天気を教えて") == "weather_query"
PY
```

After the first successful install, update the README in a separate PR with the verified `python -m pip install openif` instructions. Keep the source-checkout instructions for development.
