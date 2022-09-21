"""Query Github to check if current version is latest."""

import re
import urllib.error
import urllib.request
import warnings
import packaging.version

REPO_URL = "https://github.com/jmball/data_analysis"
RELEASES_URL = f"{REPO_URL}/releases"


def get_latest_release_version():
    """Scrape Github for the latest release version tag.

    Release version tags are formatted as "v[semantic version]", e.g. "v1.0.0".
    """
    # get html of releases webpage
    # suppress linter warning about urllib as url is hard coded
    try:
        page = urllib.request.urlopen(RELEASES_URL)  # nosec
    except urllib.error.URLError:
        warnings.warn("Cannot access URL. Check internet connection.")
        return None
    html = page.read().decode("utf-8")

    # official regex for a semantic version string
    # from https://semver.org#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
    sem_ver_re = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"

    # remove limitations of single line (i.e. remove ^ and $) from official string to
    # allow searching for a semantic version string as a substring, and prepend "v" to
    # match release tags
    tag_re = f"(v){sem_ver_re[1:-1]}"

    # search html to get all version tags on the releases page
    pattern = re.compile(tag_re)
    pos = 0
    version_tags = []
    while True:
        match = pattern.search(html, pos)
        if match is None:
            break
        version_tags.append(match.group())
        pos = match.start() + 1

    # get all unique semantic versions from the release tags
    versions = [tag[1:] for tag in set(version_tags)]

    # get latest version
    latest_version = None
    for version in versions:
        if latest_version is None:
            latest_version = version
        elif packaging.version.parse(version) > packaging.version.parse(latest_version):
            latest_version = version

    return latest_version


if __name__ == "__main__":
    print(get_latest_release_version())
