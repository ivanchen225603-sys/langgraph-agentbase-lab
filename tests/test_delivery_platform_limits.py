"""Platform exceptions preserve skipped evidence and reject unrelated omissions."""

import pytest

from scripts.verify_offline import allowed_platform_skips, junit_summary


@pytest.mark.parametrize(
    "platform,name,reason,allowed",
    [
        (
            "Windows-10",
            "tests.test_backup::test_symlinked_members_are_rejected[business.sqlite]",
            "filesystem does not permit creating symbolic links: [WinError 1314] permission denied",
            True,
        ),
        (
            "Linux-6",
            "tests.test_backup::test_symlinked_members_are_rejected[business.sqlite]",
            "filesystem does not permit creating symbolic links: [WinError 1314] permission denied",
            False,
        ),
        (
            "Windows-10",
            "tests.test_backup::test_symlinked_members_are_rejected[business.sqlite]",
            "filesystem does not permit creating symbolic links: [WinError 5] permission denied",
            False,
        ),
        (
            "Windows-10",
            "tests.other::omitted",
            "filesystem does not permit creating symbolic links: [WinError 1314] permission denied",
            False,
        ),
    ],
)
def test_only_declared_windows_privilege_exception_is_accepted(platform, name, reason, allowed):
    assert (
        allowed_platform_skips(
            {"skipped": [{"name": name, "type": "pytest.skip", "reason": reason}]}, platform
        )
        is allowed
    )


def test_junit_summary_keeps_skip_separate_from_passes(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites><testsuite><testcase classname="a" name="passed"/><testcase classname="a" name="omitted"><skipped type="pytest.skip" message="unavailable"/></testcase></testsuite></testsuites>',
        encoding="utf-8",
    )
    result = junit_summary(path)
    assert result == {
        "collected": 2,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": [{"name": "a::omitted", "type": "pytest.skip", "reason": "unavailable"}],
    }
