import nox


@nox.session
def lint(session: nox.Session) -> None:
    """Look for lint."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session(venv_backend="none")
def bench(session: nox.Session) -> None:
    """Run the benchmarks against the landlab installed in the current environment.

    Extra arguments go to ``asv run``, e.g. ``nox -s bench -- --bench openmp``.
    """
    session.run("asv", "machine", "--yes")
    session.run("asv", "run", "--python=same", "--show-stderr", *session.posargs)


@nox.session(name="bench-check", venv_backend="none")
def bench_check(session: nox.Session) -> None:
    """Check that every benchmark imports and runs once (no timing)."""
    session.run("asv", "check", "--python=same")
    session.run("asv", "machine", "--yes")
    session.run("asv", "run", "--python=same", "--quick", "--show-stderr")
