import nox


@nox.session(
    python=["3.8", "3.7", "3.6"],
    reuse_venv=True
)
def tests(session : nox.Session):
    args = session.posargs or ["--cov", "-m", "not e2e", "./tests"]
    session.run("poetry", "install", "-E", "all", external=True )
    session.run("pytest", *args)
