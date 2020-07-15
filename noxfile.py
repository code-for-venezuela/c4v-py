import nox


@nox.session(python=["3.6.11", "3.7.8", "3.8.3"])
def tests(session):
    args = session.posargs or ["--cov", "-m", "not e2e", "./tests"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
