import nox


@nox.session(python=["3.6", "3.7", "3.8"])
def tests(session):
    args = session.posargs or ["--cov", "-m", "not e2e", "./tests"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
