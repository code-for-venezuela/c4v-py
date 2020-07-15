import nox


@nox.session(python=["3.6.9", "3.7.4", "3.8.2"])
def tests(session):
    args = session.posargs or ["--cov", "-m", "not e2e", "./tests"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
