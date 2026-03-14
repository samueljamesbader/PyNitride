from contextlib import contextmanager

@contextmanager
def returner_context(val): yield val