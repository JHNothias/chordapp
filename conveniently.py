from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class PoolExecutor():
    def __init__(self):
        self.start()

    def start(self):
        self.T = ThreadPoolExecutor()
        self.P = ProcessPoolExecutor()
    def shutdown(self):
        self.T.shutdown()
        self.P.shutdown()
    def reset(self):
        self.shutdown()
        self.start()

Executor = PoolExecutor()

def threaded(f, *args, **kwargs):
    def decorated(*args, **kwargs):
        T = Executor.T.submit(f, *args, *kwargs)
        return T
    return decorated

def delegated(f, *args, **kwargs):
    def decorated(*args, **kwargs):
        P = Executor.P.submit(f, *args, *kwargs)
        return P
    return decorated
