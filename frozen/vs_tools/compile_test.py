import inspect

from frozen.vs_tools.trace import Tracer, code_trace

tracer = Tracer('/project/trace.json')


class Dummy:
    def __init__(self):
        self.dummy_attr = 1

    @code_trace(tracer, identifier='dummy')
    def dummy_method(self):
        return self.dummy_attr


if __name__ == "__main__":
    tracer.version = 0
    tracer.MODE = 1
    print(Dummy().dummy_method())

