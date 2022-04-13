import inspect
import os
import json


class TraceMode:
    TRACE = 0
    EXEC = 1
    MANUAL = 2


class Tracer:
    MODE = TraceMode.TRACE

    def __init__(self, shared_path):
        self.shared_path = shared_path
        self.state_dict = dict()
        self.version = 0
        self.latest_version = 0
        if os.path.exists(shared_path):
            self.load()

    def __getattr__(self, item):
        if not hasattr(self, item):
            return self.state_dict[item]
        return super().__getattribute__(item)

    def set_traced_result(self, key, value):
        if key not in self.state_dict:
            self.state_dict[key] = dict()
        self.state_dict[key][f'version_{self.latest_version}'] = value
        self.latest_version += 1
        self.version = self.latest_version

    def save(self):
        with open(self.shared_path, 'w') as f:
            checkpoint = dict(state_dict=self.state_dict, version=self.latest_version)
            json.dump(checkpoint, f)

    def load(self):
        with open(self.shared_path, 'r') as f:
            checkpoint = json.load(f)
            self.state_dict = checkpoint['state_dict']
            self.version = self.latest_version = checkpoint['version']

    def exec(self, key, trace_dict):
        exec(compile(self.state_dict[key][f'version_{self.version}'], '<string>', 'exec'), trace_dict)


def code_trace(tracer, identifier):
    def decorator(fn):
        def _wrapper(*args, **kwargs):
            trace_dict = dict()
            fn_name = fn.__name__
            key = f'{identifier}:{fn_name}'
            maybe_self = args[0]
            # replace super to handle methods
            source = inspect.getsource(fn).replace('super()', f'super({type(maybe_self)}, {maybe_self})')
            if key not in tracer.state_dict or source != tracer.state_dict[key]:
                if tracer.MODE == TraceMode.TRACE:
                    tracer.set_traced_result(key, source)
                    tracer.save()
                elif tracer.MODE == TraceMode.EXEC:
                    tracer.exec(key, trace_dict)
                    return trace_dict[fn_name](*args, **kwargs)
            return fn(*args, **kwargs)
        return _wrapper
    return decorator


