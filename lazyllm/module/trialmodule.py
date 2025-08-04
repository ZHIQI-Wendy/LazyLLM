from .module import ModuleBase
from lazyllm import OptionIter, ForkProcess, LOG
import time
import copy
import multiprocessing

def get_options(x):
    if isinstance(x, ModuleBase):
        return x.options
    return []

# TODO(wangzhihong): add process pool to control parallel-number and collect result
class TrialModule(object):
    def __init__(self, m):
        self.m = m

    @staticmethod
    def work(m, q):
        # update option at module.update()
        m = copy.deepcopy(m)
        m.update()
        q.put(m.eval_result)

    def update(self):
        """Perform parallel evaluation of module configurations and collect results.

This method:
1. Retrieves configurable options of the module.
2. Launches a process for each configuration using `work()`.
3. Waits for all processes to complete.
4. Collects and logs each module's evaluation result from the queue.

No return value.
"""
        options = get_options(self.m)
        q = multiprocessing.Queue()
        ps = []
        for _ in OptionIter(options, get_options):
            p = ForkProcess(target=TrialModule.work, args=(self.m, q), sync=True)
            ps.append(p)
            p.start()
            time.sleep(1)
        [p.join() for p in ps]
        result = [q.get() for p in ps]
        LOG.info(f'{result}')
