# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.launcher)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.launcher)
add_example = functools.partial(utils.add_example, module=lazyllm.launcher)

add_chinese_doc('Job', '''\
用于管理延迟执行的作业对象，结合命令（LazyLLMCMD）和启动器（Launcher）实现作业调度执行。

该类支持作业队列、命令封装、同步或异步执行、执行返回值捕获等功能。

Args:
    cmd (LazyLLMCMD): 封装了实际运行命令的命令对象。
    launcher (Any): 用于调度命令的启动器实例，通常实现 run(job) 接口。
    sync (bool): 是否同步运行作业。若为 False，则为异步启动模式。
''')

add_english_doc('Job', '''\
A job object that manages the execution of delayed commands using a launcher.

This class encapsulates job queueing, command preparation, synchronous or asynchronous execution, and capturing return values.

Args:
    cmd (LazyLLMCMD): The command object that wraps the actual execution logic.
    launcher (Any): The launcher responsible for running the job, typically implements a run(job) interface.
    sync (bool): Whether to run the job synchronously. If False, the job runs in asynchronous mode.
''')

add_example('Job', ['''\
>>> from lazyllm.runtime.job import Job
>>> from lazyllm.runtime.cmd import LazyLLMCMD
>>> from lazyllm.runtime.launcher import LocalLauncher

>>> # 定义一个命令，例如执行 echo 测试命令
>>> cmd = LazyLLMCMD(cmd=["echo", "Hello, LazyLLM!"])
>>> launcher = LocalLauncher()

>>> # 创建 Job 实例
>>> job = Job(cmd, launcher)

>>> # 获取封装后的可执行命令
>>> fixed_cmd = job.get_executable_cmd()
>>> print(fixed_cmd.cmd)
... ['echo', 'Hello, LazyLLM!']

>>> # 启动作业（通过 launcher 启动，此处假设 LocalLauncher 已支持 run）
>>> launched_job = launcher.run(job)

>>> # 等待作业完成（如果是同步执行）
>>> job.wait()

>>> # 获取命令返回值或结果
>>> print(job.return_value)
... <Job 对象本身或返回结果>
'''])

add_chinese_doc('Job.get_executable_cmd', '''\
生成最终可执行命令。

如果已缓存固定命令（fixed），则直接返回。否则根据原始命令进行包裹（wrap）并缓存为 `_fixed_cmd`。

Args:
    fixed (bool): 是否使用已固定的命令对象（若已存在）。

Returns:
    LazyLLMCMD: 可直接执行的命令对象。
''')

add_english_doc('Job.get_executable_cmd', '''\
Generate the final executable command.

If a fixed command already exists, return it. Otherwise, wrap the original command and cache it as `_fixed_cmd`.

Args:
    fixed (bool): Whether to use the cached fixed command.

Returns:
    LazyLLMCMD: The executable command object.
''')

add_chinese_doc('Job.start', '''\
对外接口：启动作业，并支持失败时的自动重试。

若作业执行失败，会根据 `restart` 参数控制重试次数。

Args:
    restart (int): 重试次数。默认为 3。
    fixed (bool): 是否使用固定后的命令。用于避免多次构建。
''')

add_english_doc('Job.start', '''\
Public interface to start the job with optional retry on failure.

If the job fails, retries execution based on the `restart` parameter.

Args:
    restart (int): Number of times to retry upon failure. Default is 3.
    fixed (bool): Whether to use the fixed version of the command.
''')

add_chinese_doc('Job.restart', '''\
重新启动作业流程。

该函数会先停止已有进程，等待 2 秒后重新启动作业。

Args:
    fixed (bool): 是否使用固定后的命令。
''')

add_english_doc('Job.restart', '''\
Restart the job by first stopping it and then restarting after a short delay.

Args:
    fixed (bool): Whether to reuse the fixed command object.
''')

add_chinese_doc('Job.wait', '''\
挂起当前线程，等待作业执行完成。当前实现为空方法（子类可重写）。
''')

add_english_doc('Job.wait', '''\
Suspend the current thread until the job finishes.

Empty implementation by default; can be overridden in subclasses.
''')

add_chinese_doc('Job.stop', '''\
停止当前作业。

该方法为接口定义，需子类实现，当前抛出 NotImplementedError。
''')

add_english_doc('Job.stop', '''\
Stop the current job.

This method is an interface placeholder and must be implemented by subclasses.
''')
