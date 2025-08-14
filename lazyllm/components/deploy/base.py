import time
from ..core import ComponentBase
import lazyllm
from lazyllm import launchers, flows, LOG
from ...components.utils.file_operate import _image_to_base64, _audio_to_base64, ocr_to_base64
import random


class LazyLLMDeployBase(ComponentBase):
    keys_name_handle = None
    message_format = None
    default_headers = {'Content-Type': 'application/json'}
    stream_url_suffix = ''
    stream_parse_parameters = {}

    encoder_map = dict(image=_image_to_base64, audio=_audio_to_base64, ocr_files=ocr_to_base64)

    @staticmethod
    def extract_result(output, inputs):
        return output

    def __init__(self, *, launcher=launchers.remote()):
        super().__init__(launcher=launcher)


class DummyDeploy(LazyLLMDeployBase, flows.Pipeline):
    """DummyDeploy(launcher=launchers.remote(sync=False), *, stream=False, **kw)

A mock deployment class for testing purposes. It extends both `LazyLLMDeployBase` and `flows.Pipeline`,
simulating a simple pipeline-style deployable service with optional streaming support.

This class is primarily intended for internal testing and demonstration. It receives inputs in the format defined
by `message_format`, and returns a dummy response or a streaming response depending on the `stream` flag.

Attributes:
- keys_name_handle (dict): Mapping of input keys for request formatting.
- message_format (dict): Default request template including input and generation parameters.

Parameters:
- launcher: Deployment launcher instance, defaulting to `launchers.remote(sync=False)`.
- stream (bool): Whether to simulate streaming output.
- kw: Additional keyword arguments passed to the superclass.

Methods:
- __call__(*args): Starts the deployment and returns the service URL.
- __repr__(): Returns a string representation of the underlying pipeline.
"""
    keys_name_handle = {'inputs': 'inputs'}
    message_format = {
        'inputs': '',
        'parameters': {
            'do_sample': False,
            'temperature': 0.1,
        }
    }

    def __init__(self, launcher=launchers.remote(sync=False), *, stream=False, **kw):
        super().__init__(launcher=launcher)

        def func():

            def impl(x):
                LOG.info(f'input is {x["inputs"]}, parameters is {x["parameters"]}')
                return f'reply for {x["inputs"]}, and parameters is {x["parameters"]}'

            def impl_stream(x):
                for s in ['reply', ' for', f' {x["inputs"]}', ', and',
                          ' parameters', ' is', f' {x["parameters"]}']:
                    yield s
                    time.sleep(0.2)
            return impl_stream if stream else impl
        flows.Pipeline.__init__(self, func,
                                lazyllm.deploy.RelayServer(port=random.randint(30000, 40000), launcher=launcher))

    def __call__(self, *args):
        url = flows.Pipeline.__call__(self)
        LOG.info(f'dummy deploy url is : {url}')
        return url

    def __repr__(self):
        return flows.Pipeline.__repr__(self)

def verify_func_factory(error_message='ERROR:',
                        running_message='Uvicorn running on'):
    def verify_func(job):
        while True:
            line = job.queue.get()
            if line.startswith(error_message):
                LOG.error(f"Capture error message: {line} \n\n")
                return False
            elif running_message in line:
                LOG.info(f"Capture startup message: {line}")
                break
            if job.status == lazyllm.launchers.status.Failed:
                LOG.error("Service Startup Failed.")
                return False
        return True
    return verify_func

verify_fastapi_func = verify_func_factory()
