import os
import sys
import json
import random
import importlib
from packaging.version import parse

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG, LazyLLMLaunchersBase
from .base import LazyLLMDeployBase, verify_fastapi_func
from ...common import LazyLLMRegisterMetaClass
from .utils import get_log_path, make_log_dir
from .ray import reallocate_launcher, Distributed, sleep_moment


class _VllmStreamParseParametersMeta(LazyLLMRegisterMetaClass):
    def __getattribute__(cls, name):
        if name == 'stream_parse_parameters':
            if not hasattr(cls, '_stream_parse_parameters'):
                vllm_version = parse(importlib.import_module('vllm').__version__)
                cls._stream_parse_parameters = {"decode_unicode": False}
                if vllm_version <= parse("0.5.0"): cls._stream_parse_parameters.update({"delimiter": b"\0"})
            return cls._stream_parse_parameters
        return super().__getattribute__(name)


class Vllm(LazyLLMDeployBase, metaclass=_VllmStreamParseParametersMeta):
    """Model deployment class based on vLLM.

This class wraps the logic for launching a vLLM inference server, constructing commands, managing configuration parameters, and retrieving access URLs. It supports OpenAI-compatible API mode and distributed deployment via LazyLLM's launcher system.

Args:
    trust_remote_code (bool): Whether to trust remote code, allowing loading of custom modeling scripts.
    launcher (LazyLLMLaunchersBase): Launcher object controlling how the model is deployed (local, remote, distributed).
    log_path (str, optional): Directory for saving deployment logs.
    openai_api (bool): Whether to launch the server in OpenAI-compatible API mode (default: False).
    **kw: Additional deployment options, such as max sequence length, parallel configs, etc.
"""
    # keys_name_handle/default_headers/message_format will lose efficacy when openai_api is True
    keys_name_handle = {'inputs': 'prompt', 'stop': 'stop'}
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'stop': ['<|im_end|>', '<|im_start|>', '</s>', '<|assistant|>', '<|user|>', '<|system|>', '<eos>'],
        'skip_special_tokens': False,
        'temperature': 0.6,
        'top_p': 0.8,
        'max_tokens': 4096
    }
    auto_map = {'tp': 'tensor-parallel-size'}
    optional_keys = set(["max-model-len"])

    # TODO(wangzhihong): change default value for `openai_api` argument to True
    def __init__(self, trust_remote_code: bool = True, launcher: LazyLLMLaunchersBase = launchers.remote(ngpus=1),
                 log_path: str = None, openai_api: bool = False, **kw):
        self.launcher_list, launcher = reallocate_launcher(launcher)
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'dtype': 'auto',
            'kv-cache-dtype': 'auto',
            'tokenizer-mode': 'auto',
            'device': 'auto',
            'block-size': 16,
            'tensor-parallel-size': 1,
            'seed': 0,
            'port': 'auto',
            'host': '0.0.0.0',
            'max-num-seqs': 256,
            'pipeline-parallel-size': 1,
            'max-num-batched-tokens': 64000,
        })
        self._vllm_cmd = 'vllm.entrypoints.openai.api_server' if openai_api else 'vllm.entrypoints.api_server'
        self.trust_remote_code = trust_remote_code
        self.kw.update(**{key: kw[key] for key in self.optional_keys if key in kw})
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'vllm') if log_path else None
        if self.launcher_list:
            ray_launcher = [Distributed(launcher=launcher) for launcher in self.launcher_list]
            parall_launcher = [lazyllm.pipeline(sleep_moment, launcher) for launcher in ray_launcher[1:]]
            self._prepare_deploy = lazyllm.pipeline(
                ray_launcher[0], post_action=(lazyllm.parallel(*parall_launcher) if len(parall_launcher) else None))

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """Build the command to launch the vLLM inference service.

This method validates the model path and constructs an executable command string based on current configuration. In distributed mode, it will also prepend the ray cluster start command.

Args:
    finetuned_model (str): Path to the fine-tuned model.
    base_model (str): Fallback base model path if finetuned_model is invalid.
    master_ip (str): IP address of the master node in a distributed setup.

Returns:
    LazyLLMCMD: The command object with shell instruction, return value handler, and health checker.
"""
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            cmd = ''
            if self.launcher_list:
                cmd += f"ray start --address='{master_ip}' && "
            cmd += f'{sys.executable} -m {self._vllm_cmd} --model {finetuned_model} '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust-remote-code '
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        """Get the inference service URL for the vLLM deployment.

Depending on the execution mode (Display or actual deployment), this method returns the appropriate URL for accessing the model's generate endpoint.

Args:
    job (Job, optional): Deployment job object. Defaults to the module's associated job.

Returns:
    str: The HTTP URL for inference service.
"""
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/generate'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x, inputs):
        return json.loads(x)['text'][0]
