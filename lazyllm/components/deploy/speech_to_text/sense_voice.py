import os
import importlib.util

import lazyllm
from lazyllm.components.utils.file_operate import _base64_to_file, _is_base64_with_mime
from lazyllm import LOG, LazyLLMLaunchersBase, is_valid_url, is_valid_path
from ..base import LazyLLMDeployBase
from ...utils.downloader import ModelManager
from lazyllm.thirdparty import funasr
from typing import Optional

supported_formats = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma')

class SenseVoice(object):
    """SenseVoice(base_path, source=None, init=False)

A speech-to-text wrapper using FunASR models for lazy initialization and audio transcription.
This class supports automatic model downloading, safe initialization, and inference from audio paths or URLs.

Parameters:
- base_path (str): Path or model identifier to download the STT model.
- source (str, optional): Model source name; defaults to `lazyllm.config['model_source']`.
- init (bool): Whether to initialize the model immediately on creation.

Attributes:
- base_path (str): Final resolved path of the model after download.
- model: Loaded FunASR model instance.
- init_flag: A lazy flag used to ensure model is only loaded once.

Methods:
- __call__(string: str | dict) -> str:
    Transcribes the input audio file or URL to text. Accepts base64-encoded content, file paths, or URLs.
- load_stt():
    Loads the FunASR speech-to-text model and related VAD (Voice Activity Detection).
- rebuild(base_path, init):
    Rebuilds the class instance (used for serialization).
- __reduce__():
    Supports pickling by ensuring proper lazy-loading on deserialization.
"""
    def __init__(self, base_path, source=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_stt)

    def load_stt(self):
        """load_stt()

Loads the speech-to-text model using FunASR with optional support for Huawei NPU via `torch_npu`.

The method initializes the model with the following characteristics:
- Uses `fsmn-vad` for voice activity detection with long utterance support.
- Sets maximum single segment time to 30 seconds.
- Selects `cuda:0` as the default inference device.

The model is stored in `self.model` and will be used to transcribe audio input.

Note:
If `torch_npu` is available in the environment, the function attempts to load it for potential Huawei Ascend acceleration.
"""
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        self.model = funasr.AutoModel(
            model=self.base_path,
            trust_remote_code=False,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_stt)
        if isinstance(string, dict):
            if string['audio']:
                string = string['audio'][-1] if isinstance(string['audio'], list) else string['audio']
            else:
                string = string['inputs']
        assert isinstance(string, str)
        string = string.strip()
        try:
            string = _base64_to_file(string) if _is_base64_with_mime(string) else string
        except Exception as e:
            LOG.error(f"Error processing base64 encoding: {e}")
            return f"Error processing base64 encoding {e}"
        if not string.endswith(supported_formats):
            return f"Only {', '.join(supported_formats)} formats in the form of file paths or URLs are supported."
        if not is_valid_path(string) and not is_valid_url(string):
            return f"This {string} is not a valid URL or file path. Please check."
        res = self.model.generate(
            input=string,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = funasr.utils.postprocess_utils.rich_transcription_postprocess(res[0]["text"])
        return text

    @classmethod
    def rebuild(cls, base_path, init):
        """rebuild(base_path: str, init: bool) -> SenseVoice

Class method used to reconstruct a `SenseVoice` instance during deserialization (e.g., when using `cloudpickle`).

Parameters:
- base_path (str): Path to the speech-to-text model.
- init (bool): Whether to immediately initialize and load the model upon creation.

Returns:
- A new instance of `SenseVoice` with the specified configuration.

Note:
This method is internally used to support model serialization and multiprocessing compatibility.
"""
        return cls(base_path, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return SenseVoice.rebuild, (self.base_path, init)

class SenseVoiceDeploy(LazyLLMDeployBase):
    """SenseVoice Model Deployment Class. This class is used to deploy the SenseVoice model to a specified server for network invocation.

`__init__(self, launcher=None)`
Constructor, initializes the deployment class.

Args:
    launcher (lazyllm.launcher): An instance of the launcher used to start the remote service.

`__call__(self, finetuned_model=None, base_model=None)`
Deploys the model and returns the remote service address.

Args:
    finetuned_model (str): If provided, this model will be used for deployment; if not provided or the path is invalid, `base_model` will be used.
    base_model (str): The default model, which will be used for deployment if `finetuned_model` is invalid.
    Return (str): The URL address of the remote service.

Notes:
    - Input for infer: `str`. The audio path or link.
    - Return of infer: `str`. The recognized content.
    - Supported models: [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)


Examples:
    >>> import os
    >>> import lazyllm
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import SenseVoiceDeploy
    >>> deployer = SenseVoiceDeploy(launchers.remote())
    >>> url = deployer(base_model='SenseVoiceSmall')
    >>> model = UrlModule(url=url)
    >>> model('path/to/audio') # support format: .mp3, .wav
    ... xxxxxxxxxxxxxxxx
    """
    keys_name_handle = {
        'inputs': 'inputs',
        'audio': 'audio',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'audio': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher: Optional[LazyLLMLaunchersBase] = None,
                 log_path: Optional[str] = None, trust_remote_code: bool = True, port: Optional[int] = None):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.pt', '.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                        f"base_model({base_model}) will be used")
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(port=self._port, func=SenseVoice(finetuned_model), launcher=self._launcher,
                                          log_path=self._log_path, cls='sensevoice')()
