import json
import time
import os
import requests
from typing import Tuple, Any, Dict, List
from urllib.parse import urljoin
import uuid

import lazyllm
from lazyllm.thirdparty import jwt
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase
from ..fileHandler import FileHandlerBase


class _SenseNovaBase(object):

    def _get_api_key(self, api_key: str, secret_key: str):
        if not api_key and not secret_key:
            api_key, secret_key = lazyllm.config['sensenova_api_key'], lazyllm.config['sensenova_secret_key']
        if secret_key.startswith('sk-'): api_key, secret_key = secret_key, None
        if not api_key: raise ValueError('api_key is required for sensecore')
        if not api_key.startswith('sk-'):
            if ':' in api_key: api_key, secret_key = api_key.split(':', 1)
            assert secret_key, 'secret_key should be provided with sensecore api_key'
            api_key = SenseNovaModule.encode_jwt_token(api_key, secret_key)
        return api_key

    @staticmethod
    def encode_jwt_token(ak: str, sk: str) -> str:
        headers = {'alg': 'HS256', 'typ': 'JWT'}
        payload = {
            'iss': ak,
            # Fill in the expected effective time, which represents the current time +24 hours
            'exp': int(time.time()) + 86400,
            # Fill in the desired effective time starting point, which represents the current time
            'nbf': int(time.time())
        }
        token = jwt.encode(payload, sk, headers=headers)
        return token

class SenseNovaModule(OnlineChatModuleBase, FileHandlerBase, _SenseNovaBase):
    TRAINABLE_MODEL_LIST = ["nova-ptc-s-v2"]
    VLM_MODEL_LIST = ['SenseNova-V6-Turbo', 'SenseChat-Vision']

    def __init__(self, base_url: str = "https://api.sensenova.cn/compatible-mode/v1/", model: str = "SenseChat-5",
                 api_key: str = None, secret_key: str = None, stream: bool = True,
                 return_trace: bool = False, **kwargs):
        api_key = self._get_api_key(api_key, secret_key)
        OnlineChatModuleBase.__init__(self, model_series="SENSENOVA", api_key=api_key, base_url=base_url,
                                      model_name=model, stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        self._deploy_paramters = None
        self._vlm_force_format_input_with_files = True

    def _get_system_prompt(self):
        return "You are an AI assistant, developed by SenseTime."

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')

    def _convert_file_format(self, filepath: str) -> None:
        with open(filepath, 'r', encoding='utf-8') as fr:
            dataset = [json.loads(line) for line in fr]

        json_strs = []
        for ex in dataset:
            lineEx = []
            messages = ex.get("messages", [])
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role in ["system", "knowledge", "user", "assistant"]:
                    lineEx.append({"role": role, "content": content})
            json_strs.append(json.dumps(lineEx, ensure_ascii=False))

        return "\n".join(json_strs)

    def _upload_train_file(self, train_file):
        headers = {
            "Authorization": "Bearer " + self._api_key
        }
        url = self._train_parameters.get("upload_url", "https://file.sensenova.cn/v1/files")
        self.get_finetune_data(train_file)
        file_object = {
            # The correct format should be to pass in a tuple in the format of:
            # (<fileName>, <fileObject>, <Content-Type>),
            # where fileObject refers to the specific value.

            "description": (None, "train_file", None),
            "scheme": (None, "FINE_TUNE_2", None),
            "file": (os.path.basename(train_file), self._dataHandler, "application/json")
        }

        train_file_id = None
        with requests.post(url, headers=headers, files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException(r.text)

            train_file_id = r.json()["id"]
            # delete temporary training file
            self._dataHandler.close()
            lazyllm.LOG.info(f"train file id: {train_file_id}")

        def _create_finetuning_dataset(description, files):
            url = urljoin(self._base_url, "fine-tune/datasets")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            data = {
                "description": description,
                "files": files
            }
            with requests.post(url, headers=headers, json=data) as r:
                if r.status_code != 200:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

                dataset_id = r.json()["dataset"]["id"]
                status = r.json()["dataset"]["status"]
                url = url + f"/{dataset_id}"
                while status.lower() != "ready":
                    try:
                        time.sleep(10)
                        with requests.get(url, headers=headers) as r:
                            if r.status_code != 200:
                                raise requests.RequestException(r.text)

                            dataset_id = r.json()["dataset"]["id"]
                            status = r.json()["dataset"]["status"]
                    except Exception as e:
                        lazyllm.LOG.error(f"error: {e}")
                        raise ValueError(f"created datasets {dataset_id} failed")
                return dataset_id

        return _create_finetuning_dataset("fine-tuning dataset", [train_file_id])

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = urljoin(self._base_url, "fine-tunes")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model": train_model,
            "training_file": train_file_id,
            "suffix": kw.get("suffix", "ft-" + str(uuid.uuid4().hex))
        }
        if "training_parameters" in kw.keys():
            data.update(kw["training_parameters"])

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException(r.text)

            fine_tuning_job_id = r.json()["job"]["id"]
            status = r.json()["job"]["status"]
            return (fine_tuning_job_id, status)

    def _validate_api_key(self):
        fine_tune_url = urljoin(self._base_url, "models")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(fine_tune_url, headers=headers)
        if response.status_code == 200:
            return True
        return False

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        fine_tune_url = urljoin(self._base_url, f"fine-tunes/{fine_tuning_job_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["job"]['status']
            fine_tuned_model = None
            if status.lower() == "succeeded":
                fine_tuned_model = r.json()["job"]["fine_tuned_model"]
            return (fine_tuned_model, status)

    def set_deploy_parameters(self, **kw):
        self._deploy_paramters = kw

    def _create_deployment(self) -> Tuple[str, str]:
        url = urljoin(self._base_url, "fine-tune/servings")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = {
            "model": self._model_name,
            "config": {
                "run_time": 0
            }
        }
        if self._deploy_paramters and len(self._deploy_paramters) > 0:
            data.update(self._deploy_paramters)

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()["job"]["id"]
            status = r.json()["job"]["status"]
            return (fine_tuning_job_id, status)

    def _query_deployment(self, deployment_id) -> str:
        fine_tune_url = urljoin(self._base_url, f"fine-tune/servings/{deployment_id}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            status = r.json()["job"]['status']
            return status

    def _format_vl_chat_image_url(self, image_url, mime):
        if image_url.startswith("http"):
            return [{"type": "image_url", "image_url": image_url}]
        else:
            return [{"type": "image_base64", "image_base64": image_url}]


class SenseNovaEmbedding(OnlineEmbeddingModuleBase, _SenseNovaBase):

    def __init__(self,
                 embed_url: str = "https://api.sensenova.cn/v1/llm/embeddings",
                 embed_model_name: str = "nova-embedding-stable",
                 api_key: str = None,
                 secret_key: str = None):
        api_key = self._get_api_key(api_key, secret_key)
        super().__init__("SENSENOVA", embed_url, api_key, embed_model_name)

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": [text],
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return response['embeddings'][0]['embedding']
