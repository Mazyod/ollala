import requests
import pydantic
import datetime as dt


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        protected_namespaces=(),
        extra="allow",
    )


class GenerateCompletionRequest(BaseModel):
    model: str
    prompt: str
    format: str = "json"
    options: dict = None
    system: str = None
    template: str = None
    context: list = None
    stream: bool = None
    raw: bool = None


class GenerateCompletionResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    total_duration: int = None
    load_duration: int = None
    sample_count: int = None
    sample_duration: int = None
    prompt_eval_count: int = None
    prompt_eval_duration: int = None
    eval_count: int = None
    eval_duration: int = None
    context: list = None


class CreateModelRequest(BaseModel):
    name: str
    description: str


class CreateModelResponse(BaseModel):
    success: bool
    message: str


class ModelInfo(BaseModel):
    name: str
    modified_at: dt.datetime
    size: int
    digest: str


class ListModelsResponse(BaseModel):
    models: list[ModelInfo]


class ModelInfoRequest(BaseModel):
    model_id: str


class ModelInfoResponse(BaseModel):
    pass


class CopyModelRequest(BaseModel):
    source_model_id: str
    new_model_name: str


class CopyModelResponse(BaseModel):
    success: bool
    message: str


class DeleteModelRequest(BaseModel):
    model_id: str


class DeleteModelResponse(BaseModel):
    success: bool
    message: str


class PullModelRequest(BaseModel):
    model_id: str


class PullModelResponse(BaseModel):
    success: bool
    message: str


class PushModelRequest(BaseModel):
    model_id: str


class PushModelResponse(BaseModel):
    success: bool
    message: str


class GenerateEmbeddingsRequest(BaseModel):
    model: str
    inputs: list


class GenerateEmbeddingsResponse(BaseModel):
    embeddings: list


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def generate_completion(
        self, request_data: GenerateCompletionRequest
    ) -> GenerateCompletionResponse:
        url = f"{self.base_url}/api/generate"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return GenerateCompletionResponse(**response.json())

    def create_model(self, request_data: CreateModelRequest) -> CreateModelResponse:
        url = f"{self.base_url}/api/model/create"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return CreateModelResponse(**response.json())

    def list_models(self) -> ListModelsResponse:
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return ListModelsResponse(**response.json())

    def model_info(self, request_data: ModelInfoRequest) -> ModelInfoResponse:
        url = f"{self.base_url}/api/model/info"
        response = requests.get(
            url, params=request_data.model_dump(exclude_defaults=True)
        )
        return ModelInfoResponse(**response.json())

    def copy_model(self, request_data: CopyModelRequest) -> CopyModelResponse:
        url = f"{self.base_url}/api/model/copy"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return CopyModelResponse(**response.json())

    def delete_model(self, request_data: DeleteModelRequest) -> DeleteModelResponse:
        url = f"{self.base_url}/api/model/delete"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return DeleteModelResponse(**response.json())

    def pull_model(self, request_data: PullModelRequest) -> PullModelResponse:
        url = f"{self.base_url}/api/model/pull"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return PullModelResponse(**response.json())

    def push_model(self, request_data: PushModelRequest) -> PushModelResponse:
        url = f"{self.base_url}/api/model/push"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return PushModelResponse(**response.json())

    def generate_embeddings(
        self, request_data: GenerateEmbeddingsRequest
    ) -> GenerateEmbeddingsResponse:
        url = f"{self.base_url}/api/embeddings/generate"
        response = requests.post(
            url, json=request_data.model_dump(exclude_defaults=True)
        )
        return GenerateEmbeddingsResponse(**response.json())
