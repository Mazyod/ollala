import pathlib
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
    # TODO: define options type
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
    path: str | pathlib.Path
    stream: bool = False


class CreateModelResponse(BaseModel):
    status: str


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
    source: str
    destination: str


class DeleteModelRequest(BaseModel):
    name: str


class PullModelRequest(BaseModel):
    name: str
    insecure: bool = False
    stream: bool = False


class PullModelResponse(BaseModel):
    status: str


class PushModelRequest(BaseModel):
    name: str
    insecure: bool = False
    stream: bool = False


class PushModelResponse(BaseModel):
    status: str


class GenerateEmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: dict = None


class GenerateEmbeddingsResponse(BaseModel):
    embedding: list[float]


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def generate_completion(
        self, request_data: GenerateCompletionRequest
    ) -> GenerateCompletionResponse:
        url = f"{self.base_url}/api/generate"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        return GenerateCompletionResponse(**response.json())

    def list_models(self) -> ListModelsResponse:
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return ListModelsResponse(**response.json())

    def model_info(self, request_data: ModelInfoRequest) -> ModelInfoResponse:
        url = f"{self.base_url}/api/show"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        return ModelInfoResponse(**response.json())

    def create_model(self, request_data: CreateModelRequest) -> CreateModelResponse:
        url = f"{self.base_url}/api/create"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        return CreateModelResponse(**response.json())

    def copy_model(self, request_data: CopyModelRequest):
        url = f"{self.base_url}/api/copy"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        # TODO: proper error handling
        response.raise_for_status()

    def delete_model(self, request_data: DeleteModelRequest):
        url = f"{self.base_url}/api/delete"
        response = requests.delete(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        # TODO: proper error handling
        response.raise_for_status()

    def pull_model(self, request_data: PullModelRequest) -> PullModelResponse:
        url = f"{self.base_url}/api/pull"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        return PullModelResponse(**response.json())

    def push_model(self, request_data: PushModelRequest) -> PushModelResponse:
        url = f"{self.base_url}/api/push"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        return PushModelResponse(**response.json())

    def generate_embeddings(
        self, request_data: GenerateEmbeddingsRequest
    ) -> GenerateEmbeddingsResponse:
        url = f"{self.base_url}/api/embeddings"
        response = requests.post(
            url, data=request_data.model_dump_json(exclude_none=True)
        )
        return GenerateEmbeddingsResponse(**response.json())
