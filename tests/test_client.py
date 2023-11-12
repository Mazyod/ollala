import pathlib
import ollala
import pytest
import vcr


@pytest.fixture()
def client():
    return ollala.Client("http://localhost:11434")


class TestOllalaClient:
    def test_ollala_client_sanity(self, client: ollala.Client):
        assert client is not None
        assert client.base_url == "http://localhost:11434"

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    @pytest.mark.parametrize(
        "params",
        [
            dict(system="you are an annoyed old person who sighs at people"),
            dict(options=dict(temperature=0.5)),
        ],
    )
    def test_ollala_client_generate_completion(
        self, client: ollala.Client, params: dict
    ):
        request = ollala.GenerateCompletionRequest(
            model="orca-mini:3b", prompt="hello", stream=False, **params
        )
        response = client.generate_completion(request)

        assert response.model == "orca-mini:3b"
        assert response.response is not None

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_list_models(self, client: ollala.Client):
        response = client.list_models()

        assert response.models is not None
        assert len(response.models) > 0
        assert "orca-mini:3b" in [m.name for m in response.models]

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_model_info(self, client: ollala.Client):
        request = ollala.ModelInfoRequest(name="orca-mini:3b")
        response = client.model_info(request)

        assert response is not None

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_create_model(self, client: ollala.Client):
        path = pathlib.Path(__file__).parent.resolve() / ".sample-model" / "Modelfile"
        request = ollala.CreateModelRequest(name="test_model", path=path)
        response = client.create_model(request)

        assert response.status == "success"

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_copy_model(self, client: ollala.Client):
        request = ollala.CopyModelRequest(
            source="orca-mini:3b", destination="orca-mini:3b-copy"
        )
        # asserting it does not raise
        client.copy_model(request)

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_delete_model(self, client: ollala.Client):
        request = ollala.DeleteModelRequest(name="orca-mini:3b-copy")
        # asserting it does not raise
        client.delete_model(request)

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_pull_model(self, client: ollala.Client):
        request = ollala.PullModelRequest(name="orca-mini:3b")
        response = client.pull_model(request)

        assert response.status == "success"

    def test_ollala_client_push_model(self, client: ollala.Client):
        request = ollala.PushModelRequest(name="orca-mini:3b")
        response = client.push_model(request)

        assert response.status == "success"

    @vcr.use_cassette("tests/ollama-responses.yaml", record_mode=vcr.mode.NEW_EPISODES)
    def test_ollala_client_generate_embeddings(self, client: ollala.Client):
        request = ollala.GenerateEmbeddingsRequest(model="orca-mini:3b", prompt="hello")
        response = client.generate_embeddings(request)

        assert response.embedding is not None
        assert len(response.embedding) > 0
