import ollala
import pytest


@pytest.fixture()
def client():
    return ollala.Client("http://localhost:11434")


def test_ollala_client_sanity(client: ollala.Client):
    assert client is not None
    assert client.base_url == "http://localhost:11434"


@pytest.mark.parametrize(
    "params",
    [
        dict(system="you are an annoyed old person who sighs at people"),
        dict(options=dict(temperature=0.5)),
    ],
)
def test_ollala_client_generate_completion(client: ollala.Client, params: dict):
    request = ollala.GenerateCompletionRequest(
        model="orca-mini:3b", prompt="hello", stream=False, **params
    )
    response = client.generate_completion(request)

    assert response.model == "orca-mini:3b"
    assert response.response is not None


def test_ollala_client_list_models(client: ollala.Client):
    response = client.list_models()

    assert response.models is not None
    assert len(response.models) > 0
    assert "orca-mini:3b" in [m.name for m in response.models]
