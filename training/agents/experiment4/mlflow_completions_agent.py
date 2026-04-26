import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams
from openai import OpenAI
import os

class MLflowCompletionsAgent(ChatModel):
    def load_context(self, context):
        # Build the client at load time, not __init__ — keeps the
        # logged artifact lightweight and lets serving inject env vars.
        self.client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def predict(
        self,
        context,
        messages: list[ChatMessage],
        params: ChatParams,
    ) -> ChatCompletionResponse:
        resp = self.client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=[m.to_dict() for m in messages],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )
        # ChatCompletionResponse mirrors the OpenAI response shape
        return ChatCompletionResponse.from_dict(resp.to_dict())
    

mlflow.openai.autolog()

AGENT = MLflowCompletionsAgent()
mlflow.models.set_model(AGENT)