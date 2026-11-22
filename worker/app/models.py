from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class WordResult(ApiModel):
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


class CheckResponse(ApiModel):
    duration_seconds: float
    elapsed_seconds: float
    transcript: str
    words: list[WordResult]
