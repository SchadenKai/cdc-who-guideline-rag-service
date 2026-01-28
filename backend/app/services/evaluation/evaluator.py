from deepeval import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseModel, GPTModel
from deepeval.test_case import LLMTestCase

from app.core.config import Settings


class EvaluationPipeline:
    def __init__(
        self,
        settings: Settings,
    ):
        self.settings: Settings = settings
        self._answer_relevancy: AnswerRelevancyMetric | None = None
        self._faithfulness: FaithfulnessMetric | None = None
        self._contextual_relevance: ContextualRelevancyMetric | None = None
        self._contextual_recall: ContextualRecallMetric | None = None
        self._deepeval_model: DeepEvalBaseModel | None = None

    @property
    def deepeval_model(self) -> DeepEvalBaseModel:
        if self._deepeval_model:
            return self._deepeval_model
        self._deepeval_model = GPTModel(
            model=self.settings.llm_model_name,
            api_key=self.settings.llm_api_key,
            base_url="https://api.studio.nebius.ai/v1/",
        )
        return self._deepeval_model

    @property
    def metrics(
        self,
    ) -> tuple[
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        ContextualRecallMetric,
    ]:
        if self._answer_relevancy is None:
            self._answer_relevancy = AnswerRelevancyMetric(
                model=self.deepeval_model, threshold=self.settings.rag_metrics_threshold
            )
        if self._faithfulness is None:
            self._faithfulness = FaithfulnessMetric(
                model=self.deepeval_model, threshold=self.settings.rag_metrics_threshold
            )
        if self._contextual_relevance is None:
            self._contextual_relevance = ContextualRelevancyMetric(
                model=self.deepeval_model, threshold=self.settings.rag_metrics_threshold
            )
        if self._contextual_recall is None:
            self._contextual_recall = ContextualRecallMetric(
                model=self.deepeval_model, threshold=self.settings.rag_metrics_threshold
            )
        return (
            self._answer_relevancy,
            self._faithfulness,
            self._contextual_relevance,
            self._contextual_recall,
        )

    def evaluate(self, test_cases: list[LLMTestCase]) -> EvaluationResult:
        answer_relevancy, faithfulness, contextual_relevance, contextual_recall = (
            self.metrics
        )
        return evaluate(
            test_cases=test_cases,
            metrics=[
                answer_relevancy,
                faithfulness,
                contextual_relevance,
                contextual_recall,
            ],
        )
