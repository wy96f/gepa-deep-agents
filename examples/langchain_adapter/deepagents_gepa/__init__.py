"""Small Deep Agents-specific framework pieces used by the GEPA example."""

from examples.langchain_adapter.deepagents_gepa.artifacts import RunArtifactCallback, RunArtifactStore
from examples.langchain_adapter.deepagents_gepa.framework import (
    CandidateMaterializer,
    ComponentSelector,
    Constraint,
    DatasetProvider,
    DefaultCandidateMaterializer,
    DefaultConstraintSet,
    DefaultDatasetProvider,
    DefaultEvaluator,
    DefaultFeedbackComponentSelector,
    DefaultReflectionTemplateRegistry,
    Evaluator,
    ReflectionTemplateRegistry,
    select_deployment_candidate_index,
)

__all__ = [
    "CandidateMaterializer",
    "ComponentSelector",
    "Constraint",
    "DatasetProvider",
    "DefaultCandidateMaterializer",
    "DefaultConstraintSet",
    "DefaultDatasetProvider",
    "DefaultEvaluator",
    "DefaultFeedbackComponentSelector",
    "DefaultReflectionTemplateRegistry",
    "Evaluator",
    "ReflectionTemplateRegistry",
    "RunArtifactCallback",
    "RunArtifactStore",
    "select_deployment_candidate_index",
]
