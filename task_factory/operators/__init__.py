"""Registered semantic task-evolution operators."""

from .base import EvolutionOperator, EvolutionProduct
from .policy_freshness import PolicyFreshnessOperator
from .capacity_reservation import CapacityReservationOperator
from .audit_checkpoint import AuditCheckpointOperator
from .execution_route import ExecutionRouteOperator
from .async_readiness import AsyncReadinessOperator
from .semantic_recovery import SemanticRecoveryOperator


OPERATORS: dict[str, EvolutionOperator] = {
    PolicyFreshnessOperator.operator_id: PolicyFreshnessOperator(),
    CapacityReservationOperator.operator_id: CapacityReservationOperator(),
    AuditCheckpointOperator.operator_id: AuditCheckpointOperator(),
    ExecutionRouteOperator.operator_id: ExecutionRouteOperator(),
    AsyncReadinessOperator.operator_id: AsyncReadinessOperator(),
    SemanticRecoveryOperator.operator_id: SemanticRecoveryOperator(),
}


def get_operator(operator_id: str) -> EvolutionOperator:
    try:
        return OPERATORS[operator_id]
    except KeyError as exc:
        raise ValueError(f"unknown evolution operator {operator_id!r}") from exc


__all__ = ["EvolutionOperator", "EvolutionProduct", "OPERATORS", "get_operator"]
