#!/usr/bin/env python3
"""Build a WikiHow-derived closed-loop stress-test task."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_validation import validate_goal_alignment
from task_factory import load_task_bundle, totalize_public_capabilities
from task_factory.bundle import TaskBundle
from task_factory.materialize import materialize_candidate
from task_factory.operators.base import manifest_metadata


DEFAULT_SOURCE = PROJECT_ROOT / (
    "outputs/task_first/wikihow_compile200_context_v5/parents/"
    "wikihow_computer_002730__task_first"
)


def build_bundle(source: Path) -> TaskBundle:
    parent = load_task_bundle(source)
    instruction = parent.instruction
    metrics = "$state.system_metrics.performance_tab"
    job = "$state.jobs.stress_test_01"
    control = "$state.stress_control"
    context = "$state.workflow_context"

    initial_state = {
        "ui": {"task_manager_open": False, "task_manager_was_opened": False},
        "system_metrics": {
            "performance_tab": {
                "cpu_usage_percent": 12,
                "ram_usage_mb": 1024,
                "ram_total_mb": 4096,
                "status": "normal",
            }
        },
        "jobs": {
            "stress_test_01": {
                "status": "not_started",
                "load": "high",
                "target_id": "system_metrics",
            }
        },
        "stress_control": {
            "ramp_profile": "gradual",
            "phase": "idle",
            "baseline_observed": False,
            "baseline_handle": "",
            "latest_measurement": "",
            "peak_observed": False,
            "peak_cpu_percent": 0,
            "peak_ram_usage_mb": 0,
            "stop_evidence": "",
            "cancellation_handle": "",
            "settled": False,
            "overrun": False,
        },
        "workflow_context": {
            "handle": "workflow_context_wikihow_computer_002730",
            "source_task_id": "wikihow_computer_002730",
        },
    }

    capabilities = {
        "wikihow.workflow_context.observe.v1": {
            "branches": [
                {
                    "id": "workflow_context_visible",
                    "when": True,
                    "response": {
                        "workflow_context_handle": f"{context}.handle",
                        "target_system": "system_metrics",
                        "target_cpu_percent": 95,
                        "target_ram_usage_mb": 3900,
                    },
                    "effects": [],
                    "reads": [f"{context}.handle"],
                    "writes": [],
                }
            ]
        },
        "wikihow.open_resource.v1": {
            "branches": [
                {
                    "id": "task_manager_opened",
                    "when": {
                        "all": [
                            {"eq": ["$args.resource_type", "app"]},
                            {"eq": ["$args.target", "Task Manager"]},
                            {"eq": ["$state.ui.task_manager_open", False]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "message": "Task Manager opened with Performance visible.",
                    },
                    "effects": [
                        {"set": "$state.ui.task_manager_open", "value": True},
                        {"set": "$state.ui.task_manager_was_opened", "value": True},
                    ],
                    "reads": ["$state.ui.task_manager_open"],
                    "writes": [
                        "$state.ui.task_manager_open",
                        "$state.ui.task_manager_was_opened",
                    ],
                }
            ]
        },
        "wikihow.close_resource.v1": {
            "branches": [
                {
                    "id": "task_manager_closed_after_settle",
                    "when": {
                        "all": [
                            {"eq": ["$args.resource_type", "app"]},
                            {"eq": ["$args.target", "Task Manager"]},
                            {"eq": ["$state.ui.task_manager_open", True]},
                            {"eq": [f"{control}.phase", "settled"]},
                            {"eq": [f"{control}.settled", True]},
                        ]
                    },
                    "response": {"status": "success", "closed": "Task Manager"},
                    "effects": [
                        {"set": "$state.ui.task_manager_open", "value": False}
                    ],
                    "reads": [
                        "$state.ui.task_manager_open",
                        f"{control}.phase",
                        f"{control}.settled",
                    ],
                    "writes": ["$state.ui.task_manager_open"],
                }
            ]
        },
        "wikihow.read_state.v1": {
            "branches": [
                {
                    "id": "baseline_metrics",
                    "when": {
                        "all": [
                            {"eq": ["$args.view", "performance_metrics"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": [f"{control}.phase", "idle"]},
                            {"eq": ["$state.ui.task_manager_open", True]},
                        ]
                    },
                    "response": {
                        "measurement_handle": "measurement_baseline_12_1024",
                        "phase": "idle",
                        "ramp_profile": f"{control}.ramp_profile",
                        "metrics": metrics,
                    },
                    "effects": [
                        {"set": f"{control}.baseline_observed", "value": True},
                        {
                            "set": f"{control}.baseline_handle",
                            "value": "measurement_baseline_12_1024",
                        },
                        {
                            "set": f"{control}.latest_measurement",
                            "value": "measurement_baseline_12_1024",
                        },
                    ],
                    "reads": [
                        metrics,
                        f"{control}.phase",
                        f"{control}.ramp_profile",
                        "$state.ui.task_manager_open",
                    ],
                    "writes": [
                        f"{control}.baseline_observed",
                        f"{control}.baseline_handle",
                        f"{control}.latest_measurement",
                    ],
                },
                {
                    "id": "cpu_ramp_observed",
                    "when": {
                        "all": [
                            {"eq": ["$args.view", "performance_metrics"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": [f"{control}.phase", "ramp_cpu"]},
                        ]
                    },
                    "response": {
                        "measurement_handle": "measurement_ramp_cpu_72_2400",
                        "phase": "ramp_cpu",
                        "ramp_profile": f"{control}.ramp_profile",
                        "target_reached": False,
                        "metrics": metrics,
                    },
                    "effects": [
                        {
                            "set": f"{control}.latest_measurement",
                            "value": "measurement_ramp_cpu_72_2400",
                        }
                    ],
                    "reads": [
                        f"{control}.phase",
                        f"{control}.ramp_profile",
                        metrics,
                    ],
                    "writes": [f"{control}.latest_measurement"],
                },
                {
                    "id": "memory_ramp_observed",
                    "when": {
                        "all": [
                            {"eq": ["$args.view", "performance_metrics"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": [f"{control}.phase", "ramp_memory"]},
                        ]
                    },
                    "response": {
                        "measurement_handle": "measurement_ramp_memory_94_3200",
                        "phase": "ramp_memory",
                        "ramp_profile": f"{control}.ramp_profile",
                        "target_reached": False,
                        "metrics": metrics,
                    },
                    "effects": [
                        {
                            "set": f"{control}.latest_measurement",
                            "value": "measurement_ramp_memory_94_3200",
                        }
                    ],
                    "reads": [f"{control}.phase", f"{control}.ramp_profile", metrics],
                    "writes": [f"{control}.latest_measurement"],
                },
                {
                    "id": "target_load_observed",
                    "when": {
                        "all": [
                            {"eq": ["$args.view", "performance_metrics"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": [f"{control}.phase", "target_load"]},
                        ]
                    },
                    "response": {
                        "measurement_handle": "measurement_peak_98_4096",
                        "phase": "target_load",
                        "ramp_profile": f"{control}.ramp_profile",
                        "target_reached": True,
                        "metrics": metrics,
                    },
                    "effects": [
                        {"set": f"{control}.peak_observed", "value": True},
                        {"set": f"{control}.peak_cpu_percent", "value": 98},
                        {"set": f"{control}.peak_ram_usage_mb", "value": 4096},
                        {
                            "set": f"{control}.latest_measurement",
                            "value": "measurement_peak_98_4096",
                        }
                    ],
                    "reads": [
                        f"{control}.phase",
                        f"{control}.ramp_profile",
                        metrics,
                    ],
                    "writes": [
                        f"{control}.peak_observed",
                        f"{control}.peak_cpu_percent",
                        f"{control}.peak_ram_usage_mb",
                        f"{control}.latest_measurement",
                    ],
                },
                {
                    "id": "settled_metrics",
                    "when": {
                        "all": [
                            {"eq": ["$args.view", "performance_metrics"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": [f"{control}.phase", "settled"]},
                        ]
                    },
                    "response": {
                        "measurement_handle": "measurement_settled_5_1200",
                        "phase": "settled",
                        "metrics": {
                            "cpu_usage_percent": 5,
                            "ram_usage_mb": 1200,
                            "ram_total_mb": 4096,
                            "status": "normal",
                        },
                    },
                    "effects": [{"set": f"{control}.settled", "value": True}],
                    "reads": [f"{control}.phase", metrics],
                    "writes": [f"{control}.settled"],
                },
            ]
        },
        "wikihow.start_job.v1": {
            "branches": [
                {
                    "id": "stress_started_from_baseline",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_type", "stress_test"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": ["$args.options", {}]},
                            {
                                "eq": [
                                    "$args.baseline_handle",
                                    f"{control}.baseline_handle",
                                ]
                            },
                            {"eq": [f"{control}.baseline_observed", True]},
                            {"eq": [f"{control}.phase", "idle"]},
                            {"eq": [f"{control}.ramp_profile", "gradual"]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "job_id": "stress_test_01",
                        "phase": "ramp_cpu",
                    },
                    "effects": [
                        {"set": f"{job}.status", "value": "running"},
                        {
                            "set": f"{control}.phase",
                            "value": "ramp_cpu",
                        },
                        {
                            "set": metrics,
                            "value": {
                                "cpu_usage_percent": 72,
                                "ram_usage_mb": 2400,
                                "ram_total_mb": 4096,
                                "status": "under_load",
                            },
                        },
                    ],
                    "reads": [
                        f"{control}.baseline_observed",
                        f"{control}.baseline_handle",
                        f"{control}.phase",
                        f"{control}.ramp_profile",
                    ],
                    "writes": [f"{job}.status", f"{control}.phase", metrics],
                },
                {
                    "id": "stress_started_with_immediate_saturation",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_type", "stress_test"]},
                            {"eq": ["$args.target_id", "system_metrics"]},
                            {"eq": ["$args.options", {}]},
                            {
                                "eq": [
                                    "$args.baseline_handle",
                                    f"{control}.baseline_handle",
                                ]
                            },
                            {"eq": [f"{control}.baseline_observed", True]},
                            {"eq": [f"{control}.phase", "idle"]},
                            {"eq": [f"{control}.ramp_profile", "immediate"]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "job_id": "stress_test_01",
                        "phase": "target_load",
                    },
                    "effects": [
                        {"set": f"{job}.status", "value": "running"},
                        {"set": f"{control}.phase", "value": "target_load"},
                        {
                            "set": metrics,
                            "value": {
                                "cpu_usage_percent": 98,
                                "ram_usage_mb": 4096,
                                "ram_total_mb": 4096,
                                "status": "maxed_out",
                            },
                        },
                    ],
                    "reads": [
                        f"{control}.baseline_observed",
                        f"{control}.baseline_handle",
                        f"{control}.phase",
                        f"{control}.ramp_profile",
                    ],
                    "writes": [f"{job}.status", f"{control}.phase", metrics],
                },
            ]
        },
        "wikihow.poll_job.v1": {
            "branches": [
                {
                    "id": "continue_after_cpu_sample",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_id", "stress_test_01"]},
                            {"eq": [f"{job}.status", "running"]},
                            {"eq": [f"{control}.phase", "ramp_cpu"]},
                            {
                                "eq": [
                                    "$args.observation_handle",
                                    "measurement_ramp_cpu_72_2400",
                                ]
                            },
                            {
                                "eq": [
                                    "$args.observation_handle",
                                    f"{control}.latest_measurement",
                                ]
                            },
                        ]
                    },
                    "response": {
                        "job_id": "stress_test_01",
                        "job_status": "running",
                        "continued": True,
                        "new_phase": "ramp_memory",
                    },
                    "effects": [
                        {"set": f"{control}.phase", "value": "ramp_memory"},
                        {
                            "set": metrics,
                            "value": {
                                "cpu_usage_percent": 94,
                                "ram_usage_mb": 3200,
                                "ram_total_mb": 4096,
                                "status": "under_load",
                            },
                        },
                    ],
                    "reads": [job, f"{control}.phase", f"{control}.latest_measurement"],
                    "writes": [f"{control}.phase", metrics],
                },
                {
                    "id": "continue_after_memory_sample",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_id", "stress_test_01"]},
                            {"eq": [f"{job}.status", "running"]},
                            {"eq": [f"{control}.phase", "ramp_memory"]},
                            {
                                "eq": [
                                    "$args.observation_handle",
                                    "measurement_ramp_memory_94_3200",
                                ]
                            },
                            {
                                "eq": [
                                    "$args.observation_handle",
                                    f"{control}.latest_measurement",
                                ]
                            },
                        ]
                    },
                    "response": {
                        "job_id": "stress_test_01",
                        "job_status": "running",
                        "continued": True,
                        "new_phase": "target_load",
                    },
                    "effects": [
                        {"set": f"{control}.phase", "value": "target_load"},
                        {
                            "set": metrics,
                            "value": {
                                "cpu_usage_percent": 98,
                                "ram_usage_mb": 4096,
                                "ram_total_mb": 4096,
                                "status": "maxed_out",
                            },
                        },
                    ],
                    "reads": [job, f"{control}.phase", f"{control}.latest_measurement"],
                    "writes": [f"{control}.phase", metrics],
                },
                {
                    "id": "unsafe_overrun_after_target",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_id", "stress_test_01"]},
                            {"eq": [f"{job}.status", "running"]},
                            {"eq": [f"{control}.phase", "target_load"]},
                        ]
                    },
                    "response": {
                        "ok": False,
                        "error_code": "STRESS_TARGET_OVERRUN",
                        "emergency_stop_required": True,
                    },
                    "effects": [
                        {"set": f"{control}.phase", "value": "overrun"},
                        {"set": f"{control}.overrun", "value": True},
                    ],
                    "reads": [job, f"{control}.phase"],
                    "writes": [f"{control}.phase", f"{control}.overrun"],
                },
                {
                    "id": "cooling_settle_observed",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_id", "stress_test_01"]},
                            {"eq": [f"{job}.status", "cancelling"]},
                            {"eq": [f"{control}.phase", "cooling"]},
                            {
                                "eq": [
                                    "$args.observation_handle",
                                    f"{control}.cancellation_handle",
                                ]
                            },
                        ]
                    },
                    "response": {
                        "job_id": "stress_test_01",
                        "job_status": "cooling",
                        "metrics": metrics,
                    },
                    "effects": [],
                    "after_response_effects": [
                        {"set": f"{job}.status", "value": "cancelled"},
                        {"set": f"{control}.phase", "value": "settled"},
                        {
                            "set": metrics,
                            "value": {
                                "cpu_usage_percent": 5,
                                "ram_usage_mb": 1200,
                                "ram_total_mb": 4096,
                                "status": "normal",
                            },
                        },
                    ],
                    "reads": [job, f"{control}.phase", f"{control}.cancellation_handle", metrics],
                    "writes": [f"{job}.status", f"{control}.phase", metrics],
                },
            ]
        },
        "wikihow.cancel_job.v1": {
            "branches": [
                {
                    "id": "stop_after_observed_target",
                    "when": {
                        "all": [
                            {"eq": ["$args.job_id", "stress_test_01"]},
                            {
                                "eq": [
                                    "$args.measurement_handle",
                                    "measurement_peak_98_4096",
                                ]
                            },
                            {"eq": [f"{control}.phase", "target_load"]},
                            {"eq": [f"{control}.peak_observed", True]},
                        ]
                    },
                    "response": {
                        "status": "success",
                        "job_id": "stress_test_01",
                        "stopped_from_measurement": "$args.measurement_handle",
                        "cancellation_handle": "cancellation_stress_test_01",
                    },
                    "effects": [
                        {"set": f"{job}.status", "value": "cancelling"},
                        {
                            "set": f"{control}.stop_evidence",
                            "value": "$args.measurement_handle",
                        },
                        {
                            "set": f"{control}.cancellation_handle",
                            "value": "cancellation_stress_test_01",
                        },
                        {"set": f"{control}.phase", "value": "cooling"},
                        {
                            "set": metrics,
                            "value": {
                                "cpu_usage_percent": 42,
                                "ram_usage_mb": 2200,
                                "ram_total_mb": 4096,
                                "status": "cooling",
                            },
                        },
                    ],
                    "reads": [
                        f"{control}.phase",
                        f"{control}.peak_observed",
                    ],
                    "writes": [
                        f"{job}.status",
                        f"{control}.stop_evidence",
                        f"{control}.cancellation_handle",
                        f"{control}.phase",
                        metrics,
                    ],
                }
            ]
        },
        "wikihow.workflow_outcome.observe.v1": {
            "branches": [
                {
                    "id": "settled_workflow_outcome",
                    "when": {
                        "all": [
                            {
                                "eq": [
                                    "$args.workflow_context_handle",
                                    f"{context}.handle",
                                ]
                            },
                            {"eq": [f"{control}.settled", True]},
                        ]
                    },
                    "response": {
                        "outcome_state": {
                            "ui": "$state.ui",
                            "jobs": "$state.jobs",
                            "system_metrics": "$state.system_metrics",
                            "stress_control": "$state.stress_control",
                            "workflow_context": context,
                        }
                    },
                    "effects": [],
                    "reads": [
                        "$state.ui",
                        "$state.jobs",
                        "$state.system_metrics",
                        "$state.stress_control",
                        context,
                    ],
                    "writes": [],
                }
            ]
        },
    }

    bindings = copy.deepcopy(parent.bindings)
    cancel = next(tool for tool in bindings["tools"] if tool["name"] == "cancel_job")
    cancel["parameters"]["properties"]["measurement_ref"] = {
        "type": "string",
        "description": "Observed performance measurement justifying the stop decision.",
    }
    cancel["parameters"]["required"].append("measurement_ref")
    cancel.setdefault("input_map", {})["measurement_ref"] = "measurement_handle"
    cancel.setdefault("provenance_required", []).append("measurement_ref")
    read = next(tool for tool in bindings["tools"] if tool["name"] == "read_state")
    read["parameters"]["properties"]["target_id"]["default"] = "system_metrics"
    read["parameters"]["required"] = ["view"]
    start = next(tool for tool in bindings["tools"] if tool["name"] == "start_job")
    start["parameters"]["properties"]["baseline_ref"] = {
        "type": "string",
        "description": "Baseline measurement returned by read_state before starting load.",
    }
    start["parameters"]["required"] = ["job_type", "target_id", "baseline_ref"]
    start.setdefault("input_map", {})["baseline_ref"] = "baseline_handle"
    start.setdefault("provenance_required", []).append("baseline_ref")
    poll = next(tool for tool in bindings["tools"] if tool["name"] == "poll_job")
    poll["parameters"]["properties"]["observation_ref"] = {
        "type": "string",
        "description": (
            "Latest measurement or cancellation acknowledgement that justifies "
            "advancing this job."
        ),
    }
    poll["parameters"]["required"].append("observation_ref")
    poll.setdefault("input_map", {})["observation_ref"] = "observation_handle"
    poll.setdefault("provenance_required", []).append("observation_ref")
    outcome_index = next(
        index
        for index, tool in enumerate(bindings["tools"])
        if tool["name"] == "observe_workflow_outcome"
    )
    bindings["tools"].insert(
        outcome_index,
        {
            "name": "close_resource",
            "description": (
                "Close an open application only after the workflow state permits "
                "it to be closed safely."
            ),
            "capability_id": "wikihow.close_resource.v1",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "description": "Kind of resource to close.",
                        "enum": ["app"],
                    },
                    "target": {
                        "type": "string",
                        "description": "Visible name of the open resource.",
                        "enum": ["Task Manager"],
                    },
                },
                "required": ["resource_type", "target"],
                "additionalProperties": False,
            },
            "provenance_required": [],
        },
    )

    goal = {
        "all": [
            {"eq": [f"{control}.baseline_observed", True]},
            {"eq": [f"{control}.peak_observed", True]},
            {"gte": [f"{control}.peak_cpu_percent", 95]},
            {"eq": [f"{control}.peak_ram_usage_mb", 4096]},
            {"eq": [f"{job}.status", "cancelled"]},
            {"eq": [f"{control}.settled", True]},
            {"eq": [f"{control}.overrun", False]},
            {"eq": [f"{metrics}.cpu_usage_percent", 5]},
            {"eq": [f"{metrics}.ram_usage_mb", 1200]},
            {"eq": [f"{metrics}.status", "normal"]},
            {"eq": ["$state.ui.task_manager_was_opened", True]},
            {"eq": ["$state.ui.task_manager_open", False]},
        ]
    }
    contract = copy.deepcopy(parent.contract)
    contract["goal_predicates"] = [{"id": "complete_stress_test", "predicate": goal}]
    contract["instruction_claims"] = [
        {
            "evidence_span": (
                "I want to stress test my computer to see if my CPU and RAM max out "
                "under heavy load."
            ),
            "kind": "goal",
            "clause_ids": ["heavy_load_observed"],
        },
        {
            "evidence_span": (
                "Please open the Task Manager, check my initial performance metrics, "
                "run a stress test on target_id 'system_metrics', check the metrics "
                "during the test, and then close everything to let the system settle."
            ),
            "kind": "goal",
            "clause_ids": [
                "task_manager_opened_and_closed",
                "baseline_checked",
                "stress_stopped_and_settled",
            ],
        },
    ]
    contract["goal_clauses"] = [
        {
            "id": "task_manager_opened_and_closed",
            "predicate": {
                "all": [
                    {"eq": ["$state.ui.task_manager_was_opened", True]},
                    {"eq": ["$state.ui.task_manager_open", False]},
                ]
            },
            "transition_paths": [
                "$state.ui.task_manager_was_opened",
                "$state.ui.task_manager_open",
            ],
            "evidence_paths": [
                "$state.ui.task_manager_was_opened",
                "$state.ui.task_manager_open",
            ],
            "witness_tools": ["open_resource", "close_resource"],
        },
        {
            "id": "baseline_checked",
            "predicate": {"eq": [f"{control}.baseline_observed", True]},
            "transition_paths": [f"{control}.baseline_observed"],
            "evidence_paths": [f"{control}.baseline_observed"],
            "witness_tools": ["read_state"],
        },
        {
            "id": "heavy_load_observed",
            "predicate": {
                "all": [
                    {"eq": [f"{control}.peak_observed", True]},
                    {"gte": [f"{control}.peak_cpu_percent", 95]},
                    {"eq": [f"{control}.peak_ram_usage_mb", 4096]},
                    {"eq": [f"{control}.overrun", False]},
                ]
            },
            "transition_paths": [
                f"{control}.peak_observed",
                f"{control}.peak_cpu_percent",
                f"{control}.peak_ram_usage_mb",
            ],
            "evidence_paths": [
                f"{control}.peak_observed",
                f"{control}.peak_cpu_percent",
                f"{control}.peak_ram_usage_mb",
                f"{control}.overrun",
            ],
            "witness_tools": ["read_state"],
        },
        {
            "id": "stress_stopped_and_settled",
            "predicate": {
                "all": [
                    {"eq": [f"{job}.status", "cancelled"]},
                    {"eq": [f"{control}.settled", True]},
                    {"eq": [f"{metrics}.cpu_usage_percent", 5]},
                    {"eq": [f"{metrics}.ram_usage_mb", 1200]},
                    {"eq": [f"{metrics}.status", "normal"]},
                ]
            },
            "transition_paths": [f"{job}.status", f"{control}.settled", metrics],
            "evidence_paths": [f"{job}.status", f"{control}.settled", metrics],
            "witness_tools": ["cancel_job", "read_state"],
        },
    ]
    contract["invariants"] = [
        {
            "id": "system_metrics_remain_bounded",
            "predicate": {
                "all": [
                    {"gte": [f"{metrics}.cpu_usage_percent", 0]},
                    {"lte": [f"{metrics}.cpu_usage_percent", 100]},
                    {"gte": [f"{metrics}.ram_usage_mb", 0]},
                    {
                        "lte": [
                            f"{metrics}.ram_usage_mb",
                            f"{metrics}.ram_total_mb",
                        ]
                    },
                ]
            },
        },
        {
            "id": "stop_evidence_matches_observed_peak",
            "predicate": {
                "any": [
                    {"eq": [f"{control}.stop_evidence", ""]},
                    {
                        "all": [
                            {"eq": [f"{control}.peak_observed", True]},
                            {
                                "eq": [
                                    f"{control}.stop_evidence",
                                    "measurement_peak_98_4096",
                                ]
                            },
                        ]
                    },
                ]
            },
        },
    ]
    contract["requirements"] = {
        "semantic_recovery": False,
        "async_decision": True,
        "goal_grounded_verification": True,
        "closed_loop_control": {
            "measurement_tool": "read_state",
            "control_tool": "cancel_job",
            "evidence_argument": "measurement_ref",
            "final_observation_tool": "observe_workflow_outcome",
            "measurement_paths": [
                metrics,
            ],
            "controlled_paths": [f"{job}.status", metrics],
            "settled_paths": [job, metrics, control, "$state.ui"],
        },
    }
    contract["forbidden_shortcuts"] = [
        "stop the stress job before observing target load",
        "continue the stress job after observing target load",
        "claim recovery without observing settled metrics",
    ]
    contract["expected_reasoning_features"] = [
        "delayed_handle_use",
        "observation_dependent_decision",
        "async_decision",
        "goal_grounded_verification",
        "alternative_plan",
        "closed_loop_control",
    ]
    contract["counterfactual_axes"] = [
        {
            "state_path": f"{control}.ramp_profile",
            "variants": ["gradual", "immediate"],
        }
    ]

    def action(tool: str, **arguments: object) -> dict:
        return {"tool": tool, "arguments": arguments}

    prefix = [
        action("observe_workflow_context"),
        action("open_resource", resource_type="app", target="Task Manager"),
        action("read_state", view="performance_metrics", target_id="system_metrics"),
        action(
            "start_job",
            job_type="stress_test",
            target_id="system_metrics",
            baseline_ref="measurement_baseline_12_1024",
        ),
    ]
    finish = [
        action(
            "cancel_job",
            job_id="stress_test_01",
            measurement_ref="measurement_peak_98_4096",
        ),
        action(
            "poll_job",
            job_id="stress_test_01",
            observation_ref="cancellation_stress_test_01",
        ),
        action("read_state", view="performance_metrics", target_id="system_metrics"),
        action("close_resource", resource_type="app", target="Task Manager"),
        action(
            "observe_workflow_outcome",
            workflow_context_handle="workflow_context_wikihow_computer_002730",
        ),
    ]
    gradual = prefix + [
        action("read_state", view="performance_metrics", target_id="system_metrics"),
        action(
            "poll_job",
            job_id="stress_test_01",
            observation_ref="measurement_ramp_cpu_72_2400",
        ),
        action("read_state", view="performance_metrics", target_id="system_metrics"),
        action(
            "poll_job",
            job_id="stress_test_01",
            observation_ref="measurement_ramp_memory_94_3200",
        ),
        action("read_state", view="performance_metrics", target_id="system_metrics"),
    ] + finish
    immediate = prefix + [
        action("read_state", view="performance_metrics", target_id="system_metrics"),
    ] + finish
    reference_plan = {
        "actions": gradual,
        "counterfactuals": [
            {
                "id": "immediate_ramp_requires_earlier_stop",
                "state_overrides": {f"{control}.ramp_profile": "immediate"},
                "actions": immediate,
            }
        ],
    }

    bundle = TaskBundle(
        root=parent.root,
        manifest={
            **copy.deepcopy(parent.manifest),
            "task_id": "wikihow_computer_002730__strict_closed_loop",
            "seed_family": "wikihow_strict_closed_loop_v1",
            "lineage": {
                "root_task_id": parent.task_id,
                "parent_task_id": parent.task_id,
                "generation": 1,
                "operators": ["closed_loop_stress_control_v1"],
            },
        },
        instruction=instruction,
        contract=contract,
        environment={
            "runtime_version": "causal-runtime-v1",
            "initial_state": initial_state,
            "capabilities": capabilities,
        },
        bindings=bindings,
        reference_plan=reference_plan,
    )
    return totalize_public_capabilities(bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = build_bundle(args.source)
    alignment = validate_goal_alignment(bundle)
    if not alignment["valid"]:
        raise SystemExit("goal alignment failed: " + "; ".join(alignment["errors"]))
    path = materialize_candidate(
        args.output_dir,
        task_id=bundle.task_id,
        contract=bundle.contract,
        candidate={
            "instruction": bundle.instruction,
            "environment": bundle.environment,
            "bindings": bundle.bindings,
            "reference_plan": bundle.reference_plan,
        },
        lineage=bundle.manifest["lineage"],
        manifest_metadata=manifest_metadata(bundle),
    )
    print(path)


if __name__ == "__main__":
    main()
