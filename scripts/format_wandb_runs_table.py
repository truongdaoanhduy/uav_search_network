#!/usr/bin/env python3
"""Curate the actual personal W&B Runs table for UAV training runs."""

from __future__ import annotations

import argparse
import json

import wandb
from wandb_workspaces import expr
from wandb_workspaces.workspaces import internal

ENTITY = "uav_search_paper"
PROJECT = "uav_search_target"
DEFAULT_WORKSPACE = "Truongdaoanhduy's workspace"

DESIRED_COLUMNS = [
    "run:displayName",
    "run:state",
    "run:createdAt",
    "run:duration",
    "config:algorithm.value",
    "config:seed.value",
    "config:kaggle_account.value",
    "config:num_envs.value",
    "config:target_episodes.value",
    "config:network_backend.value",
    "summary:episodes_completed",
]


def _find_view_node(api, display_name: str):
    query = """
    query Views($entityName:String!,$projectName:String!){
      project(name:$projectName,entityName:$entityName){
        allViews(viewType:"project-view"){
          edges{node{id name displayName spec}}
        }
      }
    }
    """
    response = internal.execute_graphql(
        api,
        query,
        {"entityName": ENTITY, "projectName": PROJECT},
    )
    nodes = [edge["node"] for edge in response["project"]["allViews"]["edges"]]
    for node in nodes:
        if node["displayName"] == display_name:
            return node
    raise RuntimeError(f"workspace not found: {display_name}")


def format_runs_table(display_name: str = DEFAULT_WORKSPACE):
    api = wandb.Api()
    node = _find_view_node(api, display_name)
    spec = internal.WorkspaceViewspec.model_validate_json(node["spec"])
    runset = spec.section.run_sets[0]
    run_feed = runset.run_feed

    existing_columns = set(run_feed.column_order) | set(run_feed.column_visible) | set(run_feed.column_pinned)
    run_feed.column_order = list(DESIRED_COLUMNS)
    # W&B's frontend treats omitted visibility keys as auto-visible in some
    # personal workspaces. Explicitly turn every old column off, then enable
    # only the curated training columns.
    run_feed.column_visible = {column: False for column in existing_columns}
    run_feed.column_visible.update({column: True for column in DESIRED_COLUMNS})
    run_feed.column_pinned = {
        "run:displayName": True,
        "run:state": True,
    }
    run_feed.column_widths = {}
    run_feed.lock_columns = True
    run_feed.page_size = 20

    runset.filters = expr.filters_tree_to_v2(
        expr.expr_to_filters("Config('run_role') = 'training'")
    )

    view = internal.View(
        entity=ENTITY,
        project=PROJECT,
        display_name=node["displayName"],
        name=node["name"],
        id=node["id"],
        spec=spec,
    )
    internal.upsert_view2(view)
    return {
        "workspace": display_name,
        "columns": DESIRED_COLUMNS,
        "filter": "run_role=training",
        "lock_columns": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    args = parser.parse_args()
    print(json.dumps(format_runs_table(args.workspace), indent=2))


if __name__ == "__main__":
    main()
