"""Graph nodes for the Nomad Job Spec Agent."""

from src.nodes.analyze import analyze_codebase_node, create_analyze_node
from src.nodes.discover import (
    discover_dockerfiles_node,
    select_dockerfile_node,
    should_select_dockerfile,
    create_discover_node,
    create_select_node,
)
from src.nodes.generate import (
    generate_spec_node,
    regenerate_spec_with_fix,
    create_generate_node,
    create_fix_node,
)
from src.nodes.question import (
    generate_questions_node,
    collect_responses_node,
    create_question_node,
    create_collect_node,
)
from src.nodes.deploy import (
    deploy_node,
    verify_node,
    create_deploy_node,
    create_verify_node,
)

__all__ = [
    # Discover
    "discover_dockerfiles_node",
    "select_dockerfile_node",
    "should_select_dockerfile",
    "create_discover_node",
    "create_select_node",
    # Analyze
    "analyze_codebase_node",
    "create_analyze_node",
    # Generate
    "generate_spec_node",
    "regenerate_spec_with_fix",
    "create_generate_node",
    "create_fix_node",
    "generate_questions_node",
    "collect_responses_node",
    "create_question_node",
    "create_collect_node",
    "deploy_node",
    "verify_node",
    "create_deploy_node",
    "create_verify_node",
]
