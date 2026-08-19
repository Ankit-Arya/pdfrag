# ruff: noqa: E501
import sys
import types
from types import SimpleNamespace

# This hotfix ZIP is an overlay, so focused unit tests stub baseline modules when
# the complete repository is not present in the packaging workspace.
if "app.config" not in sys.modules:
    config_stub = types.ModuleType("app.config")
    config_stub.get_settings = lambda: SimpleNamespace(query_model="test-query-model", query_reasoning_effort="low")
    sys.modules["app.config"] = config_stub

if "app.rag.llm" not in sys.modules:
    llm_stub = types.ModuleType("app.rag.llm")

    class LlmConfigurationError(RuntimeError):
        pass

    class FakeLlm:
        def generate(self, *args, **kwargs):
            raise LlmConfigurationError("not configured")

    llm_stub.LlmConfigurationError = LlmConfigurationError
    llm_stub.llm_service = FakeLlm()
    sys.modules["app.rag.llm"] = llm_stub

from app.rag.smart_understanding import (
    SmartInterpretation,
    _parse_interpretation,
    relevant_terminology_hints,
    should_review_evidence,
)


def test_ai_interpretation_can_rewrite_noisy_language_without_changing_facts() -> None:
    parsed = _parse_interpretation(
        "who responsble if doors dont close at platfrm",
        {
            "resolved_question": "Who is responsible for the required action when train doors do not close at a platform?",
            "intent": "requirement",
            "conversation_act": "question",
            "concepts": ["train doors not closed", "platform", "responsibility"],
            "evidence_needs": [
                "rule governing a train with doors not closed",
                "role responsible for the required action",
            ],
            "search_queries": [
                "train doors not closed platform responsibility procedure",
                "door failure required action station train operator controller",
            ],
            "scope": {"location": "platform"},
            "material_ambiguity": False,
            "ambiguity_note": "",
            "uses_history": False,
            "authority_sensitive": False,
            "route_strategy": "dedicated_procedure",
            "corrections": ["responsble -> responsible", "platfrm -> platform"],
        },
    )
    assert parsed.intent == "requirement"
    assert parsed.resolved_question.startswith("Who is responsible")
    assert "responsibility" in parsed.concepts
    assert len(parsed.evidence_needs) == 2
    assert parsed.route_strategy == "dedicated_procedure"


def test_interpretation_searches_for_evidence_requirements_not_just_paraphrases() -> None:
    parsed = _parse_interpretation(
        "can train go if indication missing",
        {
            "resolved_question": "May a train proceed when the required indication is missing?",
            "intent": "requirement",
            "conversation_act": "question",
            "concepts": ["train movement", "missing indication", "permission to proceed"],
            "evidence_needs": [
                "rule for the missing indication",
                "condition under which movement is permitted or prohibited",
                "required authorization or responsible role",
            ],
            "search_queries": [
                "missing indication train proceed permission",
                "signal indication absent authorization movement rule",
            ],
            "scope": {},
            "material_ambiguity": True,
            "ambiguity_note": "The type of indication may change the applicable rule.",
            "uses_history": False,
            "authority_sensitive": False,
            "route_strategy": "authoritative_rule",
            "corrections": [],
        },
    )
    assert parsed.material_ambiguity
    assert len(parsed.evidence_needs) == 3
    assert should_review_evidence(parsed)


def test_multi_turn_followup_is_explicitly_controlled_by_ai() -> None:
    parsed = _parse_interpretation(
        "who does it then",
        {
            "resolved_question": "Who performs the verification step in the previously discussed evacuation procedure?",
            "intent": "requirement",
            "conversation_act": "question",
            "concepts": ["evacuation", "verification", "responsible role"],
            "evidence_needs": ["explicit role assignment for the verification step"],
            "search_queries": ["evacuation verification responsible staff role"],
            "scope": {},
            "material_ambiguity": False,
            "ambiguity_note": "",
            "uses_history": True,
            "authority_sensitive": False,
            "route_strategy": "dedicated_procedure",
            "corrections": [],
        },
    )
    assert parsed.uses_history is True
    assert parsed.route_strategy == "dedicated_procedure"


def test_long_user_supplied_evidence_is_not_forced_into_definition_mode() -> None:
    parsed = _parse_interpretation(
        "The controller shall verify the route and staff shall report completion ...",
        {
            "resolved_question": "Verify the supplied operational statement against the knowledge base and summarize what it establishes.",
            "intent": "summary",
            "conversation_act": "evidence_correction",
            "concepts": ["route verification", "completion report"],
            "evidence_needs": ["source text confirming or contradicting the supplied statement"],
            "search_queries": ["route verification completion report controller staff"],
            "scope": {},
            "material_ambiguity": False,
            "ambiguity_note": "",
            "uses_history": True,
            "authority_sensitive": False,
            "route_strategy": "authoritative_rule",
            "corrections": [],
        },
    )
    assert parsed.conversation_act == "evidence_correction"
    assert parsed.intent == "summary"


def test_only_relevant_document_grounded_terminology_hints_survive() -> None:
    interpretation = SmartInterpretation(
        raw_question="What should ABC do after a brake fault?",
        resolved_question="What should ABC do after a brake fault?",
        concepts=("ABC", "brake fault"),
    )
    hints = [
        "ABC = Authorized Brake Controller — source.pdf p.2",
        "XYZ = Unrelated System — other.pdf p.9",
    ]
    assert relevant_terminology_hints(interpretation, hints) == [hints[0]]


def test_simple_definition_skips_expensive_evidence_critic() -> None:
    interpretation = SmartInterpretation(
        raw_question="What does ABC mean?",
        resolved_question="What does ABC mean?",
        intent="definition",
        concepts=("ABC",),
        evidence_needs=("explicit source definition of ABC",),
    )
    assert not should_review_evidence(interpretation)


def test_current_rule_lookup_requests_evidence_review() -> None:
    interpretation = SmartInterpretation(
        raw_question="What is the current limit?",
        resolved_question="What is the current permitted limit for the described condition?",
        intent="fact_lookup",
        concepts=("current limit",),
        evidence_needs=("governing current limit",),
        authority_sensitive=True,
        route_strategy="authoritative_rule",
    )
    assert should_review_evidence(interpretation)


def test_interpret_user_message_uses_single_ai_contract(monkeypatch) -> None:
    import app.rag.smart_understanding as module

    payload = '''{
      "resolved_question": "Who authorizes movement after the stated equipment failure?",
      "intent": "requirement",
      "conversation_act": "question",
      "concepts": ["equipment failure", "movement authorization", "responsible role"],
      "evidence_needs": ["applicable failure rule", "explicit authorization role"],
      "search_queries": ["equipment failure movement authorization role", "failure proceed permission controller"],
      "scope": {},
      "material_ambiguity": false,
      "ambiguity_note": "",
      "uses_history": false,
      "authority_sensitive": false,
      "route_strategy": "dedicated_procedure",
      "corrections": []
    }'''
    monkeypatch.setattr(module.llm_service, "generate", lambda *args, **kwargs: payload)
    interpretation = module.interpret_user_message("who authrise movment after eqpt fail")
    assert interpretation.ai_used
    assert interpretation.intent == "requirement"
    assert interpretation.resolved_question == "Who authorizes movement after the stated equipment failure?"
    assert interpretation.search_queries[0] == interpretation.resolved_question
