from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    input_guardrail,
    output_guardrail
)
from typing import List, Union
from schema.schema import ParsedGoal


@input_guardrail
async def goal_input_guadrail(
    ctx: RunContextWrapper, agent: Agent, input: Union[str, List]
) -> GuardrailFunctionOutput:
    """
    Validates user input for:
    - Clear health/wellness goal or preference
    - Avoids unsafe or unrealistic expectations
    - Allows escalation, injury, and help intent to pass through safely
    """

    print("🛡️ Raw input received:", input)

    if isinstance(input, list):
        input_text = " ".join(str(item) for item in input)
    elif isinstance(input, str):
        input_text = input
    else:
        return GuardrailFunctionOutput(
            output_info="⚠️ Invalid input format. Please provide a clear text-based goal.",
            tripwire_triggered=True
        )

    input_text = input_text.lower()
    print("✅ Normalized input:", input_text)

    escalation_keywords = ["frustrated", "angry", "issue", "problem", "not working", "annoyed"]
    if any(word in input_text for word in escalation_keywords):
        print("⚠️ Escalation-related input detected.")
        return GuardrailFunctionOutput(
            output_info="Escalation message allowed",
            tripwire_triggered=False
        )

    injury_keywords = ["injury", "pain", "hurt", "sore", "can’t move", "sprain", "broken", "brain injury"]
    if any(word in input_text for word in injury_keywords):
        print("⚠️ Injury-related input detected.")
        return GuardrailFunctionOutput(
            output_info="Injury-related message allowed",
            tripwire_triggered=False
        )

    if "100kg" in input_text and ("days" in input_text or "week" in input_text):
        return GuardrailFunctionOutput(
            output_info="❌ Unrealistic goal. Please provide a more achievable target like 5-10kg in 2-3 months.",
            tripwire_triggered=True
        )

    valid_keywords = [
        "lose", "gain", "build", "improve", "track", "plan",
        "workout", "exercise", "meal", "diet", "schedule",
        "check-in", "weight", "muscle", "routine", "sleep"
    ]

    if not any(keyword in input_text for keyword in valid_keywords):
        return GuardrailFunctionOutput(
            output_info="❌ Input unclear. Please clearly state your health goal. For example: 'I want to lose 5kg in 2 months'.",
            tripwire_triggered=True
        )

    print("✅ Input passed all checks.")
    return GuardrailFunctionOutput(
        output_info="Input valid",
        tripwire_triggered=False
    )


@output_guardrail
async def goal_output_guardrail(ctx, agent, output) -> GuardrailFunctionOutput:
    """
    Ensures output is a valid ParsedGoal or a user-friendly message.
    Prevents LLM from returning invalid or unexpected formats.
    """

    print("🔍 Raw output type:", type(output))

    if isinstance(output, ParsedGoal):
        return GuardrailFunctionOutput(
            output_info="✅ Parsed goal is valid.",
            tripwire_triggered=False
        )

    if isinstance(output, str):
        print("✅ Detected user-facing message.")
        return GuardrailFunctionOutput(
            output_info="String output allowed (user message)",
            tripwire_triggered=False
        )

    return GuardrailFunctionOutput(
        output_info="❌ Output unrecognized or not allowed. Please return a valid ParsedGoal or assistant message.",
        tripwire_triggered=True
    )