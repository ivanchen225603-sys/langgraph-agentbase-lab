"""极简模板与有界消息装配；不可信数据永远不会被当作指令。"""

import string

from pydantic import JsonValue

from .configuration import AgentDefinition
from .context import ContextBundle, ContextProvider
from .contracts import AgentError, Message, ModelRequest
from .schema import SchemaContract, canonical_json


class PromptAssembler:
    @staticmethod
    def _render(template: str, variables: dict[str, str]) -> str:
        try:
            for _, name, specification, conversion in string.Formatter().parse(template):
                if name is not None and (
                    name not in variables or specification or conversion or not name.isidentifier()
                ):
                    raise AgentError("prompt_variable", "template contains an unsupported variable")
            return template.format_map(variables)
        except AgentError:
            raise
        except (ValueError, KeyError) as exc:
            raise AgentError("prompt_template", "prompt template is invalid") from exc

    @staticmethod
    def validate(definition: AgentDefinition):
        variables = {name: "" for name in definition.prompt.variables}
        PromptAssembler._render(definition.prompt.system_template, variables)
        PromptAssembler._render(definition.prompt.user_template, {**variables, "input": ""})

    @staticmethod
    def assemble(
        definition: AgentDefinition, value, context: ContextBundle, tools: list[dict]
    ) -> ModelRequest:
        SchemaContract(definition.input_schema).validate(value, code="agent_input")
        PromptAssembler.validate(definition)
        if not ContextProvider._valid_message_sequence(context.history):
            raise AgentError(
                "history_messages", "history contains invalid roles or tool observations"
            )
        variables = {
            name: data if isinstance(data, str) else canonical_json(data).decode()
            for name, data in definition.prompt.variables.items()
        }
        messages = [
            Message(
                role="system",
                content=PromptAssembler._render(definition.prompt.system_template, variables),
            )
        ]
        if context.memories:
            sources: list[dict[str, JsonValue]] = [
                {
                    "kind": "memory",
                    "id": record.id,
                    "agent_id": record.agent_id,
                    "conversation_id": record.conversation_id,
                    "version": record.version,
                }
                for record in context.memories
            ]
            messages.append(
                Message(
                    role="user",
                    content=canonical_json(
                        {"untrusted_memory_data": [record.content for record in context.memories]}
                    ).decode(),
                    sources=sources,
                )
            )
        messages.extend(context.history)
        serialized = value if isinstance(value, str) else canonical_json(value).decode()
        messages.append(
            Message(
                role="user",
                content=PromptAssembler._render(
                    definition.prompt.user_template, {**variables, "input": serialized}
                ),
            )
        )
        request = ModelRequest(
            messages=messages,
            tools=tools,
            output_schema=definition.output_schema,
            max_output_tokens=definition.budget.max_output_tokens,
        )
        assembled = canonical_json(
            request.model_dump(mode="json"),
            max_bytes=2097152 if definition.context.management is not None else 262144,
        )
        if (
            definition.context.management is None
            and len(assembled.decode()) > definition.budget.max_input_chars
        ):
            raise AgentError(
                "prompt_limit", "assembled model request character limit exceeded", 409
            )
        return request
