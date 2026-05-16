import { createFileRoute } from "@tanstack/react-router";
import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { ResultPanel } from "#/components/rag/ResultPanel.tsx";
import { Button } from "#/components/ui/button.tsx";
import { FieldGroup, FieldLabel } from "#/components/ui/field.tsx";
import {
	Tabs,
	TabsContent,
	TabsList,
	TabsTrigger,
} from "#/components/ui/tabs.tsx";
import { useAppForm } from "#/hooks/use-app-form.ts";
import {
	promptTestSchema,
	questionSchema,
	textSchema,
} from "#/lib/forms/schemas.ts";
import {
	useEnhanceMutation,
	useLlmTestMutation,
	usePromptTestMutation,
	useSelfConsistencyMutation,
	useTokenCountMutation,
} from "#/lib/api/hooks.ts";

export const Route = createFileRoute("/tools")({
	component: ToolsPage,
});

const PROMPT_TEMPLATES = [
	"qa_over_context.txt",
	"qa_cot.txt",
	"summarization.txt",
	"structured_extraction.txt",
] as const;

function EnhanceTab() {
	const enhance = useEnhanceMutation();

	const form = useAppForm({
		defaultValues: { question: "" },
		validators: { onSubmit: questionSchema },
		onSubmit: async ({ value }) => {
			await enhance.mutateAsync(value.question);
		},
	});

	return (
		<form
			className="space-y-4 pt-4"
			onSubmit={(event) => {
				event.preventDefault();
				void form.handleSubmit();
			}}
		>
			<form.AppForm>
				<FieldGroup>
					<form.AppField name="question">
						{(field) => <field.TextAreaField label="Question" />}
					</form.AppField>
				</FieldGroup>
				<form.SubmitButton label="Enhance" />
			</form.AppForm>
			{enhance.isError ? <ApiErrorAlert error={enhance.error} /> : null}
			{enhance.isSuccess ? (
				<div className="grid gap-4 sm:grid-cols-2">
					<ResultPanel title="Original">{enhance.data.original}</ResultPanel>
					<ResultPanel title="Enhanced">{enhance.data.enhanced}</ResultPanel>
				</div>
			) : null}
		</form>
	);
}

function TokensTab() {
	const tokens = useTokenCountMutation();

	const form = useAppForm({
		defaultValues: { text: "" },
		validators: { onSubmit: textSchema },
		onSubmit: async ({ value }) => {
			await tokens.mutateAsync(value.text);
		},
	});

	return (
		<form
			className="space-y-4 pt-4"
			onSubmit={(event) => {
				event.preventDefault();
				void form.handleSubmit();
			}}
		>
			<form.AppForm>
				<FieldGroup>
					<form.AppField name="text">
						{(field) => (
							<field.TextAreaField label="Text" rows={6} />
						)}
					</form.AppField>
				</FieldGroup>
				<form.SubmitButton label="Count tokens" />
			</form.AppForm>
			{tokens.isError ? <ApiErrorAlert error={tokens.error} /> : null}
			{tokens.isSuccess ? (
				<div className="space-y-4">
					<ResultPanel title="Token count">
						{String(tokens.data.token_count)}
					</ResultPanel>
					<details className="island-shell text-sm">
						<summary className="cursor-pointer font-medium">
							Token IDs ({tokens.data.tokens.length})
						</summary>
						<p className="mt-2 font-mono text-xs break-all">
							{tokens.data.tokens.join(", ")}
						</p>
					</details>
				</div>
			) : null}
		</form>
	);
}

function PromptTab() {
	const promptTest = usePromptTestMutation();

	const form = useAppForm({
		defaultValues: {
			template: PROMPT_TEMPLATES[0],
			variables: [{ key: "context", value: "" }],
		},
		validators: { onSubmit: promptTestSchema },
		onSubmit: async ({ value }) => {
			const variables = Object.fromEntries(
				value.variables
					.filter((entry) => entry.key.trim())
					.map((entry) => [entry.key.trim(), entry.value]),
			);
			await promptTest.mutateAsync({
				template: value.template,
				variables,
			});
		},
	});

	return (
		<form
			className="space-y-4 pt-4"
			onSubmit={(event) => {
				event.preventDefault();
				void form.handleSubmit();
			}}
		>
			<form.AppForm>
				<FieldGroup>
					<form.AppField name="template">
						{(field) => (
							<field.SelectField
								label="Template"
								options={PROMPT_TEMPLATES}
								description="Template file name including .txt extension."
							/>
						)}
					</form.AppField>

					<form.Field name="variables" mode="array">
						{(field) => (
							<div className="space-y-3">
								<div className="flex items-center justify-between gap-2">
									<FieldLabel>Variables</FieldLabel>
									<Button
										type="button"
										variant="outline"
										size="sm"
										onClick={() => field.pushValue({ key: "", value: "" })}
									>
										Add variable
									</Button>
								</div>
								{field.state.value.map((_, index) => (
									<div
										key={`variable-${index}`}
										className="grid gap-2 sm:grid-cols-2"
									>
										<form.AppField name={`variables[${index}].key`}>
											{(entry) => (
												<entry.TextField label="Key" placeholder="context" />
											)}
										</form.AppField>
										<form.AppField name={`variables[${index}].value`}>
											{(entry) => (
												<entry.TextField label="Value" placeholder="..." />
											)}
										</form.AppField>
									</div>
								))}
							</div>
						)}
					</form.Field>
				</FieldGroup>
				<form.SubmitButton label="Run prompt" />
			</form.AppForm>
			{promptTest.isError ? <ApiErrorAlert error={promptTest.error} /> : null}
			{promptTest.isSuccess ? (
				promptTest.data.error ? (
					<ApiErrorAlert error={new Error(promptTest.data.error)} />
				) : (
					<ResultPanel title="Response">
						{promptTest.data.response ?? ""}
					</ResultPanel>
				)
			) : null}
		</form>
	);
}

function LlmTab() {
	const llmTest = useLlmTestMutation();

	const form = useAppForm({
		defaultValues: { question: "" },
		validators: { onSubmit: questionSchema },
		onSubmit: async ({ value }) => {
			await llmTest.mutateAsync(value.question);
		},
	});

	return (
		<form
			className="space-y-4 pt-4"
			onSubmit={(event) => {
				event.preventDefault();
				void form.handleSubmit();
			}}
		>
			<form.AppForm>
				<FieldGroup>
					<form.AppField name="question">
						{(field) => <field.TextAreaField label="Question" />}
					</form.AppField>
				</FieldGroup>
				<form.SubmitButton label="Send to LLM" />
			</form.AppForm>
			{llmTest.isError ? <ApiErrorAlert error={llmTest.error} /> : null}
			{llmTest.isSuccess ? (
				<div className="space-y-4">
					<ResultPanel title="Content">
						{llmTest.data.response.content}
					</ResultPanel>
					<p className="text-sm text-[var(--sea-ink-soft)]">
						Model: {llmTest.data.response.model ?? "unknown"}
					</p>
				</div>
			) : null}
		</form>
	);
}

function SelfConsistencyTab() {
	const selfConsistency = useSelfConsistencyMutation();

	const form = useAppForm({
		defaultValues: { question: "" },
		validators: { onSubmit: questionSchema },
		onSubmit: async ({ value }) => {
			await selfConsistency.mutateAsync(value.question);
		},
	});

	return (
		<form
			className="space-y-4 pt-4"
			onSubmit={(event) => {
				event.preventDefault();
				void form.handleSubmit();
			}}
		>
			<form.AppForm>
				<FieldGroup>
					<form.AppField name="question">
						{(field) => <field.TextAreaField label="Question" />}
					</form.AppField>
				</FieldGroup>
				<p className="text-sm text-[var(--sea-ink-soft)]">
					Runs 5 parallel samples and picks a majority answer. May take a minute.
				</p>
				<form.SubmitButton label="Run self-consistency" />
			</form.AppForm>
			{selfConsistency.isError ? (
				<ApiErrorAlert error={selfConsistency.error} />
			) : null}
			{selfConsistency.isSuccess ? (
				<ResultPanel title="Final answer">
					{selfConsistency.data.final_answer}
				</ResultPanel>
			) : null}
		</form>
	);
}

function ToolsPage() {
	return (
		<div className="page-wrap py-10">
			<div className="mb-6 space-y-2">
				<p className="island-kicker m-0">Debug</p>
				<h1 className="m-0 text-2xl font-semibold text-[var(--sea-ink)]">Tools</h1>
				<p className="m-0 text-sm text-[var(--sea-ink-soft)]">
					Standalone API utilities for experimenting with prompts and models.
				</p>
			</div>

			<Tabs defaultValue="enhance" className="island-shell px-4 pb-4">
				<TabsList>
					<TabsTrigger value="enhance">Enhance</TabsTrigger>
					<TabsTrigger value="tokens">Tokens</TabsTrigger>
					<TabsTrigger value="prompt">Prompt</TabsTrigger>
					<TabsTrigger value="llm">LLM test</TabsTrigger>
					<TabsTrigger value="consistency">Self-consistency</TabsTrigger>
				</TabsList>
				<TabsContent value="enhance">
					<EnhanceTab />
				</TabsContent>
				<TabsContent value="tokens">
					<TokensTab />
				</TabsContent>
				<TabsContent value="prompt">
					<PromptTab />
				</TabsContent>
				<TabsContent value="llm">
					<LlmTab />
				</TabsContent>
				<TabsContent value="consistency">
					<SelfConsistencyTab />
				</TabsContent>
			</Tabs>
		</div>
	);
}
