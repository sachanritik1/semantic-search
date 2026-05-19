import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { AnswerContent } from "#/components/rag/AnswerContent.tsx";
import { WorkspacePanel } from "#/components/rag/WorkspacePanel.tsx";
import { FieldGroup } from "#/components/ui/field.tsx";
import { useAppForm } from "#/hooks/use-app-form.ts";
import { useAskMutation } from "#/lib/api/hooks.ts";
import { questionSchema } from "#/lib/forms/schemas.ts";

interface AskSectionProps {
	documentId: string | null;
}

export function AskSection({ documentId }: AskSectionProps) {
	const ask = useAskMutation();
	const canAsk = Boolean(documentId);

	const form = useAppForm({
		defaultValues: { question: "" },
		validators: { onSubmit: questionSchema },
		onSubmit: async ({ value }) => {
			if (!documentId) {
				return;
			}
			await ask.mutateAsync({
				question: value.question,
				documentId,
			});
		},
	});

	return (
		<section className="space-y-3">
			<p className="island-kicker m-0">Query path</p>
			<div className="grid gap-4 lg:grid-cols-2 lg:items-stretch">
				<WorkspacePanel
					title="Your question"
					description="Runs enhance → retrieve → rerank → generate. May take 10–30 seconds."
					className="min-h-[22rem]"
				>
					{!canAsk ? (
						<p className="m-0 text-sm text-(--sea-ink-soft)">
							Ingest a PDF and select it above before asking.
						</p>
					) : null}

					<form
						id="ask-form"
						className="flex flex-1 flex-col gap-4"
						onSubmit={(event) => {
							event.preventDefault();
							void form.handleSubmit();
						}}
					>
						<form.AppForm>
							<FieldGroup className="flex-1 gap-4">
								<form.AppField name="question">
									{(field) => (
										<field.TextAreaField
											label="Question"
											placeholder="What are the main findings in the document?"
											rows={6}
										/>
									)}
								</form.AppField>
							</FieldGroup>
							<form.SubmitButton label="Ask" disabled={!canAsk} className="w-fit" />
						</form.AppForm>

						{ask.isError ? <ApiErrorAlert error={ask.error} /> : null}
					</form>
				</WorkspacePanel>

				<WorkspacePanel
					title="Answer"
					description="Grounded response from the selected document."
					className="min-h-[22rem]"
				>
					{ask.isPending ? (
						<p className="m-0 text-sm text-(--sea-ink-soft)">Generating answer…</p>
					) : null}

					{ask.isSuccess ? (
						<div className="flex flex-1 flex-col gap-4 overflow-hidden">
							<AnswerContent content={ask.data.response} embedded />
							<details className="shrink-0 rounded-lg border border-(--line) bg-(--chip-bg)/50 px-3 py-2 text-sm text-(--sea-ink-soft)">
								<summary className="cursor-pointer font-medium text-(--sea-ink)">
									Query enhancement
								</summary>
								<div className="mt-2 space-y-2">
									<p className="m-0">
										<span className="font-medium">Original:</span>{" "}
										{ask.data.original_question}
									</p>
									{ask.data.enhanced_questions &&
									ask.data.enhanced_questions.length > 0 ? (
										<div className="m-0">
											<p className="m-0 font-medium">Enhanced queries:</p>
											<ul className="mt-1 list-inside list-disc space-y-1">
												{ask.data.enhanced_questions.map((q) => (
													<li key={q}>{q}</li>
												))}
											</ul>
										</div>
									) : (
										<p className="m-0">
											<span className="font-medium">Enhanced:</span>{" "}
											{ask.data.enhanced_question}
										</p>
									)}
								</div>
							</details>
						</div>
					) : null}

					{!ask.isPending && !ask.isSuccess && !ask.isError ? (
						<p className="m-0 text-sm text-(--sea-ink-soft)">
							Submit a question to see the answer here.
						</p>
					) : null}
				</WorkspacePanel>
			</div>
		</section>
	);
}
