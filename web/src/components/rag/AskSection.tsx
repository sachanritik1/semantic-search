import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { AnswerContent } from "#/components/rag/AnswerContent.tsx";
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
		<section className="space-y-4">
			<div className="space-y-2">
				<p className="island-kicker m-0">Query path</p>
				<h2 className="m-0 text-xl font-semibold text-(--sea-ink)">Ask a question</h2>
				<p className="m-0 text-sm text-(--sea-ink-soft)">
					Runs enhance → dense + sparse retrieval → rerank → generate. This may
					take 10–30 seconds.
				</p>
			</div>

			{!canAsk ? (
				<p className="m-0 text-sm text-(--sea-ink-soft)">
					Ingest a PDF first to enable questions on that document.
				</p>
			) : null}

			<form
				id="ask-form"
				className="island-shell max-w-2xl space-y-4"
				onSubmit={(event) => {
					event.preventDefault();
					void form.handleSubmit();
				}}
			>
				<form.AppForm>
					<FieldGroup>
						<form.AppField name="question">
							{(field) => (
								<field.TextAreaField
									label="Question"
									placeholder="What are the main findings in the document?"
								/>
							)}
						</form.AppField>
					</FieldGroup>
					<form.SubmitButton label="Ask" disabled={!canAsk} />
				</form.AppForm>

				{ask.isError ? <ApiErrorAlert error={ask.error} /> : null}

				{ask.isSuccess ? (
					<div className="space-y-4">
						<AnswerContent content={ask.data.response} />
						<details className="island-shell text-sm text-(--sea-ink-soft)">
							<summary className="cursor-pointer font-medium text-(--sea-ink)">
								Query enhancement
							</summary>
							<div className="mt-3 space-y-2">
								<p className="m-0">
									<span className="font-medium">Original:</span>{" "}
									{ask.data.original_question}
								</p>
								<p className="m-0">
									<span className="font-medium">Enhanced:</span>{" "}
									{ask.data.enhanced_question}
								</p>
							</div>
						</details>
					</div>
				) : null}
			</form>
		</section>
	);
}
