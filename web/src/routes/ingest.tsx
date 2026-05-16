import { createFileRoute } from "@tanstack/react-router";
import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { ResultPanel } from "#/components/rag/ResultPanel.tsx";
import { FieldGroup } from "#/components/ui/field.tsx";
import { useAppForm } from "#/hooks/use-app-form.ts";
import { ingestSchema } from "#/lib/forms/schemas.ts";
import { useIngestMutation } from "#/lib/api/hooks.ts";

export const Route = createFileRoute("/ingest")({
	component: IngestPage,
});

function IngestPage() {
	const ingest = useIngestMutation();

	const form = useAppForm({
		defaultValues: { file: null as File | null },
		validators: { onSubmit: ingestSchema },
		onSubmit: async ({ value }) => {
			await ingest.mutateAsync(value.file);
		},
	});

	return (
		<div className="page-wrap py-10">
			<div className="mb-6 space-y-2">
				<p className="island-kicker m-0">Write path</p>
				<h1 className="m-0 text-2xl font-semibold text-[var(--sea-ink)]">
					Ingest PDF
				</h1>
				<p className="m-0 text-sm text-[var(--sea-ink-soft)]">
					Upload a PDF to chunk, embed, and index. Only `.pdf` files are
					supported.
				</p>
			</div>

			<form
				id="ingest-form"
				className="island-shell max-w-xl space-y-4"
				onSubmit={(event) => {
					event.preventDefault();
					void form.handleSubmit();
				}}
			>
				<form.AppForm>
					<FieldGroup>
						<form.AppField name="file">
							{(field) => (
								<field.FileField
									label="PDF file"
									accept=".pdf,application/pdf"
									description="Only PDF documents are indexed."
								/>
							)}
						</form.AppField>
					</FieldGroup>
					<form.SubmitButton label="Upload and index" />
				</form.AppForm>

				{ingest.isError ? <ApiErrorAlert error={ingest.error} /> : null}

				{ingest.isSuccess ? (
					<ResultPanel title="Ingest complete">
						{`document_id: ${ingest.data.document_id}\nchunks_total: ${ingest.data.chunks_total}\nchunks_saved: ${ingest.data.chunks_saved}`}
					</ResultPanel>
				) : null}
			</form>
		</div>
	);
}
