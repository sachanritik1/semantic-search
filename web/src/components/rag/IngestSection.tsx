import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { ResultPanel } from "#/components/rag/ResultPanel.tsx";
import { FieldGroup } from "#/components/ui/field.tsx";
import { useAppForm } from "#/hooks/use-app-form.ts";
import { useIngestMutation } from "#/lib/api/hooks.ts";
import type { IngestResponse } from "#/lib/api/types.ts";
import { ingestSchema } from "#/lib/forms/schemas.ts";

interface IngestSectionProps {
	onIngested: (result: IngestResponse, source?: string) => void;
}

export function IngestSection({ onIngested }: IngestSectionProps) {
	const ingest = useIngestMutation();

	const form = useAppForm({
		defaultValues: { file: null as File | null },
		validators: { onSubmit: ingestSchema },
		onSubmit: async ({ value }) => {
			const file = value.file;
			if (!file) {
				return;
			}
			const result = await ingest.mutateAsync(file);
			onIngested(result, file.name);
		},
	});

	return (
		<section className="space-y-4">
			<div className="space-y-2">
				<p className="island-kicker m-0">Write path</p>
				<h2 className="m-0 text-xl font-semibold text-(--sea-ink)">Ingest PDF</h2>
				<p className="m-0 text-sm text-(--sea-ink-soft)">
					Upload a PDF to chunk, embed, and index. Only `.pdf` files are supported.
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
		</section>
	);
}
