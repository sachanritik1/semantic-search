import { ApiErrorAlert } from "#/components/rag/ApiErrorAlert.tsx";
import { ResultPanel } from "#/components/rag/ResultPanel.tsx";
import { WorkspacePanel } from "#/components/rag/WorkspacePanel.tsx";
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
		<WorkspacePanel
			kicker="Write path"
			title="Upload PDF"
			description="Chunk, embed, and index a PDF. Only `.pdf` files are supported."
		>
			<form
				id="ingest-form"
				className="flex flex-1 flex-col gap-4"
				onSubmit={(event) => {
					event.preventDefault();
					void form.handleSubmit();
				}}
			>
				<form.AppForm>
					<FieldGroup className="flex-1">
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
					<form.SubmitButton label="Upload and index" className="w-fit" />
				</form.AppForm>

				{ingest.isError ? <ApiErrorAlert error={ingest.error} /> : null}

				{ingest.isSuccess ? (
					<ResultPanel title="Ingest complete">
						{`document_id: ${ingest.data.document_id}\nchunks_total: ${ingest.data.chunks_total}\nchunks_saved: ${ingest.data.chunks_saved}`}
					</ResultPanel>
				) : null}
			</form>
		</WorkspacePanel>
	);
}
