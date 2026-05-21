import { WorkspacePanel } from "#/components/rag/WorkspacePanel.tsx";
import { Badge } from "#/components/ui/badge.tsx";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select.tsx";
import type { DocumentSession } from "#/lib/session/documents.ts";

interface DocumentScopeBarProps {
	session: DocumentSession;
	onActiveDocumentChange: (documentId: string | null) => void;
}

function formatDocumentLabel(
	documentId: string,
	source?: string,
): string {
	if (source) {
		return `${source} (${documentId.slice(0, 8)}…)`;
	}
	return documentId;
}

export function DocumentScopeBar({
	session,
	onActiveDocumentChange,
}: DocumentScopeBarProps) {
	const { activeDocumentId, recent } = session;
	const hasDocuments = recent.length > 0;

	return (
		<WorkspacePanel
			kicker="Scope"
			title="Select document"
			description="Choose which ingested PDF to query. Questions only search this document."
		>
			{activeDocumentId ? (
				<div className="flex flex-wrap items-center gap-2">
					<Badge variant="secondary" className="font-mono text-xs">
						{activeDocumentId}
					</Badge>
				</div>
			) : (
				<p className="m-0 text-sm text-(--sea-ink-soft)">
					No document selected yet.
				</p>
			)}

			<div className="mt-auto flex flex-col gap-1.5">
				<span className="text-xs font-medium text-(--sea-ink-soft)">
					{hasDocuments ? "Recent documents" : "Documents"}
				</span>
				<Select
					value={activeDocumentId ?? undefined}
					disabled={!hasDocuments}
					onValueChange={(value) => onActiveDocumentChange(value || null)}
				>
					<SelectTrigger className="w-full">
						<SelectValue
							placeholder={
								hasDocuments
									? "Select a document…"
									: "Ingest a PDF to get started"
							}
						/>
					</SelectTrigger>
					<SelectContent position="popper">
						{recent.map((doc) => (
							<SelectItem key={doc.documentId} value={doc.documentId}>
								{formatDocumentLabel(doc.documentId, doc.source)}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			</div>
		</WorkspacePanel>
	);
}
