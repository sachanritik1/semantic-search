import { Badge } from "#/components/ui/badge.tsx";
import { WorkspacePanel } from "#/components/rag/WorkspacePanel.tsx";
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

			<label className="mt-auto flex flex-col gap-1.5">
				<span className="text-xs font-medium text-(--sea-ink-soft)">
					{recent.length > 0 ? "Recent documents" : "Documents"}
				</span>
				<select
					className="w-full rounded-md border border-(--line) bg-transparent px-3 py-2 text-sm text-(--sea-ink) outline-none focus-visible:border-(--lagoon) disabled:cursor-not-allowed disabled:opacity-60"
					value={activeDocumentId ?? ""}
					disabled={recent.length === 0}
					onChange={(event) => {
						const value = event.target.value;
						onActiveDocumentChange(value || null);
					}}
				>
					<option value="">
						{recent.length > 0
							? "Select a document…"
							: "Ingest a PDF to get started"}
					</option>
					{recent.map((doc) => (
						<option key={doc.documentId} value={doc.documentId}>
							{formatDocumentLabel(doc.documentId, doc.source)}
						</option>
					))}
				</select>
			</label>
		</WorkspacePanel>
	);
}
